"""PDF -> Book/Chapter/Section hierarchy, using the embedded table of contents.

Chapters come from level-1 TOC entries. When a PDF has no embedded
outline, chapters fall back -- in order -- to an LLM-extracted table of
contents (only when an LLM client and DB connection are supplied) parsed
from the book's own printed TOC pages, and only after that to a regex
heading scan. The LLM path exists because heading regexes alone are too
weak for books like "Professional C++" (Gregoire): the real chapter
headings don't match, while literal "CHAPTER N:" strings quoted inline in
an appendix do, promoting body text into false chapter boundaries. See
ARCHITECTURE.md for the full fallback chain and offset-resolution
rationale. Each chapter is then split into sections, in strict priority
order:

1. TOC sub-entries (level >= 2) that fall inside the chapter's page range.
2. If there are none and the chapter is longer than `section_split_tokens`,
   auto-split into consecutive ~`section_split_tokens`-token sections,
   breaking on paragraph boundaries where possible.
3. Otherwise, the whole chapter is a single section.

A chunk is later built from one section's text only, so it can never cross
a section boundary -- the hierarchy (chapter -> section -> chunk) always
exists, even for a flat chapter with no sub-headings at all.

Cleans repeated running headers/footers and stray page numbers before any
of the above, so page-range slicing operates on clean text.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

from .. import db
from ..ollama_client import OllamaClient
from .common import Chapter, ExtractedBook, Section, content_hash_of

logger = logging.getLogger(__name__)

_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?[ivxlcdm\d]+\s*$", re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_HEADING_RE = re.compile(r"^\s*(?:chapter|part)\s+[ivxlcdm\d]+\b", re.IGNORECASE)
_PARA_MARKER = "\x00"
_CHARS_PER_TOKEN = 4

# -- LLM-extracted TOC fallback (no embedded outline) ------------------------
_TOC_HEADER_RE = re.compile(r"^\s*(?:table of contents|contents)\s*$", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"^.{3,}?\D[\s.…]+\d{1,4}\s*$")
_TOP_LEVEL_TOC_ENTRY_RE = re.compile(
    r"^\s*(?:part\s+[ivxlcdm\d]+|chapter\s+\d+|appendix\s+[a-z])\b", re.IGNORECASE
)
_TOC_FRONT_MATTER_SCAN_LIMIT = 50
_TOC_LINE_DENSITY_THRESHOLD = 0.3
_MAX_TOC_REGION_PAGES = 50
_TOC_LLM_BATCH_PAGES = 2
_MAX_SANE_CHAPTER_COUNT = 100
# A genuine opener page reliably scores >= 0.938 (exact substring 1.0, bag
# of words 0.95, or -- for a long title that pushes "WHAT'S IN THIS
# CHAPTER?" out of `_page_top_text`'s window, losing the bag tier's lone
# "chapter" word -- a fuzzy ratio that still lands in the high 0.9s, since
# the rest of the title matches verbatim). A threshold of 0.6 let a
# coincidentally similar pair of titles (e.g. two consecutive chapters
# both ending in "...Classes and Objects") cross it via fuzzy ratio alone
# on the wrong page, well below where any genuine match has ever landed --
# confirmed concretely on the Gregoire "Professional C++" PDF.
_TITLE_MATCH_THRESHOLD = 0.85
_OFFSET_SEARCH_RADII = (5, 15, 40)

_TOC_SYSTEM_PROMPT = (
    "You extract a book's top-level table-of-contents entries from raw "
    "text. Respond with ONLY a JSON array, no prose, no markdown fences, "
    "no explanation. Each element must be an object with exactly two "
    'keys: "title" (string) and "declared_page" (integer -- the page '
    "number printed next to that entry in the table of contents). Include "
    "only top-level entries (chapters, parts, or appendices) in their "
    "original reading order. Skip indented sub-entries (sections, "
    "subsections) and skip front-matter entries such as Preface, "
    "Acknowledgments, or Index."
)


class PdfParser:
    source_type = "pdf"
    extensions = (".pdf",)

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def extract(
        self,
        path: Path,
        section_split_tokens: int = 3000,
        conn: sqlite3.Connection | None = None,
        llm_client: OllamaClient | None = None,
        llm_model: str | None = None,
    ) -> ExtractedBook:
        return extract_book(str(path), section_split_tokens, conn, llm_client, llm_model)


def extract_book(
    pdf_path: str,
    section_split_tokens: int = 3000,
    conn: sqlite3.Connection | None = None,
    llm_client: OllamaClient | None = None,
    llm_model: str | None = None,
) -> ExtractedBook:
    """Extract the full chapter/section hierarchy, content hash, and metadata from a PDF.

    `conn`/`llm_client`/`llm_model` are optional and only used by the
    LLM-TOC fallback (see module docstring); omitting them simply skips
    that branch, preserving the exact behavior of every caller that
    extracted a PDF before this fallback existed.
    """
    content_hash = content_hash_of(Path(pdf_path))

    doc = pymupdf.open(pdf_path)
    try:
        page_count = doc.page_count
        raw_pages_text = [page.get_text("text") for page in doc]
        raw_pages_text = _strip_running_headers_footers(raw_pages_text)
        pages_text = [_clean_text(t) for t in raw_pages_text]

        toc = doc.get_toc(simple=True)
        chapter_bounds = _chapter_bounds_from_toc(toc, page_count)
        if chapter_bounds is None:
            chapter_bounds = _llm_toc_chapter_bounds(
                raw_pages_text, page_count, content_hash, conn, llm_client, llm_model
            )
        if chapter_bounds is None:
            chapter_bounds = _chapter_bounds_from_headings(pages_text)

        chapters = []
        for chapter_index, (title, page_start, page_end) in enumerate(chapter_bounds):
            sub_entries = _sub_entries_for_chapter(toc, page_start, page_end)
            sections = _derive_sections(
                sub_entries, pages_text, page_start, page_end, section_split_tokens
            )
            chapters.append(Chapter(chapter_index, title, page_start, sections))

        metadata = doc.metadata or {}
        title = (metadata.get("title") or "").strip() or None
        author = (metadata.get("author") or "").strip() or None
        return ExtractedBook(
            content_hash=content_hash,
            page_count=page_count,
            chapters=chapters,
            title=title,
            author=author,
            source_type="pdf",
        )
    finally:
        doc.close()


# -- chapter boundary detection -----------------------------------------------


def _chapter_bounds_from_toc(toc: list, page_count: int) -> list[tuple[str, int, int]] | None:
    top = [(title.strip(), page) for level, title, page in toc if level == 1]
    if not top:
        return None
    bounds = []
    for i, (title, page) in enumerate(top):
        page_start = max(1, page)
        page_end = top[i + 1][1] - 1 if i + 1 < len(top) else page_count
        page_end = max(page_start, min(page_end, page_count))
        bounds.append((title, page_start, page_end))
    return bounds


def _chapter_bounds_from_headings(pages_text: list[str]) -> list[tuple[str, int, int]]:
    """Heuristic fallback: treat lines matching 'Chapter N' / 'Part N' as boundaries."""
    n_pages = len(pages_text)
    boundaries: list[tuple[str, int]] = []
    for page_idx, text in enumerate(pages_text, start=1):
        for line in text.split("\n")[:3]:
            if _HEADING_RE.match(line.strip()):
                boundaries.append((line.strip(), page_idx))
                break

    if not boundaries:
        return [("Full Document", 1, n_pages)]

    bounds = []
    for i, (title, page_start) in enumerate(boundaries):
        page_end = boundaries[i + 1][1] - 1 if i + 1 < len(boundaries) else n_pages
        page_end = max(page_start, page_end)
        bounds.append((title, page_start, page_end))
    return bounds


# -- LLM-extracted TOC fallback (no embedded outline) -------------------------


def _llm_toc_chapter_bounds(
    raw_pages_text: list[str],
    page_count: int,
    content_hash: str,
    conn: sqlite3.Connection | None,
    llm_client: OllamaClient | None,
    llm_model: str | None,
) -> list[tuple[str, int, int]] | None:
    """LLM-extracted, offset-resolved TOC -- second in the fallback chain, before headings.

    Cached by content_hash (via `conn`) so re-ingesting the same PDF never
    re-calls the LLM. Returns None (deferring to heading detection) if no
    `conn`/`llm_client` was supplied, no text-based TOC region is found, or
    the LLM's output fails validation at any stage -- this branch never
    hands back an unvalidated result.

    Both TOC *detection* and offset *resolution* below use `raw_pages_text`
    (per-line structure intact, only header/footer-stripped), never the
    `_clean_text`-flattened text: the line-by-line "<title> ... <page>"
    pattern TOC detection relies on disappears once `_clean_text` collapses
    single newlines into spaces, and so does `_page_top_text`'s "first few
    lines" guarantee that resolution depends on for title-matching against
    real chapter-start pages -- once flattened, a page's "lines" become
    whole paragraphs, which can swallow a page's entire text.
    """
    if conn is not None:
        cached = db.get_cached_pdf_toc(conn, content_hash)
        if cached is not None:
            return cached

    if llm_client is None or llm_model is None:
        return None

    toc_region = _detect_toc_pages(raw_pages_text)
    if toc_region is None:
        logger.warning("No text-based TOC region detected; deferring to heading detection.")
        return None

    entries = _extract_toc_entries_in_batches(raw_pages_text, toc_region, llm_client, llm_model)
    if not _validate_declared_entries(entries):
        return None

    search_start = min(toc_region[1] + 1, page_count)
    chapter_bounds = _resolve_chapter_pages(entries, raw_pages_text, page_count, search_start)
    if chapter_bounds is None:
        logger.warning(
            "Could not resolve physical pages for the LLM-extracted TOC; "
            "deferring to heading detection."
        )
        return None
    if not _validate_resolved_bounds(chapter_bounds, page_count):
        return None

    if conn is not None:
        db.set_cached_pdf_toc(conn, content_hash, chapter_bounds)
    return chapter_bounds


def _extract_toc_entries_in_batches(
    raw_pages_text: list[str],
    toc_region: tuple[int, int],
    llm_client: OllamaClient,
    llm_model: str,
) -> list[tuple[str, int]]:
    """Extract TOC entries in small, fixed-size, non-overlapping page batches.

    A single call asking an 8B model to enumerate every entry of a long,
    repetitive printed TOC (dozens of similar-looking lines) tends to
    sample a handful of representative entries instead of listing all of
    them -- proven empirically on the Gregoire "Professional C++" PDF's
    32-page TOC region, where one call returned 11 of the real 38 entries.
    Shrinking the batch size keeps lowering the miss rate (6-page batches
    silently dropped every heading after a chapter with a long sub-entry
    list filled most of the batch; 3-page batches still missed several
    headings scattered across different chapters), but it never reaches
    zero misses, and going below 2 pages stopped helping -- 1-page batches
    missed a *different* couple of entries instead of fewer. 2 pages is
    the empirically best point found; a handful of dropped headings can
    still get through (see `_validate_resolved_bounds` and the
    all-or-nothing rejection in `_resolve_chapter_pages`'s caller for the
    backstop), so this is a mitigation, not a guarantee.

    Batches are non-overlapping, so the same heading is never sent to two
    calls -- a batch boundary can only ever split a "Part" header from its
    first chapter (each is its own top-level entry, extracted correctly on
    either side), never a single entry's own line. The one remaining edge
    case -- a model echoing the heading nearest a cut edge -- is guarded
    against by dropping an exact duplicate title at a batch boundary. A
    batch whose response fails to parse is logged and skipped rather than
    failing the whole extraction; the final defensive checks this feeds
    into (`_validate_declared_entries`, and physical-page title-matching in
    `_resolve_chapter_pages`) still reject the merged result outright if
    anything is structurally wrong, falling back to heading detection.

    A small model asked to "skip indented sub-entries" does so unreliably
    once a chapter's printed TOC nests several levels deep (e.g. a chapter
    -> section -> sub-section list, as in the Gregoire book's design
    patterns and templates chapters) -- some batches comply, others dump
    every sub-heading too. `_is_top_level_toc_entry` re-checks each entry
    against the same structural pattern ("Part"/"Chapter"/"Appendix" plus a
    number or letter) the prompt already asked for, so pollution is caught
    deterministically rather than trusted to the model's judgment alone.

    A batch whose response fails to parse, or whose parsed entries fall
    short of the headings structurally present in its raw text, is retried
    at a smaller page granularity (see `_extract_batch_with_retry`) rather
    than accepted as-is -- confirmed concretely on the Gregoire PDF's pages
    20-21 (malformed JSON from ~75 sub-entries crammed into one call) and
    pages 36-37 (well-formed JSON that simply omitted "CHAPTER 30").
    """
    region_start, region_end = toc_region
    entries: list[tuple[str, int]] = []
    for batch_start in range(region_start, region_end + 1, _TOC_LLM_BATCH_PAGES):
        batch_end = min(batch_start + _TOC_LLM_BATCH_PAGES - 1, region_end)
        batch_entries = _extract_batch_with_retry(
            raw_pages_text, batch_start, batch_end, llm_client, llm_model
        )
        if (
            entries
            and batch_entries
            and _normalize_title(entries[-1][0]) == _normalize_title(batch_entries[0][0])
        ):
            batch_entries = batch_entries[1:]
        entries.extend(batch_entries)
    return entries


def _extract_batch_with_retry(
    raw_pages_text: list[str],
    page_start: int,
    page_end: int,
    llm_client: OllamaClient,
    llm_model: str,
) -> list[tuple[str, int]]:
    """One LLM call over a page range; bisect and retry if the response looks incomplete.

    Two distinct failure modes get the same bisect-and-retry treatment. A
    page span dense enough with sub-entries can run the model's output
    past whatever it was going to emit before closing the JSON array,
    producing malformed JSON for the *whole* span even though either half,
    asked alone, would have parsed fine (confirmed on pages 20-21).
    Separately, a well-formed response can still just silently omit one
    real top-level heading -- confirmed on pages 36-37, where "CHAPTER 30:
    BECOMING ADEPT AT TESTING" was dropped while the neighboring "CHAPTER
    29" came through correctly in the very same response. The second case
    is caught by `_count_structural_top_level_lines`: a real heading always
    starts its own line in the raw text regardless of whether the model
    found it, so a shortfall against that count is a reliable signal of
    under-extraction even when the JSON itself parsed fine.

    A single page that's still short of its structural count has nowhere
    smaller to retry, so whatever it returns is accepted as final -- this
    happens for "Part" divider headings, which often have no page number
    of their own in the printed TOC (it shares its first chapter's number
    instead) and so get dropped by the model somewhat unpredictably
    regardless of how much surrounding context it's given.
    """
    batch_text = _join_pages(raw_pages_text, page_start, page_end)
    if not batch_text.strip():
        return []
    raw_response = llm_client.generate(llm_model, _TOC_SYSTEM_PROMPT, batch_text)
    batch_entries = _parse_llm_toc_response(raw_response)
    if batch_entries is not None:
        batch_entries = [e for e in batch_entries if _is_top_level_toc_entry(e[0])]

    if page_start < page_end:
        if batch_entries is None:
            logger.warning(
                "TOC batch (pages %d-%d) returned invalid JSON; retrying as smaller batches.",
                page_start,
                page_end,
            )
        else:
            expected = _count_structural_top_level_lines(batch_text)
            if len(batch_entries) >= expected:
                return batch_entries
            logger.warning(
                "TOC batch (pages %d-%d) returned %d top-level entries but %d are "
                "structurally present in the raw text; retrying as smaller batches.",
                page_start,
                page_end,
                len(batch_entries),
                expected,
            )
        midpoint = (page_start + page_end) // 2
        first_half = _extract_batch_with_retry(
            raw_pages_text, page_start, midpoint, llm_client, llm_model
        )
        second_half = _extract_batch_with_retry(
            raw_pages_text, midpoint + 1, page_end, llm_client, llm_model
        )
        if (
            first_half
            and second_half
            and _normalize_title(first_half[-1][0]) == _normalize_title(second_half[0][0])
        ):
            second_half = second_half[1:]
        return first_half + second_half

    if batch_entries is None:
        logger.warning(
            "TOC batch (page %d) returned invalid JSON; skipping this page.",
            page_start,
        )
        return []
    return batch_entries


def _detect_toc_pages(pages_text: list[str]) -> tuple[int, int] | None:
    """Find the contiguous run of leading pages that look like a printed TOC.

    Anchors on an explicit "Contents"/"Table of Contents" header within the
    front matter, then keeps including subsequent pages while either of two
    independent signals holds:

    1. A sizeable fraction of the page's lines look like
       "<title> ... <page number>" -- the original heuristic, which is all
       a short/flat TOC ever shows.
    2. The page repeats the same "Contents"/"Table of Contents" header that
       anchored the region. Some books (e.g. Gregoire's "Professional C++")
       print a page number only next to top-level entries and never next
       to the (far more numerous) indented sub-entries, so signal 1 alone
       collapses to ~0 a page or two into a TOC that is still very much
       running -- but the repeating header survives `_clean_text` and
       `_strip_running_headers_footers` (which only strips a line if it
       repeats across at least half of *all* pages in the document, far
       more than a multi-page TOC ever spans) and stays a reliable signal
       for exactly as long as the TOC itself runs.

    Extension stops at the first page where neither signal holds, i.e. the
    first real content page. The page cap is a runaway-match safety valve,
    not an assumption about how long any given book's TOC actually is.
    """
    scan_limit = min(len(pages_text), _TOC_FRONT_MATTER_SCAN_LIMIT)
    anchor = None
    for page_idx in range(scan_limit):
        if _has_toc_header(pages_text[page_idx]):
            anchor = page_idx
            break
    if anchor is None:
        return None

    end = anchor
    region_limit = min(len(pages_text), anchor + _MAX_TOC_REGION_PAGES)
    for page_idx in range(anchor + 1, region_limit):
        page_text = pages_text[page_idx]
        if (
            _toc_line_density(page_text) < _TOC_LINE_DENSITY_THRESHOLD
            and not _has_toc_header(page_text)
        ):
            break
        end = page_idx
    return (anchor + 1, end + 1)  # 1-indexed, inclusive page range


def _has_toc_header(page_text: str) -> bool:
    lines = (line.strip() for line in page_text.split("\n"))
    return any(_TOC_HEADER_RE.match(line) for line in lines if line)


def _is_top_level_toc_entry(title: str) -> bool:
    return bool(_TOP_LEVEL_TOC_ENTRY_RE.match(title))


def _count_structural_top_level_lines(text: str) -> int:
    """How many raw lines structurally look like a top-level TOC entry.

    A deterministic count, independent of the LLM, used to tell a
    genuinely short batch apart from one the model under-extracted: every
    real "Part"/"Chapter"/"Appendix" heading starts its own line in the
    raw text regardless of whether the model found it.
    """
    return sum(1 for line in text.split("\n") if _TOP_LEVEL_TOC_ENTRY_RE.match(line.strip()))


def _toc_line_density(page_text: str) -> float:
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    matches = sum(1 for line in lines if _TOC_LINE_RE.match(line))
    return matches / len(lines)


def _parse_llm_toc_response(raw: str) -> list[tuple[str, int]] | None:
    """Defensively parse the LLM's JSON TOC response: strip fences, skip malformed entries."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None

    entries = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        try:
            page = int(item.get("declared_page"))
        except (TypeError, ValueError):
            continue
        entries.append((title.strip(), page))
    return entries


def _validate_declared_entries(entries: list[tuple[str, int]]) -> bool:
    """Reject the raw LLM TOC outright if it's empty, absurdly long, or out of page order."""
    if not entries:
        logger.warning("LLM TOC extraction returned 0 entries.")
        return False
    if len(entries) > _MAX_SANE_CHAPTER_COUNT:
        logger.warning(
            "LLM TOC extraction returned %d entries, exceeding the sane maximum of %d.",
            len(entries),
            _MAX_SANE_CHAPTER_COUNT,
        )
        return False
    declared_pages = [page for _, page in entries]
    if any(a > b for a, b in zip(declared_pages, declared_pages[1:])):
        logger.warning(
            "LLM TOC declared page numbers are not monotonically non-decreasing: %r",
            declared_pages,
        )
        return False
    return True


def _validate_resolved_bounds(bounds: list[tuple[str, int, int]], page_count: int) -> bool:
    """Final defensive check on the offset-resolved bounds before they're trusted or cached."""
    starts = [page_start for _, page_start, _ in bounds]
    if any(s < 1 or s > page_count for s in starts):
        logger.warning("LLM TOC resolved a chapter start outside the document's page range.")
        return False
    if any(a >= b for a, b in zip(starts, starts[1:])):
        logger.warning("LLM TOC resolved chapter starts are not strictly increasing.")
        return False
    return True


def _normalize_title(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _page_top_text(pages_text: list[str], page_num: int, n_lines: int = 4) -> str:
    lines = [line.strip() for line in pages_text[page_num - 1].split("\n") if line.strip()]
    return " ".join(lines[:n_lines])


_PART_PREFIX_RE = re.compile(r"^part\s+[ivxlcdm]+\s+")
_PART_DIVIDER_LINE_RE = re.compile(r"^part\s+[ivxlcdm]+\b", re.IGNORECASE)


def _title_similarity(title: str, page_num: int, pages_text: list[str]) -> float:
    """Similarity between a TOC title and a candidate page's top-of-page text.

    An exact (normalized) substring match scores 1.0 outright -- the common
    case where a real chapter heading is immediately followed by body text,
    which would otherwise drag a plain SequenceMatcher ratio down just
    because the page has more text than the title. A "Part" divider page is
    often laid out with its words in a different order than the TOC prints
    them (e.g. a page showing "Appendices" as a large title and "PART VI" as
    a small label above it, extracted in that visual order rather than the
    TOC's "Part VI: Appendices") -- scored 0.95 if every word of a
    multi-word title is present on the page regardless of order, just below
    a real substring match so an exact match elsewhere in the window still
    wins. Otherwise falls back to a length-matched fuzzy ratio, comparing
    against only a similarly-sized window of the page text so trailing body
    text doesn't dilute it.

    A "Part" divider's title (e.g. "PART V: C++ SOFTWARE ENGINEERING") is
    additionally tried with its "PART <roman>" label stripped, since these
    pages print just the descriptive name ("C++ Software Engineering") as
    the literal opening line, with the "PART V" label itself, and the
    bulleted list of child chapters, appearing only several lines further
    down -- below where `_page_top_text` looks. Confirmed concretely on the
    Gregoire "Professional C++" PDF: every one of its Part dividers follows
    this layout. Widening `_page_top_text`'s own line count to reach that
    label was tried and rejected -- it let inline forward-references like
    "discussed in Chapter 26, 'Advanced Templates'" (found as deep as line
    10 of an unrelated chapter's body page) start registering as false
    matches again, the exact failure this matching scheme exists to avoid.
    """
    normalized_title = _normalize_title(title)
    normalized_page = _normalize_title(_page_top_text(pages_text, page_num))
    if not normalized_title or not normalized_page:
        return 0.0
    candidates = [normalized_title]
    without_part_label = _PART_PREFIX_RE.sub("", normalized_title)
    if without_part_label != normalized_title:
        candidates.append(without_part_label)

    best = 0.0
    for candidate in candidates:
        if candidate in normalized_page:
            return 1.0
        words = candidate.split()
        if len(words) > 1 and set(words) <= set(normalized_page.split()):
            best = max(best, 0.95)
    if best:
        return best

    window = normalized_page[: len(normalized_title) + 20]
    return SequenceMatcher(None, normalized_title, window).ratio()


def _is_part_divider_page(pages_text: list[str], page_num: int) -> bool:
    """Whether this page carries a standalone "PART <roman numeral>" line anywhere in its text.

    A "Part" divider page commonly lists the titles of its first few child
    chapters verbatim, as a kind of mini table of contents for that part
    (confirmed concretely on the Gregoire "Professional C++" PDF, where
    every divider page does this). That listing can make the divider page
    itself score a perfect substring match for one of those chapters' own
    titles -- particularly the first one, before its own real opener page
    has even been reached. Checked against the page's full text rather than
    just `_page_top_text`'s narrow window: unlike matching a chapter's own
    title (where a deep search risks catching an unrelated inline
    cross-reference), this only needs to recognize the divider page's own
    distinctive, low-risk "Part <roman numeral>" label, which can appear
    anywhere on the page depending on layout.
    """
    return any(
        _PART_DIVIDER_LINE_RE.match(line.strip())
        for line in pages_text[page_num - 1].split("\n")
    )


def _find_first_match(
    title: str, pages_text: list[str], page_start: int, page_end: int
) -> int | None:
    """First page (in order) whose top-of-page text resembles `title`, above the match threshold.

    "First match wins" rather than "best match wins" matters for any
    chapter, not just chapter 1: a chapter's title typically appears once on
    its own opener page and then again, verbatim, in every subsequent
    page's running header. The opener's score can only ever reach the
    "bag of words" tier of `_title_similarity` (0.95, since extra title-page
    furniture like "WHAT'S IN THIS CHAPTER?" dilutes a plain substring
    match), while any later page's running header scores a perfect 1.0.
    Picking the highest-scoring page in the window would therefore
    systematically prefer a later page over the chapter's true opener --
    confirmed concretely on the Gregoire "Professional C++" PDF, where this
    bug affected essentially every chapter. For chapter 1 specifically,
    "first match wins" is additionally safe because nothing -- including
    the in-body false positives that motivated this whole fallback -- can
    precede the real chapter 1; for chapters 2..N, the window passed in by
    `_resolve_chapter_pages` is what keeps an earlier false positive (e.g. a
    stray cross-reference landing in some page's top few lines) from being
    picked instead.

    A page that is itself a "Part" divider is skipped when searching for
    anything other than a "Part" title (see `_is_part_divider_page`) --
    otherwise the same "first match wins" policy that protects against a
    later running header would just as readily lock onto an even earlier
    divider page that happens to list the target chapter's title too.
    """
    title_is_part_label = bool(_PART_DIVIDER_LINE_RE.match(_normalize_title(title)))
    for page_num in range(page_start, page_end + 1):
        if not title_is_part_label and _is_part_divider_page(pages_text, page_num):
            continue
        if _title_similarity(title, page_num, pages_text) >= _TITLE_MATCH_THRESHOLD:
            return page_num
    return None


def _resolve_chapter_pages(
    entries: list[tuple[str, int]],
    raw_pages_text: list[str],
    page_count: int,
    search_start: int,
) -> list[tuple[str, int, int]] | None:
    """Map each TOC entry's declared (printed) page to its real physical PDF page.

    The delta between printed and physical pages is rarely constant across
    a whole book (front matter, color plates, unnumbered pages), so instead
    of computing one offset and adding it everywhere, each chapter after
    the first re-anchors near an *estimate* derived from the previous
    chapter's resolved offset, searching an expanding window rather than
    trusting the estimate outright. Search windows never look before the
    previous chapter's resolved page, which is what keeps this from ever
    locking onto a same-titled false positive earlier in the book.

    Takes `raw_pages_text` (header/footer-stripped, but not `_clean_text`'d)
    specifically so `_page_top_text`'s "first few lines" actually means the
    physical top of the page. Once `_clean_text` collapses single newlines
    into spaces, a page's "lines" become whole paragraphs, so capping at 4
    of them can swallow a page's *entire* text -- including, on a page with
    few paragraph breaks, an inline cross-reference like "see Chapter 10"
    buried in the body. That inline mention then scores a perfect substring
    match against "Chapter 10"'s own title, just like a real running
    header would. Confirmed concretely on the Gregoire "Professional C++"
    PDF, which forward-references chapters this way throughout its prose.
    """
    first_title, first_declared = entries[0]
    first_physical = _find_first_match(first_title, raw_pages_text, search_start, page_count)
    if first_physical is None:
        return None

    resolved_physical = [first_physical]
    prev_physical, prev_declared = first_physical, first_declared
    for title, declared in entries[1:]:
        physical = None
        offset = prev_physical - prev_declared
        estimate = declared + offset
        for radius in _OFFSET_SEARCH_RADII:
            window_start = max(prev_physical + 1, estimate - radius)
            window_end = min(page_count, estimate + radius)
            if window_start > window_end:
                continue
            physical = _find_first_match(title, raw_pages_text, window_start, window_end)
            if physical is not None:
                break
        if physical is None:
            return None
        resolved_physical.append(physical)
        prev_physical, prev_declared = physical, declared

    bounds = []
    for i, (title, _) in enumerate(entries):
        page_start = resolved_physical[i]
        page_end = resolved_physical[i + 1] - 1 if i + 1 < len(entries) else page_count
        page_end = max(page_start, page_end)
        bounds.append((title, page_start, page_end))
    return bounds


# -- section derivation --------------------------------------------------------


def _sub_entries_for_chapter(toc: list, page_start: int, page_end: int) -> list[tuple[str, int]]:
    seen_pages: set[int] = set()
    entries = []
    for level, title, page in toc:
        if level >= 2 and page_start <= page <= page_end and page not in seen_pages:
            entries.append((title.strip(), page))
            seen_pages.add(page)
    return sorted(entries, key=lambda e: e[1])


def _derive_sections(
    sub_entries: list[tuple[str, int]],
    pages_text: list[str],
    page_start: int,
    page_end: int,
    section_split_tokens: int,
) -> list[Section]:
    if sub_entries:
        return _sections_from_subentries(sub_entries, pages_text, page_start, page_end)

    chapter_text = _join_pages(pages_text, page_start, page_end)
    if len(chapter_text) > section_split_tokens * _CHARS_PER_TOKEN:
        return _auto_split_chapter(pages_text, page_start, page_end, section_split_tokens)
    return [Section(0, None, chapter_text, page_start)]


def _sections_from_subentries(
    sub_entries: list[tuple[str, int]], pages_text: list[str], page_start: int, page_end: int
) -> list[Section]:
    boundaries: list[tuple[str | None, int]] = []
    if sub_entries[0][1] > page_start:
        boundaries.append((None, page_start))
    boundaries.extend(sub_entries)

    sections = []
    for i, (title, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] - 1 if i + 1 < len(boundaries) else page_end
        end = max(end, start)
        text = _join_pages(pages_text, start, end)
        sections.append(Section(i, title, text, start))
    return sections


def _auto_split_chapter(
    pages_text: list[str], page_start: int, page_end: int, section_split_tokens: int
) -> list[Section]:
    """Greedily pack paragraphs into ~section_split_tokens-sized sections.

    Breaks always land on a paragraph boundary, except when a single
    paragraph alone exceeds the target size, in which case it simply
    forms its own oversized section.
    """
    max_chars = section_split_tokens * _CHARS_PER_TOKEN
    paragraphs: list[tuple[int, str]] = []
    for offset, page_text in enumerate(pages_text[page_start - 1 : page_end]):
        page_num = page_start + offset
        for para in page_text.split("\n\n"):
            para = para.strip()
            if para:
                paragraphs.append((page_num, para))

    if not paragraphs:
        return [Section(0, None, "", page_start)]

    sections: list[Section] = []
    current_parts: list[str] = []
    current_len = 0
    current_page = paragraphs[0][0]
    for page_num, para in paragraphs:
        added_len = len(para) + 2
        if current_parts and current_len + added_len > max_chars:
            sections.append(
                Section(len(sections), None, "\n\n".join(current_parts), current_page)
            )
            current_parts = []
            current_len = 0
            current_page = page_num
        current_parts.append(para)
        current_len += added_len
    if current_parts:
        sections.append(Section(len(sections), None, "\n\n".join(current_parts), current_page))
    return sections


def _join_pages(pages_text: list[str], page_start: int, page_end: int) -> str:
    return "\n\n".join(p for p in pages_text[page_start - 1 : page_end] if p)


# -- text cleaning ----------------------------------------------------------

_EMBEDDED_PAGE_NUMBER_RE = re.compile(r"\d{1,4}")
_PAGE_NUMBER_OFFSET_TOLERANCE = 1
_PAGE_NUMBER_OFFSET_SEARCH_RADIUS = 20


def _embedded_page_number(line: str) -> int | None:
    """A number sharing its line with other text -- unambiguously a running header/footer.

    "398  |  CHAPTER 11    Odds and Ends" can only be a genuine running
    header: nothing else on a chapter-opener page combines a number with
    other text on the same line. A standalone numeral alone on its line is
    the ambiguous case `_is_running_page_number` resolves, so it's
    deliberately excluded here (it would otherwise count as its own
    evidence for itself).
    """
    stripped = line.strip()
    if not stripped or _PAGE_NUMBER_RE.match(stripped):
        return None
    match = _EMBEDDED_PAGE_NUMBER_RE.search(stripped)
    return int(match.group()) if match else None


def _expected_page_number(split_pages: list[list[str]], page_index: int) -> int | None:
    """This page's printed number, estimated from the nearest confirmed running header/footer.

    Looked up by physical-to-printed offset from the closest page that has
    one, rather than a fixed guess, since the gap between physical and
    printed page number grows over the course of a book (front matter,
    part dividers, color plates) and isn't constant.
    """
    for radius in range(_PAGE_NUMBER_OFFSET_SEARCH_RADIUS + 1):
        for candidate in {page_index - radius, page_index + radius}:
            if not (0 <= candidate < len(split_pages)):
                continue
            lines = split_pages[candidate]
            if not lines:
                continue
            for line in (lines[0], lines[-1]):
                number = _embedded_page_number(line)
                if number is not None:
                    return number + (page_index - candidate)
    return None


def _is_running_page_number(line: str, page_index: int, split_pages: list[list[str]]) -> bool:
    """A standalone first/last line is a genuine page number, not a chapter-opener's bare number.

    Some chapter-opener pages print just the bare chapter number, styled
    large, with no running header at all -- confirmed concretely on the
    Gregoire "Professional C++" PDF for every one of its 34 chapters.
    Unconditionally stripping any standalone numeral in first/last-line
    position erased that title's leading number, which made
    `_title_similarity` miss the opener page and resolve the chapter a page
    late (or, when a "Part" divider page happened to repeat the same title
    words, onto that unrelated divider instead). A genuine running page
    number tracks this page's own printed position; a chapter number does
    not, so they're told apart by comparing against the locally expected
    printed number rather than stripping on sight.
    """
    stripped = line.strip()
    if not _PAGE_NUMBER_RE.match(stripped):
        return False
    digits = re.findall(r"\d+", stripped)
    if not digits:
        return True  # pure roman numeral -- this book's front matter, never a chapter number
    expected = _expected_page_number(split_pages, page_index)
    if expected is None:
        return True  # no nearby evidence either way -- preserve prior behavior
    return abs(int(digits[0]) - expected) <= _PAGE_NUMBER_OFFSET_TOLERANCE


def _strip_running_headers_footers(pages_text: list[str]) -> list[str]:
    """Drop the first/last line of each page if it repeats on most pages, or is a stray page number.

    The page-number check is intentionally position-gated (first/last line
    only), not applied to every line -- a printed TOC's dot leader puts each
    entry's page number alone on the line right after its title (PyMuPDF's
    "text" mode breaks "<title>\\t<page>" onto two lines), and an
    unconditional same-pattern strip would silently delete every one of
    those before the LLM-TOC fallback ever sees them, leaving it nothing
    to extract `declared_page` from. Confirmed concretely on the Gregoire
    "Professional C++" PDF, where this caused the LLM to fabricate page
    numbers wholesale instead of reading the real (intact) ones.
    """
    if len(pages_text) < 3:
        return pages_text
    split_pages = [text.split("\n") for text in pages_text]
    first_lines = Counter(lines[0].strip() for lines in split_pages if lines)
    last_lines = Counter(lines[-1].strip() for lines in split_pages if lines)
    threshold = max(3, len(pages_text) // 2)
    repeated_first = {line for line, n in first_lines.items() if line and n >= threshold}
    repeated_last = {line for line, n in last_lines.items() if line and n >= threshold}

    cleaned = []
    for page_index, lines in enumerate(split_pages):
        if lines and lines[0].strip() in repeated_first:
            lines = lines[1:]
        if lines and lines[-1].strip() in repeated_last:
            lines = lines[:-1]
        if lines and _is_running_page_number(lines[0], page_index, split_pages):
            lines = lines[1:]
        if lines and _is_running_page_number(lines[-1], page_index, split_pages):
            lines = lines[:-1]
        cleaned.append("\n".join(lines))
    return cleaned


def _clean_text(raw: str) -> str:
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", raw)
    text = re.sub(r"\n{2,}", _PARA_MARKER, text)
    text = text.replace("\n", " ")
    text = text.replace(_PARA_MARKER, "\n\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
