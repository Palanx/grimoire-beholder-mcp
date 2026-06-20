"""LLM-extracted TOC fallback: used only when a PDF has no embedded outline.

Covers the Gregoire "Professional C++" failure mode -- no embedded outline,
a multi-page printed TOC whose declared page numbers don't match physical
PDF pages, and an inline "CHAPTER N:" string inside an appendix that the
regex-only heading fallback misreads as a chapter boundary. All tests here
run with no Ollama daemon: the LLM client is a small canned-response fake.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from grimoire_beholder import db
from grimoire_beholder.sources import pdf

_VALID_TOC_RESPONSE = json.dumps(
    [
        {"title": "Chapter 1: Origins", "declared_page": 1},
        {"title": "Chapter 2: Consequences", "declared_page": 6},
        {"title": "Appendix A: Common Mistakes", "declared_page": 10},
    ]
)


class _CannedTocClient:
    """Returns a fixed canned response to every generate() call, regardless of prompt."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.generate_calls: list[tuple[str, str, str]] = []

    def generate(self, model: str, system: str, prompt: str) -> str:
        self.generate_calls.append((model, system, prompt))
        return self.response

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        raise NotImplementedError


class _SequencedTocClient:
    """Returns one canned response per call, in order -- one per expected batch."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.generate_calls: list[tuple[str, str, str]] = []

    def generate(self, model: str, system: str, prompt: str) -> str:
        self.generate_calls.append((model, system, prompt))
        return self.responses[len(self.generate_calls) - 1]

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        raise NotImplementedError


def test_extract_toc_entries_in_batches_concatenates_in_order() -> None:
    """A 4-page region splits into two non-overlapping 2-page batches."""
    raw_pages_text = [f"page {i}" for i in range(1, 5)]
    responses = [
        json.dumps([{"title": "Chapter 1", "declared_page": 1}, {"title": "Chapter 2", "declared_page": 5}]),
        json.dumps([{"title": "Chapter 3", "declared_page": 10}]),
    ]
    client = _SequencedTocClient(responses)

    entries = pdf._extract_toc_entries_in_batches(raw_pages_text, (1, 4), client, "cogito")

    assert len(client.generate_calls) == 2
    assert entries == [("Chapter 1", 1), ("Chapter 2", 5), ("Chapter 3", 10)]


def test_extract_toc_entries_in_batches_drops_duplicate_at_boundary() -> None:
    """If the same heading is echoed at both sides of a batch boundary, keep only one."""
    raw_pages_text = [f"page {i}" for i in range(1, 5)]
    responses = [
        json.dumps([{"title": "Chapter 1", "declared_page": 1}, {"title": "Chapter 2", "declared_page": 5}]),
        json.dumps([{"title": "Chapter 2", "declared_page": 5}, {"title": "Chapter 3", "declared_page": 10}]),
    ]
    client = _SequencedTocClient(responses)

    entries = pdf._extract_toc_entries_in_batches(raw_pages_text, (1, 4), client, "cogito")

    assert entries == [("Chapter 1", 1), ("Chapter 2", 5), ("Chapter 3", 10)]


def test_extract_toc_entries_in_batches_retries_malformed_batch_as_single_pages() -> None:
    """A batch too dense to parse as a whole is retried page-by-page instead of dropped.

    A 2-page span with enough sub-entries can run the model's response past
    whatever it would have emitted before closing the JSON array, even
    though either page alone parses fine -- confirmed concretely on the
    Gregoire "Professional C++" PDF's pages 20-21.
    """
    raw_pages_text = [f"page {i}" for i in range(1, 5)]
    responses = [
        "not valid json at all",
        json.dumps([{"title": "Chapter 1", "declared_page": 1}]),
        json.dumps([{"title": "Chapter 2", "declared_page": 5}]),
        json.dumps([{"title": "Chapter 3", "declared_page": 10}]),
    ]
    client = _SequencedTocClient(responses)

    entries = pdf._extract_toc_entries_in_batches(raw_pages_text, (1, 4), client, "cogito")

    assert len(client.generate_calls) == 4
    assert entries == [("Chapter 1", 1), ("Chapter 2", 5), ("Chapter 3", 10)]


def test_extract_toc_entries_in_batches_retries_a_batch_that_drops_a_real_heading() -> None:
    """Well-formed JSON that silently omits a real heading is retried, same as malformed JSON.

    Confirmed concretely on the Gregoire "Professional C++" PDF's pages
    36-37: the model's response for the 2-page batch parsed fine and
    correctly included "Chapter 1" but silently dropped "Chapter 2",
    even though "Chapter 2" is structurally present (its own line,
    matching the same Part/Chapter/Appendix pattern) in the raw text --
    recovered once each page is asked about alone.
    """
    raw_pages_text = ["Chapter 1: First\nsome body text", "Chapter 2: Second\nmore body text"]
    responses = [
        json.dumps([{"title": "Chapter 1: First", "declared_page": 1}]),
        json.dumps([{"title": "Chapter 1: First", "declared_page": 1}]),
        json.dumps([{"title": "Chapter 2: Second", "declared_page": 2}]),
    ]
    client = _SequencedTocClient(responses)

    entries = pdf._extract_toc_entries_in_batches(raw_pages_text, (1, 2), client, "cogito")

    assert len(client.generate_calls) == 3
    assert entries == [("Chapter 1: First", 1), ("Chapter 2: Second", 2)]


def test_extract_toc_entries_in_batches_accepts_a_short_batch_with_no_more_headings() -> None:
    """A batch whose entry count matches its structural heading count is accepted without retry."""
    raw_pages_text = ["Chapter 1: First\nsome body text", "more body text, no heading here"]
    responses = [json.dumps([{"title": "Chapter 1: First", "declared_page": 1}])]
    client = _SequencedTocClient(responses)

    entries = pdf._extract_toc_entries_in_batches(raw_pages_text, (1, 2), client, "cogito")

    assert len(client.generate_calls) == 1
    assert entries == [("Chapter 1: First", 1)]


def test_extract_toc_entries_in_batches_drops_a_page_still_malformed_alone() -> None:
    """A page that still fails to parse on its own has nowhere smaller to retry -- it's dropped."""
    raw_pages_text = [f"page {i}" for i in range(1, 5)]
    responses = [
        "not valid json at all",
        "still not valid json",
        json.dumps([{"title": "Chapter 2", "declared_page": 5}]),
        json.dumps([{"title": "Chapter 3", "declared_page": 10}]),
    ]
    client = _SequencedTocClient(responses)

    entries = pdf._extract_toc_entries_in_batches(raw_pages_text, (1, 4), client, "cogito")

    assert len(client.generate_calls) == 4
    assert entries == [("Chapter 2", 5), ("Chapter 3", 10)]


def test_extract_toc_entries_in_batches_drops_sub_entries_the_model_failed_to_skip() -> None:
    """A small model asked to skip sub-entries does so unreliably on deeply nested TOC pages.

    Even when a batch's response is valid JSON, it may still include
    indented sub-headings the prompt explicitly said to skip. Only
    structurally top-level titles (Part/Chapter/Appendix + number or
    letter) must survive, regardless of what the model included.
    """
    raw_pages_text = [f"page {i}" for i in range(1, 3)]
    responses = [
        json.dumps(
            [
                {"title": "Chapter 7: Memory Management", "declared_page": 31},
                {"title": "Memory Leaks", "declared_page": 37},
                {"title": "Smart Pointers", "declared_page": 41},
                {"title": "Chapter 8: Gaining Proficiency", "declared_page": 50},
            ]
        ),
    ]
    client = _SequencedTocClient(responses)

    entries = pdf._extract_toc_entries_in_batches(raw_pages_text, (1, 2), client, "cogito")

    assert entries == [
        ("Chapter 7: Memory Management", 31),
        ("Chapter 8: Gaining Proficiency", 50),
    ]


def test_is_top_level_toc_entry_accepts_part_chapter_appendix_only() -> None:
    assert pdf._is_top_level_toc_entry("Chapter 7: Memory Management") is True
    assert pdf._is_top_level_toc_entry("Part III: C++ Coding the Professional Way") is True
    assert pdf._is_top_level_toc_entry("Appendix A: C++ Interviews") is True
    assert pdf._is_top_level_toc_entry("Introduction") is False
    assert pdf._is_top_level_toc_entry("Memory Leaks") is False
    assert pdf._is_top_level_toc_entry("Introducing the Spreadsheet Example") is False


def test_valid_llm_toc_resolves_correct_chapters_and_offsets(
    conn: sqlite3.Connection, llm_toc_pdf_path: Path
) -> None:
    """A valid mocked TOC produces the right chapter count, titles, and resolved pages.

    Declared pages (1, 6, 10) are deliberately offset from the physical
    pages chapters actually start on (5, 7, 8), by a *different* amount each
    time -- this is the offset-resolution requirement, not just a constant
    shift. The false-positive "Chapter 4:" page (9) must never become a
    fourth chapter.
    """
    client = _CannedTocClient(_VALID_TOC_RESPONSE)

    book = pdf.extract_book(str(llm_toc_pdf_path), conn=conn, llm_client=client, llm_model="cogito")

    assert len(client.generate_calls) == 1
    assert [c.title for c in book.chapters] == [
        "Chapter 1: Origins",
        "Chapter 2: Consequences",
        "Appendix A: Common Mistakes",
    ]
    assert [c.page_start for c in book.chapters] == [5, 7, 8]
    assert book.chapters[-1].sections
    full_text = " ".join(s.text for c in book.chapters for s in c.sections)
    assert "Chapter 4: Counting Mistakes" not in [c.title for c in book.chapters]
    assert "off-by-one errors" in full_text


def test_heading_only_fallback_would_misfire_on_this_fixture(llm_toc_pdf_path: Path) -> None:
    """Sanity-check that the fixture really triggers the bug this fallback exists to fix.

    Without the LLM-TOC step, heading-only detection promotes the inline
    "Chapter 4:" appendix sub-heading into its own false chapter, and also
    misreads the TOC's own dot-leader text as a chapter heading. This
    documents *why* the LLM-TOC fallback takes priority over heading
    detection, not just that it does.
    """
    book = pdf.extract_book(str(llm_toc_pdf_path))

    titles = [c.title for c in book.chapters]
    assert any(t.startswith("Chapter 4: Counting Mistakes") for t in titles)


def test_cached_toc_is_reused_without_calling_the_llm_again(
    conn: sqlite3.Connection, llm_toc_pdf_path: Path
) -> None:
    """Re-extracting the same PDF must hit the pdf_toc_cache, never re-call the LLM."""
    client = _CannedTocClient(_VALID_TOC_RESPONSE)

    first = pdf.extract_book(str(llm_toc_pdf_path), conn=conn, llm_client=client, llm_model="cogito")
    second = pdf.extract_book(str(llm_toc_pdf_path), conn=conn, llm_client=client, llm_model="cogito")

    assert len(client.generate_calls) == 1
    assert [c.title for c in first.chapters] == [c.title for c in second.chapters]
    assert [c.page_start for c in first.chapters] == [c.page_start for c in second.chapters]


def test_cache_is_keyed_by_content_hash_and_persists_in_db(
    conn: sqlite3.Connection, llm_toc_pdf_path: Path
) -> None:
    client = _CannedTocClient(_VALID_TOC_RESPONSE)
    book = pdf.extract_book(str(llm_toc_pdf_path), conn=conn, llm_client=client, llm_model="cogito")

    cached = db.get_cached_pdf_toc(conn, book.content_hash)

    assert cached is not None
    assert cached[0] == ("Chapter 1: Origins", 5, 6)
    assert cached[1] == ("Chapter 2: Consequences", 7, 7)
    assert cached[2] == ("Appendix A: Common Mistakes", 8, 9)


def test_empty_llm_toc_falls_back_to_heading_detection_without_crashing(
    conn: sqlite3.Connection, llm_toc_pdf_path: Path
) -> None:
    client = _CannedTocClient(json.dumps([]))

    book = pdf.extract_book(str(llm_toc_pdf_path), conn=conn, llm_client=client, llm_model="cogito")

    assert len(book.chapters) >= 1
    assert db.get_cached_pdf_toc(conn, book.content_hash) is None


def test_non_monotonic_declared_pages_falls_back_to_heading_detection(
    conn: sqlite3.Connection, llm_toc_pdf_path: Path
) -> None:
    out_of_order = json.dumps(
        [
            {"title": "Chapter 2: Consequences", "declared_page": 6},
            {"title": "Chapter 1: Origins", "declared_page": 1},
        ]
    )
    client = _CannedTocClient(out_of_order)

    book = pdf.extract_book(str(llm_toc_pdf_path), conn=conn, llm_client=client, llm_model="cogito")

    assert len(book.chapters) >= 1
    assert db.get_cached_pdf_toc(conn, book.content_hash) is None


def test_malformed_json_response_falls_back_to_heading_detection(
    conn: sqlite3.Connection, llm_toc_pdf_path: Path
) -> None:
    client = _CannedTocClient("not valid json at all")

    book = pdf.extract_book(str(llm_toc_pdf_path), conn=conn, llm_client=client, llm_model="cogito")

    assert len(book.chapters) >= 1
    assert db.get_cached_pdf_toc(conn, book.content_hash) is None


def test_fenced_json_response_is_parsed_defensively(
    conn: sqlite3.Connection, llm_toc_pdf_path: Path
) -> None:
    fenced = f"```json\n{_VALID_TOC_RESPONSE}\n```"
    client = _CannedTocClient(fenced)

    book = pdf.extract_book(str(llm_toc_pdf_path), conn=conn, llm_client=client, llm_model="cogito")

    assert [c.title for c in book.chapters] == [
        "Chapter 1: Origins",
        "Chapter 2: Consequences",
        "Appendix A: Common Mistakes",
    ]


def test_no_llm_client_supplied_skips_straight_to_heading_detection(
    llm_toc_pdf_path: Path,
) -> None:
    """Omitting conn/llm_client (the pre-existing call signature) must keep working unchanged."""
    book = pdf.extract_book(str(llm_toc_pdf_path))

    assert len(book.chapters) >= 1


def test_resolved_bounds_out_of_range_are_rejected() -> None:
    assert pdf._validate_resolved_bounds([("A", 0, 5), ("B", 6, 10)], page_count=10) is False
    assert pdf._validate_resolved_bounds([("A", 1, 5), ("B", 11, 20)], page_count=10) is False
    assert pdf._validate_resolved_bounds([("A", 5, 8), ("B", 3, 10)], page_count=10) is False
    assert pdf._validate_resolved_bounds([("A", 1, 5), ("B", 6, 10)], page_count=10) is True


def test_declared_entries_validation_rejects_absurd_chapter_count() -> None:
    entries = [(f"Chapter {i}", i) for i in range(1, 200)]
    assert pdf._validate_declared_entries(entries) is False


def test_detect_toc_pages_spans_the_full_multi_page_region(llm_toc_pdf_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open(str(llm_toc_pdf_path))
    raw_pages_text = [page.get_text("text") for page in doc]
    doc.close()
    raw_pages_text = pdf._strip_running_headers_footers(raw_pages_text)

    region = pdf._detect_toc_pages(raw_pages_text)

    assert region == (2, 3)


def test_detect_toc_pages_extends_via_repeating_header_when_density_collapses(
    sparse_density_toc_pdf_path: Path,
) -> None:
    """Region extension must not stop just because a TOC page's sub-entries lack page numbers.

    Page 3 of this fixture has 0% line-density (no trailing page numbers at
    all) and page 4 has 25% (below the 0.3 threshold) -- under the old,
    density-only heuristic both would have ended the region right after the
    anchor. The repeating "Contents" header is what correctly carries
    detection through to page 4, stopping only at the real Preface page.
    """
    import pymupdf

    doc = pymupdf.open(str(sparse_density_toc_pdf_path))
    raw_pages_text = [page.get_text("text") for page in doc]
    doc.close()
    raw_pages_text = pdf._strip_running_headers_footers(raw_pages_text)

    region = pdf._detect_toc_pages(raw_pages_text)

    assert region == (2, 4)
    toc_text = pdf._join_pages(raw_pages_text, *region)
    assert "Chapter 1: Origins" in toc_text
    assert "Chapter 2: Consequences" in toc_text


def test_strip_running_headers_footers_keeps_a_chapter_openers_bare_number() -> None:
    """A chapter-opener's bare number must survive even though it's alone on its line.

    Confirmed concretely on the Gregoire "Professional C++" PDF: every one
    of its 34 chapter-opener pages prints just the bare chapter number
    (e.g. "30"), styled large, with no combined running header at all --
    unlike every other page in the chapter, which prints a combined
    "<printed-page> | CHAPTER 30 Title" header. Stripping the bare number
    erased the title's leading word, which broke `_title_similarity` and
    resolved the chapter a page late.
    """
    pages_text = [f"{1000 + i}  |  CHAPTER 30   Becoming Adept at Testing\nbody text" for i in range(5)]
    pages_text.insert(0, "30\nBecoming Adept at Testing\nWHAT'S IN THIS CHAPTER?")

    cleaned = pdf._strip_running_headers_footers(pages_text)

    assert cleaned[0].startswith("30\n")


def test_strip_running_headers_footers_still_strips_a_genuine_page_number() -> None:
    """A standalone number that actually tracks this page's position still gets stripped.

    Most pages here carry a combined "<page> | CHAPTER 5 Something" header,
    establishing what a real page number looks like in this neighborhood;
    page 5 prints that same running number alone, with no other text on the
    line -- still a genuine page number, just one page that happens to omit
    the rest of the header.
    """
    pages_text = [
        f"{200 + i}  |  CHAPTER 5    Something\nsome unrelated body prose" for i in range(10)
    ]
    pages_text[5] = "205"

    cleaned = pdf._strip_running_headers_footers(pages_text)

    assert cleaned[5] == ""


def test_title_similarity_matches_part_divider_with_label_stripped() -> None:
    """A "Part" divider's descriptive name, alone, must still score a full match.

    Confirmed concretely on the Gregoire "Professional C++" PDF: every Part
    divider page prints its descriptive name ("C++ Software Engineering")
    as the literal opening line, with the "PART V" label itself, and the
    bulleted list of child chapters, appearing only several lines further
    down -- below where `_page_top_text`'s default window looks. Matching
    only the literal, unstripped title would miss this page entirely.
    """
    raw_pages_text = [
        "C++ Software Engineering\n"
        "▸\n"
        "▸CHAPTER 28: Maximizing Software Engineering Methods\n"
        "▸\n"
        "▸CHAPTER 29: Writing Efficient C++\n"
        "PART V\n"
        "Professional C++, Fifth Edition."
    ]

    score = pdf._title_similarity("PART V: C++ Software Engineering", 1, raw_pages_text)

    assert score == 1.0


def test_find_first_match_skips_a_part_divider_listing_the_target_chapter() -> None:
    """A "Part" divider that lists the target chapter's title must not win over its real opener.

    Confirmed concretely on the Gregoire "Professional C++" PDF: a "Part"
    divider page often lists the titles of its first few child chapters
    verbatim, as a kind of mini table of contents -- which can score a
    perfect substring match for one of those chapters' own titles, on a
    page that precedes that chapter's real opener. This is the same
    "first match wins" policy from `_find_first_match` working against
    itself unless divider pages are explicitly excluded as candidates for
    a non-"Part" title.
    """
    raw_pages_text = [
        "C++ Coding the Professional Way\nPART III\n▸\n▸CHAPTER 7: Memory Management",
        "filler page with unrelated body text",
        "7\nMemory Management\nWHAT'S IN THIS CHAPTER?",
    ]

    match = pdf._find_first_match("CHAPTER 7: Memory Management", raw_pages_text, 1, 3)

    assert match == 3


def test_title_match_threshold_rejects_a_weak_fuzzy_collision_between_similar_titles() -> None:
    """Two textually-similar but distinct chapter titles must not cross-match via fuzzy ratio.

    Confirmed concretely on the Gregoire "Professional C++" PDF: Chapter 8
    ("Gaining Proficiency with Classes and Objects") and Chapter 9
    ("Mastering Classes and Objects") share enough wording that Chapter 8's
    body text scored a 0.673 fuzzy-ratio match for Chapter 9's title -- comfortably
    above the old 0.6 threshold, even though every genuine opener page in
    the book scores at least 0.938. `_find_first_match` must keep scanning
    past that weak collision to reach the real opener.
    """
    raw_pages_text = [
        "278 | CHAPTER 8   Gaining Proficiency with Classes and Objects\n"
        "As usual, if you don't write your own assignment operator, C++ writes one for you "
        "to allow objects to be assigned to one another.",
        "filler page with unrelated body text",
        "9\nMastering Classes and Objects\nWHAT'S IN THIS CHAPTER?",
    ]

    match = pdf._find_first_match("CHAPTER 9: Mastering Classes and Objects", raw_pages_text, 1, 3)

    assert match == 3
