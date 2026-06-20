# Architecture

This document describes how grimoire-beholder-mcp is layered, the two explicit extension
points (`SourceParser`, `RetrievalStrategy`), and the recipes for using
them. It assumes you've read the README's data-model and resume sections;
this is the "how it's built," not the "how to run it." For operating
rules see [AGENTS.md](AGENTS.md); for current state and decision history
see [PROJECT_LOG.md](PROJECT_LOG.md).

## Layers and the dependency rule

```
            cli.py              mcp_server.py        (interfaces -- thin, no business logic)
                \                    /
                 \                  /
                ingest.py   <-- orchestrates one book's pipeline
                 /    |   \    \
          sources/  chunk.py  contextualize.py  embed.py
        (parsing)                                  |
                                              search.py  <-- composition root for retrieval
                                                    |
                                            retrieval/  (strategies + fusion)
                                                    |
                                                  db.py  (the only module that touches SQLite)
```

The dependency rule: **everything points down to `db.py`, never sideways
through the CLI or MCP server, and never up from `db.py`.**
`sources/`, `retrieval/`, `chunk.py`, and `contextualize.py` know nothing
about Typer, FastMCP, or each other. `db.py` knows nothing about any of
them -- it only knows rows and SQL.

`ollama_client.py` is a separate, orthogonal seam: an `OllamaClient`
Protocol (`generate`, `embed`) with `RealOllamaClient` for production and a
`FakeOllamaClient` (in `tests/fakes.py`) for the test suite. Every module
that needs Ollama takes a client as a parameter rather than importing a
concrete implementation -- this is what lets the entire pipeline (ingest,
contextualize, embed, search) be tested with no Ollama daemon running.

## Data model

`Book -> Chapter -> Section -> Chunk`, stored in one SQLite database
(`db.py`). A chunk's `status` column (`pending -> contextualized ->
embedded`) drives resumability: every stage queries for rows in the state
it cares about, so a crash mid-ingest loses at most one in-flight chunk or
batch, never a whole book. Natural composite-key primary keys
(`book_id, chapter_index, section_index[, chunk_index]`) make re-ingesting
the same source idempotent -- existing rows are never duplicated.

`page_start` is the one field whose *name* is shared across all source
types but whose *meaning* varies: a real 1-indexed page number for PDF, or
a synthetic, strictly increasing logical-location ordinal for sources with
no fixed pagination (EPUB, markdown, plain text). Every parser guarantees
it sorts correctly within a book; nothing downstream needs to know which
kind it's looking at.

## Extension point 1: `SourceParser` (`sources/`)

`sources/__init__.py` defines the protocol and a registry:

```python
class SourceParser(Protocol):
    source_type: str
    extensions: tuple[str, ...]
    def can_parse(self, path: Path) -> bool: ...
    def extract(
        self,
        path: Path,
        section_split_tokens: int = 3000,
        conn: sqlite3.Connection | None = None,
        llm_client: OllamaClient | None = None,
        llm_model: str | None = None,
    ) -> ExtractedBook: ...
```

`get_parser(path)` walks a fixed list of registered parser instances and
returns the first whose `can_parse` claims the file, raising `ValueError`
if none do. `ingest.py` calls `get_parser` once per ingest and never knows
which concrete parser it got -- everything after `extract()` (chunking,
contextualization, embedding, indexing) operates on the same
`ExtractedBook` / `Chapter` / `Section` dataclasses (`sources/common.py`)
regardless of source format. `conn`/`llm_client`/`llm_model` exist solely
for `PdfParser`'s LLM-TOC fallback (below); every other parser accepts and
ignores them so `ingest.py` can still call `extract()` uniformly.

Currently registered: `PdfParser` (`sources/pdf.py`, see the three-tier
chapter-detection fallback below), `EpubParser` (`sources/epub.py`,
one chapter per spine document via `ebooklib`), and `MarkdownParser` /
`PlaintextParser` (`sources/plaintext.py`, sharing one paragraph-based
chapter/section splitter). `sources/common.py`'s `auto_split_paragraphs`
is the shared greedy section-packer used by every parser except PDF's
(PDF keeps its own, separately tested, page-aware variant to avoid any
behavior change to existing PDF tests).

### PDF chapter detection: a three-tier fallback

Not every PDF carries an embedded outline, and heading regexes alone are
unreliable, so `sources/pdf.py`'s `extract_book` tries three strategies in
order, using the first one that produces a usable result:

1. **Embedded outline** (`doc.get_toc()`): level-1 entries become chapters.
   Used whenever present -- it's authoritative and needs no further checks.
2. **LLM-extracted TOC** (`_llm_toc_chapter_bounds`): only attempted when a
   DB connection and an `OllamaClient` were supplied, and only when the
   PDF's front matter contains a detectable printed "Table of
   Contents"/"Contents" page. This step exists because of a real failure
   case ("Professional C++", Gregoire): it has no embedded outline, and its
   real chapter headings don't match any heading regex, while literal
   "CHAPTER N:" strings quoted *inline* inside an appendix's body text do --
   the regex fallback alone promotes that false text into a chapter
   boundary while the entire real book collapses into one bucket.
3. **Heading regex scan** (`_chapter_bounds_from_headings`): last resort,
   unchanged from before this fallback existed.

**Header/footer stripping** (`_strip_running_headers_footers`) runs once,
before any of the three tiers, on every page's raw extracted text. It drops
a page's first/last line if that exact line repeats across most pages (a
running header/footer), or if it's a standalone page-number-shaped line --
but only if that number is actually consistent with the page's position,
estimated by interpolating from the nearest page with unambiguous evidence
(`_expected_page_number`, via a number sharing its line with other text,
e.g. "398 | CHAPTER 11 Odds and Ends"). A flat "any standalone numeral in
first/last-line position is a page number" rule is too broad: some books
print a chapter's bare number alone, styled large, on that chapter's own
opener page (confirmed concretely on the Gregoire "Professional C++" PDF,
for every one of its 34 chapters) -- stripping it erases the title's
leading word and breaks the title matching that offset resolution (below)
depends on.

**TOC-region detection** (`_detect_toc_pages`) anchors on a page whose text
matches "Contents"/"Table of Contents", then keeps including subsequent
pages while a sizeable fraction of their lines look like
`<title> ... <page>` -- stopping at the first page that doesn't (a real
content or preface page). Both this and offset resolution (below) run
against `raw_pages_text` (header/footer-stripped only, real per-line
structure preserved), not the fully cleaned `pages_text` used for chapter
*body* text -- `_clean_text` collapses any single `\n` into a space, which
destroys the per-line pattern TOC detection needs and, for offset
resolution, turns a page's "lines" into whole paragraphs. The LLM prompt
text is built from the same `raw_pages_text` region for the same reason.

**Extraction and validation**: the detected TOC text is sent to `llm_model`
with a prompt demanding a strict JSON array of `{title, declared_page}`
(`_parse_llm_toc_response` strips code fences and skips malformed entries
defensively). The raw entries are rejected outright -- falling back to
heading detection -- if empty, absurdly long (`_MAX_SANE_CHAPTER_COUNT`), or
not monotonically non-decreasing by declared page (`_validate_declared_entries`).
An unvalidated LLM TOC is never used.

**Offset resolution** (`_resolve_chapter_pages`) is the part that can't be a
constant-offset add: a TOC's printed page numbers and the PDF's physical
page indices are rarely related by one fixed delta across an entire book
(front matter, plates, and unnumbered pages all shift it). It runs against
`raw_pages_text`, not the `_clean_text`-flattened `pages_text` -- once
single newlines are collapsed into spaces, a page's "lines" become whole
paragraphs, and `_page_top_text`'s line-count cap can then swallow a page's
*entire* text, including an inline cross-reference to a later chapter's
exact title. The first entry is located by an unconstrained forward scan
(`_find_first_match`) for the first page (in page order, not best-scoring)
whose top-of-page text matches its title -- safe specifically for the first
entry, since nothing, including the false positives this fallback exists to
avoid, can precede it. Every later entry re-anchors near an *estimate*
derived from the previous entry's own resolved offset, searching an
expanding window (`_OFFSET_SEARCH_RADII`) rather than trusting that
estimate outright, and never searching before the previous entry's resolved
page -- which is what keeps it from locking onto an earlier same-titled
false positive (notably, the chapters listed on a preceding "Part" divider
page; see below).

"First match wins" rather than "best match wins" matters everywhere, not
just the first entry: a chapter's title typically appears once on its own
opener page and then again, verbatim, in every later page's running
header. The opener's score can only ever reach the "bag of words" tier
below (running-header furniture like "WHAT'S IN THIS CHAPTER?" dilutes a
plain substring match on the opener itself), while a later running header
scores a perfect exact match -- so picking the page with the highest score
in the window would systematically prefer a later page over the chapter's
true opener.

Title matching (`_title_similarity`) treats an exact (normalized) substring
match as 1.0, falls back to 0.95 if every word of a multi-word title is
present on the page regardless of order (handles a "Part" divider's title
being laid out in a different visual order than the TOC prints it -- and,
for any title starting with "Part `<roman numeral>`", is also tried with that
label stripped, since a divider page typically prints just its descriptive
name as the literal opening line, with the "Part `<roman numeral>`" label
itself appearing only several lines further down), and otherwise falls back
to a length-windowed `difflib.SequenceMatcher` ratio. The match has to clear
`_TITLE_MATCH_THRESHOLD` (0.85) to count -- high enough that two textually
similar but distinct chapter titles (e.g. two consecutive chapters both
ending in "...Classes and Objects") can't cross it via fuzzy ratio alone,
since every genuine opener page scores at least 0.938. A page that itself
carries a standalone "Part `<roman numeral>`" line anywhere in its text
(`_is_part_divider_page`) is skipped as a candidate for any non-"Part"
title, since a "Part" divider commonly lists its first few child chapters'
titles verbatim as a kind of mini table of contents -- otherwise it could
win a perfect substring match for one of those chapters before that
chapter's real opener is ever reached. All of the above was tuned
empirically against the Gregoire "Professional C++" PDF. The resolved
bounds get one more check (`_validate_resolved_bounds`: in-range, strictly
increasing) before being trusted.

**Caching**: a validated, offset-resolved TOC is persisted in the
`pdf_toc_cache` table (`db.py`), keyed by `content_hash` rather than
`book_id` -- extraction runs before the book row exists, and this also
makes re-ingesting the same file under a different slug or name reuse the
cache for free. Re-running ingestion on the same file never re-calls the
LLM.

### Recipe: add a new source type

1. Create `sources/<format>.py` with a class implementing `SourceParser`:
   `source_type`, `extensions`, `can_parse`, and `extract` returning an
   `ExtractedBook` (reuse `sources.common.auto_split_paragraphs` and
   `content_hash_of` if your format has no native section structure).
2. Append an instance to `_PARSERS` in `sources/__init__.py`.
3. Nothing else changes -- `ingest.py`, `chunk.py`, `contextualize.py`,
   `embed.py`, `db.py`, the CLI, and the MCP server are all source-agnostic.
4. Add a test file (`tests/test_sources_<format>.py`) with a tiny synthetic
   fixture built in-test (see `test_sources_epub.py` for a from-scratch
   EPUB built with `ebooklib`, or `test_sources_plaintext.py` for plain
   strings) and assertions on chapters/sections/`page_start` ordering.

## Extension point 2: `RetrievalStrategy` (`retrieval/`)

Unlike `SourceParser`, this is **a protocol with two hardcoded call sites in
`search.py`, not a registry.** There is no `_STRATEGIES` list to append to:
`search()` names `VectorStrategy` and `FtsStrategy` directly and decides in
an `if mode == "vector"` branch which of them run. Adding a third strategy
means editing that branching logic (and likely `_MODES`) in the composition
root, not appending one line to a list -- see the recipe below, which is
honest about this. Don't assume it's as drop-in as adding a `SourceParser`.

```python
class RetrievalStrategy(Protocol):
    name: str
    def run(
        self, conn, question, query_vector, book_id, author, source_type, pool_size,
    ) -> list[RankedHit]: ...
```

A `RankedHit` is `(key, score)` where `key` is the natural chunk key
`(book_id, chapter_index, section_index, chunk_index)` -- not a surrogate
ID, since chunks have none. Two strategies exist today:

- `VectorStrategy` (`retrieval/vector.py`): takes the filtered, embedded
  rows (`db.SearchRow`) at construction -- `search()` fetches them once via
  `db.get_search_rows` and reuses the same list for the final result-row
  lookup, rather than this strategy re-querying them itself -- and computes
  cosine similarity against `query_vector` in numpy, returning the top
  `pool_size`.
- `FtsStrategy` (`retrieval/fts.py`): tokenizes the question, quotes and
  OR-joins the terms (neutralizing FTS5 query syntax so arbitrary user text
  never produces a MATCH syntax error), and calls `db.search_fts`, which
  queries the `chunks_fts` FTS5 virtual table and returns BM25-ranked hits
  with the same filters applied.

`retrieval/fusion.py`'s `reciprocal_rank_fusion(rankings, k=60)` combines
any number of best-first `RankedHit` lists by **rank position, not raw
score** -- `score(chunk) = Σ 1/(k + rank)` over every ranking it appears
in. This is deliberate: cosine similarity lives in `[-1, 1]` and BM25 is an
unbounded, more-negative-is-better weight (negated in `db.search_fts` so
"higher is better" holds everywhere above the database layer); RRF is the
standard way to fuse rankings on incomparable scales without inventing an
arbitrary normalization.

`search.py` is the composition root: it embeds the question once, always
runs `VectorStrategy`, and -- unless `mode="vector"` -- also runs
`FtsStrategy` and fuses both with `reciprocal_rank_fusion`. This is the
hard invariant the hybrid-search feature was built around: **hybrid search
augments the existing section-based contextual retrieval, it doesn't
replace it.** Both arms rank the exact same corpus of contextualized,
embedded chunks; FTS5 is indexed from the same `raw_text` + `context` that
gets embedded, not from some separate, less-processed copy of the text.

### Recipe: add a new retrieval strategy

1. Write a class in `retrieval/<name>.py` implementing `RetrievalStrategy`.
2. Re-export it from `retrieval/__init__.py`.
3. Edit `search.py`'s `search()` function -- this is a real code change to
   the composition root, not a registration: decide whether the new
   strategy always runs, runs only in some mode (extending `_MODES`), or is
   one more arm fused via `reciprocal_rank_fusion`, and write that branch.
4. Add tests: a focused one for the strategy's `run()` against seeded rows,
   and (if it changes fused ranking) an integration test in
   `tests/test_search.py` with hand-built vectors/text proving the new
   arm changes the outcome relative to the old set of strategies.

## Why FTS5 is a standalone table, not external-content

`chunks_fts` (declared in `db.py`'s `_SCHEMA`) is a **standalone** FTS5
virtual table with `UNINDEXED` natural-key columns
(`book_id, chapter_index, section_index, chunk_index`), populated by
`db.index_chunk_fts` and queried by `db.search_fts`. SQLite's documented
"recommended" pattern links an FTS5 table to a content table via
`content='chunks', content_rowid='rowid'`, which assumes the content table
has a single-column integer rowid/PK. `chunks` has a four-column composite
primary key and no surrogate rowid alias, so external-content mode would
have required either adding a surrogate integer key to `chunks` purely to
satisfy FTS5, or hand-rolling the sync triggers external-content mode
normally generates for you. A standalone table avoids both: it's just
another row store that happens to be full-text-indexed, kept in sync by
the same call sites that already write to `chunks` (`embed.py`, plus
`db.rebuild_fts_index` for from-scratch repopulation).

## Filters and metadata

`books` carries `author` (best-effort from PDF/EPUB metadata; absent for
markdown/text) and `source_type` (always set, defaulting to `pdf` for
backward compatibility with pre-hybrid-search databases via
`_migrate_books_table`'s `ALTER TABLE`). `book_id`, `author`, and
`source_type` are accepted as filters by `db.get_search_rows`,
`db.search_fts`, both `RetrievalStrategy` implementations, `search.search`,
the `query` CLI command, and the `search_book` MCP tool -- the same three
filters, the same names, at every layer. Date filtering (`--after` /
`--before`) was considered and declined: neither PDF nor EPUB metadata
reliably exposes a publication date, and `created_at` is an ingest
timestamp, not a publication date -- filtering on it would be misleading,
not useful.

## Interfaces stay thin

`cli.py` (Typer) and `mcp_server.py` (FastMCP) only parse arguments, load
config, open a connection, call into `ingest` / `search` / `db`, and format
output. Neither contains retrieval or ingestion logic. The MCP server is
strictly read-only by construction: it has exactly five tools
(`list_books`, `get_book_outline`, `search_book`, `get_section`,
`book_status`), none of which can mutate the database; `ingest`, `delete`,
and `reindex-fts` are CLI-only and are never imported into `mcp_server.py`.
`get_book_outline` is the navigational counterpart to `search_book`: it
returns a book's chapter/section tree (indices, titles, page_start, and a
cheap `approx_tokens` size hint) so an agent can resolve "the section
about X" to a real `(chapter_index, section_index)` pair for `get_section`
without first needing a `search_book` hit to supply one. It's a pure read
over `chapters`/`sections` -- no Ollama call, no embedding.
