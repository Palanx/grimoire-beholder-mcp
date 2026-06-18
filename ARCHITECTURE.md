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
    def extract(self, path: Path, section_split_tokens: int = 3000) -> ExtractedBook: ...
```

`get_parser(path)` walks a fixed list of registered parser instances and
returns the first whose `can_parse` claims the file, raising `ValueError`
if none do. `ingest.py` calls `get_parser` once per ingest and never knows
which concrete parser it got -- everything after `extract()` (chunking,
contextualization, embedding, indexing) operates on the same
`ExtractedBook` / `Chapter` / `Section` dataclasses (`sources/common.py`)
regardless of source format.

Currently registered: `PdfParser` (`sources/pdf.py`, table of contents ->
chapters, falling back to heading regexes), `EpubParser` (`sources/epub.py`,
one chapter per spine document via `ebooklib`), and `MarkdownParser` /
`PlaintextParser` (`sources/plaintext.py`, sharing one paragraph-based
chapter/section splitter). `sources/common.py`'s `auto_split_paragraphs`
is the shared greedy section-packer used by every parser except PDF's
(PDF keeps its own, separately tested, page-aware variant to avoid any
behavior change to existing PDF tests).

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

- `VectorStrategy` (`retrieval/vector.py`): fetches every embedded chunk
  matching the filters via `db.get_search_rows`, computes cosine similarity
  against `query_vector` in numpy, returns the top `pool_size`.
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
3. Wire it into `search.py`'s `search()` function -- decide whether it
   always runs, runs only in some mode, or is one more arm fused via
   `reciprocal_rank_fusion`.
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
