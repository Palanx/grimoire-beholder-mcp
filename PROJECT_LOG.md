# PROJECT_LOG.md

Living state + append-only decision history. Distinct from
[ARCHITECTURE.md](ARCHITECTURE.md) (how the system is built, permanent
structure) and [AGENTS.md](AGENTS.md) (operating rules). This file answers
"where are we, and why did we do that."

**Maintenance rule — every future session that changes the project must, before finishing:**
1. Overwrite the **Orientation** section below with current reality.
2. Append one entry to the **Decision log** below, dated, reverse-chronological. Never edit or delete a past entry.
3. If the change was structural, update ARCHITECTURE.md too; if it changed a rule/command/invariant, update AGENTS.md too. Reference, don't restate.

---

## Orientation (overwritten every update — last updated 2026-06-18)

**Current state:**
- MCP server exposes exactly 5 read-only tools: `list_books`, `get_book_outline`, `search_book`, `get_section`, `book_status`. No mutating tool is, or should be, exposed there.
- 4 source formats parse via the `SourceParser` registry: `.pdf`, `.epub`, `.md`/`.markdown`, `.txt`.
- PDF chapter detection is now a three-tier fallback: embedded outline -> LLM-extracted TOC (new) -> heading regex scan. The LLM-TOC tier only fires when a DB connection + `OllamaClient` are supplied and a printed TOC page is detected; its output is validated and offset-resolved before use, and cached in `db.py`'s `pdf_toc_cache` table (keyed by `content_hash`) so re-ingestion never re-calls the LLM. See ARCHITECTURE.md's "PDF chapter detection" section for the full algorithm and why a constant page-offset doesn't work. `SourceParser.extract()` grew three new optional kwargs (`conn`, `llm_client`, `llm_model`) to thread this through; every parser except `PdfParser` accepts and ignores them.
- Retrieval has 2 modes: `hybrid` (vector cosine + SQLite FTS5/BM25, fused by RRF — the default) and `vector` (cosine only, CLI/debug override via `--mode`, not exposed over MCP).
- Filters `book_id`/`author`/`source_type` are composable and apply identically to both retrieval arms, in the CLI (`query`) and MCP (`search_book`).
- `uv run pytest` is green: **92 passed**, 0 failed (was 80; `tests/test_sources_pdf_llm_toc.py` added, 12 tests). Ollama is mocked throughout (`tests/fakes.py:FakeOllamaClient` plus a small canned-response fake local to the new test file) — the suite needs no daemon and no pulled models.
- CLI commands: `ingest`, `list`, `delete`, `query`, `status`, `reindex-fts`, `serve-mcp`. `ingest`/`delete`/`reindex-fts` are CLI-only by design.
- Single SQLite file is the whole library + checkpoint (WAL mode, FTS5 standalone virtual table, content-hash-idempotent re-ingest, per-chunk-status resumability, now also the LLM-TOC cache).
- `.mcpb` manifest (`mcpb/manifest.json`) lists all 5 tools; the bundle ships no code/deps of its own (just wires `grimoire-beholder serve-mcp` into Claude Desktop).
- `search.search()` now calls `db.ensure_embedding_model` itself (previously only `ingest.run_ingest` did) — the CLI `query` and MCP `search_book` paths are no longer able to silently embed a query in the wrong vector space after a `config.toml` model change.
- `RetrievalStrategy` (`retrieval/`) is documented as a protocol with two hardcoded call sites in `search.py`, not a registry — don't read it as symmetrically pluggable with `SourceParser`'s `_PARSERS` list. See ARCHITECTURE.md's extension-point-2 section.

**Open / pending (deferred, not forgotten):**
- book_id 1 ("Professional C++, 5th Edition", Gregoire) was ingested before the LLM-TOC fallback existed and has the broken single-bucket chapter hierarchy that motivated this fix. It has **not** been re-ingested yet — that's the user's call to make (re-ingest needs `--force` or a delete+re-ingest, both destructive to the existing rows, so this agent deliberately did not run it). See the decision-log entry below for the exact command.
- P2 "consider" items from the local-rag idea-mining pass — proposed, not committed: JSON export of book/library metadata; an additional MCP transport (SSE/HTTP) beyond stdio; OCR for scanned PDFs via Tesseract. No design work done on any of these yet.
- Date filtering (`--after`/`--before`) on `query`/`search_book` was evaluated and declined — see 2026-06-18ish "Evolve" entry below. Don't re-propose it without a real publication-date metadata source first.
- No CI is configured; `uv run pytest` is run manually before calling a session done.

**How to resume / verify health:**
```
uv run pytest                 # expect: all passed, 0 failed
uv run grimoire-beholder --help        # confirms CLI wiring end-to-end
```
If you need to check the MCP tool surface directly without spinning up `serve-mcp`:
```
uv run python -c "import asyncio; from grimoire_beholder import mcp_server; print(sorted(t.name for t in asyncio.run(mcp_server.mcp.list_tools())))"
```

---

## Decision log (append-only, reverse-chronological — never edit or delete a past entry)

```
2026-06-18 — Add LLM-extracted TOC fallback for PDFs with no embedded outline
Did: Added a second tier to sources/pdf.py's chapter-detection fallback,
between the embedded-outline path and the regex heading scan: when a PDF
has no outline, detect a printed "Table of Contents" region across however
many leading pages it actually spans (_detect_toc_pages, anchored on a
header line, extended while line density still looks TOC-shaped), send
that text to the already-configured llm_model for strict-JSON extraction
({title, declared_page} per top-level entry), validate the raw entries
(non-empty, sane count, monotonic declared pages), then resolve each
entry's *declared* (printed) page to its *physical* PDF page. Offset
resolution does not assume one constant delta: chapter 1 is found by an
unconstrained forward scan, and every later chapter re-anchors near an
estimate from the previous chapter's own resolved offset, searching an
expanding window and never looking earlier than the previous chapter's
resolved page. A final bounds check (in-range, strictly increasing) runs
before anything is trusted. A validated result is cached in a new
pdf_toc_cache table (db.py), keyed by content_hash, so re-ingesting the
same file never re-calls the LLM. Any validation failure at any stage logs
why and falls back to the pre-existing heading-regex scan -- this fallback
either returns a fully validated result or returns None, never an
unvalidated one. SourceParser.extract() gained three optional kwargs
(conn, llm_client, llm_model) to thread this through from ingest.py;
EpubParser/MarkdownParser/PlaintextParser accept and ignore them so the
call site in ingest.py stays uniform. Added tests/test_sources_pdf_llm_toc.py
(12 tests, all Ollama-free via a small canned-response fake) with a 9-page
synthetic PDF fixture (llm_toc_pdf_path in conftest.py) that reproduces the
triggering bug directly: no embedded outline, a TOC spanning physical pages
2-3 with declared page numbers (1, 6, 10) that resolve to different
physical pages (5, 7, 8) by a different offset each time, and a page whose
literal first line is "Chapter 4: Counting Mistakes" -- an inline
appendix sub-heading that the regex-only fallback misreads as its own
chapter (verified directly: test_heading_only_fallback_would_misfire_on_this_fixture
asserts the bug is real on this fixture), but which the LLM-TOC path never
promotes. Updated ARCHITECTURE.md (new "PDF chapter detection: a
three-tier fallback" subsection) and AGENTS.md (new invariant #11: never
use an unvalidated LLM-extracted TOC; db.py's table row in "where things
live" now mentions the TOC cache).

Why: book_id 1, "Professional C++, 5th Edition" (Gregoire), has no
embedded PDF outline -- its table of contents is hardcoded text on the
first few pages. The existing heading-regex fallback is too weak for it in
both directions: it misses the real chapter headings (they don't match
"Chapter N"-style patterns at the top of a page) while it *does* match a
literal "CHAPTER N:" string quoted inline inside Appendix A's body text,
promoting that false text into a chapter boundary while the entire real
book collapses into one giant synthetic chapter. An LLM-based TOC
extraction step, gated strictly behind validation and offset resolution,
fixes both failure modes without weakening the embedded-outline path (still
tried first, unchanged) or removing the regex fallback (still the last
resort, unchanged). The non-constant-offset design specifically came from
checking Gregoire's real numbers: a TOC-declared page 304 corresponds to
physical PDF page 346, a ~42-page front-matter delta that is not safe to
assume is identical for every chapter in the book (color plates, blank
pages, and part-divider pages all shift it further in different places).
Caching by content_hash (not book_id) was chosen because extraction runs
before the book row exists, and it makes re-ingesting the same file under a
different slug or name reuse the cache for free -- consistent with this
project's existing content-hash-idempotency pattern, not a new mechanism.

Affects: src/grimoire_beholder/sources/pdf.py, src/grimoire_beholder/sources/__init__.py,
src/grimoire_beholder/sources/epub.py, src/grimoire_beholder/sources/plaintext.py,
src/grimoire_beholder/ingest.py, src/grimoire_beholder/db.py,
tests/test_sources_pdf_llm_toc.py (new), tests/conftest.py, ARCHITECTURE.md, AGENTS.md.

Follow-ups: book_id 1 has not been re-ingested under this fix yet -- that
is the user's call, not run automatically by this session. To re-ingest
once Ollama is running with the configured llm_model pulled (cogito by
default): either re-run the original `uv run grimoire-beholder ingest
<path-to-professional-c++.pdf> --force` (replaces the existing book_id 1
row and its chapters/sections/chunks, re-running the full pipeline -- this
is destructive to the old broken rows by design), or `uv run
grimoire-beholder delete 1` followed by a plain `ingest` of the same file.
Either way, expect one new LLM call to extract the TOC (cached afterward)
plus the normal summarize/contextualize/embed calls ingest already makes.
uv run pytest: 92 passed (was 80).
```

```
2026-06-18 — Architecture review fixes: embedding-model guard, page_start bug, double-fetch, CLI tests, honest RetrievalStrategy docs
Did: Fixed five issues an architecture review surfaced, all small and
local, no rewrites: (1) search.search() now calls db.ensure_embedding_model
itself, closing a gap where query/search_book could silently embed against
a mismatched model -- only ingest.run_ingest checked this before. (2)
sources/common.py's auto_split_paragraphs hardcoded page_start=0 for its
empty-input fallback; gave it a fallback_location parameter and had
plaintext.py pass the chapter's real starting location, fixing a
reproducible bug where a markdown file with two adjacent headings and no
body between them produced a section sorting before its own chapter. (3)
VectorStrategy used to re-fetch db.get_search_rows itself even though
search.py already fetches the same filtered rows for the post-fusion
metadata lookup -- it now takes rows via its constructor, fetched once.
(4) Added tests/test_cli.py (5 tests) -- cli.py had zero dedicated tests
before this. (5) Reworded ARCHITECTURE.md's extension-point-2 section and
AGENTS.md's one-liner: RetrievalStrategy is a protocol with two hardcoded
call sites in search.py, not a registry like SourceParser's _PARSERS list;
adding a strategy means editing search()'s branching, not appending to a
list. The docs previously presented both as parallel "two explicit
extension points" without flagging that asymmetry.

Why: All five came from actually reading the code against what
ARCHITECTURE.md/AGENTS.md claimed, rather than trusting the docs. (1) was a
real invariant violation (AGENTS.md claimed model-mismatch was "enforced on
every connect()" -- it wasn't, anywhere on the search path). (2) was
verified live (reproduced the page_start=0 sort inversion, then confirmed
the fix). (3)-(4) are the kind of small, low-risk gaps a thin-interface,
read-only architecture invites if "thin" gets read as "doesn't need
tests/doesn't matter how many times we query." (5) matters because an
agent reading "two explicit extension points" with parallel recipes will
reasonably assume parallel effort to extend either one; only one of them
actually is.

Affects: src/grimoire_beholder/search.py, src/grimoire_beholder/sources/common.py,
src/grimoire_beholder/sources/plaintext.py, src/grimoire_beholder/retrieval/vector.py,
tests/test_cli.py (new), ARCHITECTURE.md, AGENTS.md.

Follow-ups: None outstanding. uv run pytest: 80 passed (was 75).
```

```
2026-06-18 — Add continuity docs: AGENTS.md + PROJECT_LOG.md
Did: Created AGENTS.md (operating rules/invariants/commands) and this file
(living state + decision log), alongside the existing ARCHITECTURE.md.

Why: A future agent (or a future me) needs to enter this repo cold and
both behave correctly (AGENTS.md) and know what's already been decided and
why, so settled debates (date filtering, EPUB page_start, Python over Go)
don't get silently reopened (PROJECT_LOG.md). Keeping these as three
strictly separate files, cross-referenced rather than duplicated, was
chosen over one large doc so each has a single clear purpose and doesn't
rot into overlapping, inconsistent copies of the same facts.

Affects: AGENTS.md (new), PROJECT_LOG.md (new). No code changed.

Follow-ups: None. Next agent: follow the maintenance rule at the top of
this file at the end of your session.
```

```
2026-06-18 — Add get_book_outline MCP tool
Did: Added a 5th read-only MCP tool, get_book_outline(book_id), returning
a book's chapter/section tree (indices, titles, page_start, and a cheap
approx_tokens size hint per section — never full section text). Auto-split
sections with no native title get a synthesized "Section N -- <snippet>"
label so they stay identifiable. Added db.Chapter dataclass, db.list_chapters,
db.list_sections. Updated mcpb/manifest.json's tool list and README/
ARCHITECTURE's tool-count references (four -> five tools).

Why: get_section(book_id, chapter_index, section_index) needs indices that
nothing previously surfaced -- an agent could only discover them via a
search_book hit. get_book_outline is the missing map: list_books ->
get_book_outline(book_id) -> get_section(...) for targeted reading,
alongside search_book for semantic lookup. Kept deliberately lightweight
(no section text) so it stays a cheap "map" call, distinct from the "read"
call get_section already is -- considered and rejected returning section
text previews inline, since that would blur the two tools' roles and bloat
every outline response for books with many sections.

Affects: src/grimoire_beholder/db.py, src/grimoire_beholder/mcp_server.py,
mcpb/manifest.json, README.md, ARCHITECTURE.md, tests/test_db.py,
tests/test_mcp_server.py.

Follow-ups: None outstanding.
```

```
(reconstructed — exact date unknown) — Evolve: hybrid search (vector + FTS5 + RRF) and pluggable source types
Did: Mined github.com/sebastianhutter/local-rag (a Go tool) for ideas only
(no porting, no language switch) and added: SQLite FTS5 keyword search
fused with existing cosine vector search via Reciprocal Rank Fusion;
book metadata (author, source_type) with composable book_id/author/
source_type filters on both CLI query and MCP search_book; a SourceParser
extension point with EPUB and markdown/plaintext parsers added alongside
the original PDF parser. Wrote ARCHITECTURE.md to document the resulting
layering and both extension points (SourceParser, RetrievalStrategy).

Why (the load-bearing "why nots"):
- RRF over weighted score blending: cosine similarity ([-1,1]) and BM25
  (unbounded) live on incomparable scales; RRF fuses by rank position, not
  raw score, avoiding an arbitrary, fragile normalization/weight constant.
- Standalone chunks_fts table over SQLite's "recommended" external-content
  FTS5 mode: external-content mode assumes a single-column integer rowid
  on the content table, but `chunks` has a 4-column composite primary key
  and no rowid alias -- external-content mode would have meant either
  adding a surrogate key purely to satisfy FTS5, or hand-rolling its sync
  triggers. A standalone table, kept in sync by the same call sites that
  already write to `chunks`, avoids both.
- Hybrid search augments retrieval, it does not replace it: search.search()
  always runs VectorStrategy first; FtsStrategy is strictly additive. This
  was a hard requirement, not a default we could have silently dropped.
- page_start for EPUB/markdown/text is a synthetic, strictly-increasing
  ordinal, not a real page number -- and we explicitly decided not to
  invest in approximating a "real" page number for these formats. Reason:
  people who need precise page citations are almost always citing
  technical/reference PDFs, which already have real page numbers; EPUB/
  markdown/text ingestion mainly serves fiction/general non-fiction where
  an approximate, consistently-ordered citation is an acceptable tradeoff.
  Don't revisit this without a concrete case where it actually matters.
- Declined: --after/--before date filtering. Neither PDF nor EPUB metadata
  reliably exposes a publication date, and created_at is an ingest
  timestamp, not a publication date -- filtering on it would mislead, not
  help. Don't add it on a database field that doesn't mean what it'd
  imply; would need a real metadata source first.
- Considered, not implemented (proposed-not-forced P2 items): JSON export
  of book/library metadata; an additional MCP transport (SSE/HTTP) beyond
  stdio; OCR for scanned PDFs (Tesseract). None had a forcing requirement;
  left as future work, not oversights.
- Explicitly skipped (out of scope, not just deferred): local-rag's email/
  RSS/source-code/Obsidian-vault ingestion, its menubar GUI app, and its
  DMG installer -- grimoire-beholder-mcp is a focused multi-book RAG library, not a
  general ingestion platform or a desktop app.
- MCP stayed strictly read-only: ingest/delete/reindex-fts are CLI-only
  and never wired into mcp_server.py, so a chat agent can search and read
  a library but can never mutate it -- removes a whole risk class (an
  agent corrupting or deleting a user's library) for a capability nobody
  asked for.

Affects: src/grimoire_beholder/sources/ (new package), src/grimoire_beholder/retrieval/ (new
package), src/grimoire_beholder/db.py, src/grimoire_beholder/search.py (rewritten),
src/grimoire_beholder/embed.py, src/grimoire_beholder/ingest.py, src/grimoire_beholder/cli.py,
src/grimoire_beholder/mcp_server.py, src/grimoire_beholder/config.py, mcpb/manifest.json,
ARCHITECTURE.md (new), README.md, full test suite (38 -> 72 tests).

Follow-ups: P2 items above remain unimplemented by design. get_book_outline
(see newer entry above) was a direct follow-up gap this work exposed.
```

```
(reconstructed — exact date unknown) — Initial grimoire-beholder-mcp build: hierarchy + contextual retrieval + Ollama DI seam
Did: Built the original grimoire-beholder-mcp: PDF-only ingestion into a Book -> Chapter
-> Section -> Chunk hierarchy; Anthropic's Contextual Retrieval technique
(LLM-generated context per chunk, scoped to its section's summary) before
embedding; a single shared SQLite library file as both the index and the
crash-safe checkpoint, with resumability driven by a per-chunk status
column (pending -> contextualized -> embedded); an OllamaClient Protocol
seam (RealOllamaClient / FakeOllamaClient) so the pipeline is fully
unit-testable with no daemon; a Typer CLI and a 4-tool read-only FastMCP
server (list_books, search_book, get_section, book_status).

Why (the load-bearing "why nots"):
- Python + uv chosen over Go or Node: this is the project's own language
  going in (not a port of anything), and uv gives a single-binary-feeling,
  fast, reproducible toolchain without a interpreter-management story to
  maintain. (This choice was later reaffirmed, not revisited, when the
  hybrid-search/local-rag evolution work explicitly mined a Go codebase
  for ideas only and ruled out porting it or switching ecosystems.)
- Section level between chapter and chunk: a single summary for a long,
  dense chapter (e.g. 40 pages of philosophy) is too generic to usefully
  situate every chunk inside it; sections get their own focused summary,
  keeping per-chunk context tight even in unstructured books.
- SQLite as the whole index instead of a vector-DB service (Chroma/Qdrant/
  pgvector/etc.): fully-offline, single-file, zero-extra-infrastructure
  requirement -- the only running service this project ever needs is
  Ollama itself.
- Resumability via an inline status column instead of a separate
  checkpoint file: the database already is the durable state; a crash
  anywhere loses at most one in-flight summary, one chunk's context, or
  one embedding batch, and "resume" is just "run ingest again" -- no
  separate resume command or state file to keep in sync.
- MCP server scoped to exactly 4 tools, all read-only, from the start:
  ingest/delete were CLI-only by design before hybrid search ever existed
  -- this wasn't a retrofit, it was the original posture.

Affects: src/grimoire_beholder/{cli,config,contextualize,db,embed,extract,ingest,
mcp_server,ollama_client}.py (original layout; `extract.py` was later
replaced by the sources/ package), tests/ (original suite).

Follow-ups: see the "Evolve" entry above for what was built on top of this.
```
