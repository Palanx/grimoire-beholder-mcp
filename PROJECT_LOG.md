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
- Retrieval has 2 modes: `hybrid` (vector cosine + SQLite FTS5/BM25, fused by RRF — the default) and `vector` (cosine only, CLI/debug override via `--mode`, not exposed over MCP).
- Filters `book_id`/`author`/`source_type` are composable and apply identically to both retrieval arms, in the CLI (`query`) and MCP (`search_book`).
- `uv run pytest` is green: **75 passed**, 0 failed. Ollama is mocked throughout (`tests/fakes.py:FakeOllamaClient`) — the suite needs no daemon and no pulled models.
- CLI commands: `ingest`, `list`, `delete`, `query`, `status`, `reindex-fts`, `serve-mcp`. `ingest`/`delete`/`reindex-fts` are CLI-only by design.
- Single SQLite file is the whole library + checkpoint (WAL mode, FTS5 standalone virtual table, content-hash-idempotent re-ingest, per-chunk-status resumability).
- `.mcpb` manifest (`mcpb/manifest.json`) lists all 5 tools; the bundle ships no code/deps of its own (just wires `book-rag serve-mcp` into Claude Desktop).

**Open / pending (deferred, not forgotten):**
- P2 "consider" items from the local-rag idea-mining pass — proposed, not committed: JSON export of book/library metadata; an additional MCP transport (SSE/HTTP) beyond stdio; OCR for scanned PDFs via Tesseract. No design work done on any of these yet.
- Date filtering (`--after`/`--before`) on `query`/`search_book` was evaluated and declined — see 2026-06-18ish "Evolve" entry below. Don't re-propose it without a real publication-date metadata source first.
- No CI is configured; `uv run pytest` is run manually before calling a session done.

**How to resume / verify health:**
```
uv run pytest                 # expect: all passed, 0 failed
uv run book-rag --help        # confirms CLI wiring end-to-end
```
If you need to check the MCP tool surface directly without spinning up `serve-mcp`:
```
uv run python -c "import asyncio; from book_rag import mcp_server; print(sorted(t.name for t in asyncio.run(mcp_server.mcp.list_tools())))"
```

---

## Decision log (append-only, reverse-chronological — never edit or delete a past entry)

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

Affects: src/book_rag/db.py, src/book_rag/mcp_server.py,
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
  DMG installer -- book-rag is a focused multi-book RAG library, not a
  general ingestion platform or a desktop app.
- MCP stayed strictly read-only: ingest/delete/reindex-fts are CLI-only
  and never wired into mcp_server.py, so a chat agent can search and read
  a library but can never mutate it -- removes a whole risk class (an
  agent corrupting or deleting a user's library) for a capability nobody
  asked for.

Affects: src/book_rag/sources/ (new package), src/book_rag/retrieval/ (new
package), src/book_rag/db.py, src/book_rag/search.py (rewritten),
src/book_rag/embed.py, src/book_rag/ingest.py, src/book_rag/cli.py,
src/book_rag/mcp_server.py, src/book_rag/config.py, mcpb/manifest.json,
ARCHITECTURE.md (new), README.md, full test suite (38 -> 72 tests).

Follow-ups: P2 items above remain unimplemented by design. get_book_outline
(see newer entry above) was a direct follow-up gap this work exposed.
```

```
(reconstructed — exact date unknown) — Initial book-rag build: hierarchy + contextual retrieval + Ollama DI seam
Did: Built the original book-rag: PDF-only ingestion into a Book -> Chapter
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

Affects: src/book_rag/{cli,config,contextualize,db,embed,extract,ingest,
mcp_server,ollama_client}.py (original layout; `extract.py` was later
replaced by the sources/ package), tests/ (original suite).

Follow-ups: see the "Evolve" entry above for what was built on top of this.
```
