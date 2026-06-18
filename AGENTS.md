# AGENTS.md

Operating rules for AI agents working in this repo. This file is the
rulebook (terse, imperative) — read it every session. For *how the system
is built*, see [ARCHITECTURE.md](ARCHITECTURE.md). For *current state and
why past decisions were made*, see [PROJECT_LOG.md](PROJECT_LOG.md).

## Setup & commands

- Install deps: `uv sync` — **uv only, never `pip install`, never hand-edit `.venv`.**
- Run the test suite: `uv run pytest`
- Run the CLI: `uv run grimoire-beholder <command>` — `ingest`, `list`, `delete`, `query`, `status`, `reindex-fts`, `serve-mcp`
- Start the MCP server directly: `uv run grimoire-beholder serve-mcp`
- Add a dependency: `uv add <package>` (never edit `pyproject.toml`'s `dependencies` by hand)
- Rebuild the `.mcpb` bundle after touching `mcpb/manifest.json`: `mcpb validate mcpb/manifest.json && mcpb pack mcpb grimoire-beholder-mcp.mcpb`

## Hard invariants — do not violate

1. **MCP server is strictly read-only.** Exactly five tools in `mcp_server.py`: `list_books`, `get_book_outline`, `search_book`, `get_section`, `book_status`. Never import `ingest`, `delete_book`, or `rebuild_fts_index` call paths into it — those stay CLI-only.
2. **All Ollama access goes through `OllamaClient`** (`ollama_client.py`). Never `import ollama` in `ingest.py`, `contextualize.py`, `embed.py`, `search.py`, or `cli.py`/`mcp_server.py` logic — take a client as a parameter. This is what lets the whole pipeline run under test with `FakeOllamaClient` (`tests/fakes.py`) and no daemon.
3. **`db.py` is the only module that touches `sqlite3`.** Every other module calls a `db.*` function; none opens its own connection/cursor or writes raw SQL.
4. **A chunk never crosses a section boundary.** `chunk_section`/`split_text` is always called with exactly one section's text.
5. **One embedding model per database**, enforced by `db.ensure_embedding_model` on every `connect()`. Never bypass or weaken this check.
6. **Hybrid search augments vector search, never replaces it.** `search.search()` always runs `VectorStrategy`; `FtsStrategy` is additive in `mode="hybrid"` (the default) and skipped only in `mode="vector"`. Both arms must always receive the same filters (`book_id`/`author`/`source_type`) and rank the same contextualized, embedded chunk corpus.
7. **`cogito` runs in standard mode only** — never set a system/prompt that triggers its deep-thinking subroutine. Keep the defensive `<think>...</think>` strip in `RealOllamaClient.generate` even if it looks unreachable.
8. **Natural composite keys only** (`book_id, chapter_index, section_index[, chunk_index]`) on `chapters`/`sections`/`chunks`. Inserts must stay idempotent (`INSERT OR REPLACE` / `ON CONFLICT DO NOTHING`); never add a surrogate id to these tables.
9. **FTS5 availability is probed at `db.connect()`** and must fail loudly (`FTS5UnavailableError`) — never silently degrade to vector-only.
10. **Keep `uv run pytest` green.** New logic ships with tests; mock Ollama via `FakeOllamaClient`, never a real daemon, in a test.

## Conventions

- Type hints + a short docstring on every public function; docstrings explain **why**, not what.
- Dependency rule (one-way): `cli.py`/`mcp_server.py` → `ingest.py` → `{sources/, chunk.py, contextualize.py, embed.py}` → `search.py` → `retrieval/` → `db.py`. Never reach sideways or upward.
- Two explicit extension points: `SourceParser` (`sources/`) and `RetrievalStrategy` (`retrieval/`). Use ARCHITECTURE.md's recipes to add a new one — don't invent a different registration pattern.
- No comments stating what code does; only why, when it's non-obvious.

## Where things live

| module | role |
|---|---|
| `cli.py` | Typer CLI: ingest, list, delete, query, status, reindex-fts, serve-mcp |
| `mcp_server.py` | FastMCP server, 5 read-only tools |
| `ingest.py` | per-book orchestrator: extract → load → summarize → contextualize → embed |
| `sources/` | `SourceParser` impls (pdf, epub, plaintext/markdown) + registry |
| `chunk.py` | recursive token-budget text splitter |
| `contextualize.py` | section summaries + per-chunk context via Ollama |
| `embed.py` | batched embedding + FTS5 indexing |
| `search.py` | hybrid retrieval composition root |
| `retrieval/` | `VectorStrategy`, `FtsStrategy`, RRF fusion, `RetrievalStrategy` protocol |
| `db.py` | sole SQLite touchpoint: schema, CRUD, FTS5, embedding-model guard |
| `ollama_client.py` | `OllamaClient` protocol + `RealOllamaClient` |
| `config.py` | `config.toml` loader with built-in defaults |
| `mcpb/` | `.mcpb` bundle manifest + doc stub (no bundled code/deps) |
| `tests/` | mirrors `src/`; `fakes.py` has `FakeOllamaClient`, `conftest.py` has shared fixtures |

## End of session

1. Overwrite `PROJECT_LOG.md`'s orientation block (top section).
2. Append one decision-log entry to `PROJECT_LOG.md`, dated today.
3. If the change was structural, update `ARCHITECTURE.md`. If it changed a rule/command/invariant, update this file. Reference the other docs — don't restate them.
