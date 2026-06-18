# book-rag

A self-contained, fully-offline RAG library for PDFs, EPUBs, markdown, and
plain text, built for Apple Silicon. It's a **library**, not a single book:
ingest as many books as you like, in any mix of supported formats, into one
shared index. Each book is parsed into a **Book → Chapter → Section → Chunk**
hierarchy, every chunk is enriched with LLM-generated context (Anthropic's
Contextual Retrieval technique) scoped to its section, embedded locally via
Ollama, and stored in a single SQLite file that doubles as a crash-safe,
resumable checkpoint. Retrieval is **hybrid by default**: vector (cosine)
search and SQLite FTS5 keyword/BM25 search both run over the same
contextualized chunks, fused with Reciprocal Rank Fusion -- see
[ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and how
to extend them. A built-in MCP server exposes the whole library to Claude
as read-only tools.

Getting book-rag running is two separate jobs: **setting it up** (Python
deps, Ollama, models, and at least one ingested book -- all manual, all
local) and **connecting it to Claude Desktop** (one click, via a `.mcpb`
bundle). Only the second part is "one click" -- there is no zero-prerequisite
install. Do Setup first; the bundle does not do it for you.

## Supported source types

| extension | parser | chapters from | `page_start` is |
|-----------|--------|----------------|-----------------|
| `.pdf` | PDF | table-of-contents level-1 entries (falls back to heading detection) | a real 1-indexed PDF page number |
| `.epub` | EPUB | one chapter per spine document, titled from the EPUB's nav/TOC | a synthetic, strictly increasing location ordinal (not a real page) |
| `.md`, `.markdown` | Markdown | top-level (`# `) headings | a 1-based paragraph ordinal within the file |
| `.txt` | Plain text | the whole file is one chapter | a 1-based paragraph ordinal within the file |

`ingest` picks the parser by file extension automatically -- there's no flag
to set. Everything downstream of parsing (sectioning, chunking,
contextualization, embedding, indexing, retrieval) is identical regardless
of source type. See [ARCHITECTURE.md](ARCHITECTURE.md) for how to add
another one.

## Why sections?

Long, dense chapters (e.g. a 40-page chapter on philosophy or psychology)
are too broad for one summary to usefully situate every chunk inside them.
book-rag inserts a **Section** level between chapter and chunk, derived per
chapter with this priority:

1. If the source has sub-headings under the chapter (a PDF's TOC
   sub-entries; nothing for EPUB/markdown/text), each one becomes a section.
2. Otherwise, if the chapter is longer than `section_split_tokens`
   (~3000 tokens by default), it's auto-split into ~3000-token sections,
   breaking on paragraph boundaries where possible.
3. Otherwise, the whole chapter is a single section.

This hierarchy always exists, and a chunk never crosses a section boundary.
Chunk context is generated from its **section's** summary, not the whole
chapter, so contextualization stays tight even in dense, unstructured books.

## Hybrid search

`query` and `search_book` rank chunks with two independent retrieval
strategies over the same contextualized, embedded chunks:

- **Vector**: cosine similarity between the query embedding and every
  chunk's embedding.
- **FTS5**: SQLite's full-text index (BM25-ranked) over each chunk's raw
  text and generated context.

Both arms run with the same filters (`book_id`, `author`, `source_type`),
each returning its own top-`candidate_pool_size` candidates, and are fused
with **Reciprocal Rank Fusion** (`score = Σ 1/(rrf_k + rank)` per chunk,
summed across whichever ranking(s) it appears in). RRF combines rankings by
relative position, not raw score, which is what makes it possible to fuse
cosine similarity (bounded, `[-1, 1]`) with BM25 (unbounded) at all. A
keyword match that vector search alone would have missed or under-ranked
can out-rank a vector-only hit, and vice versa.

Set `retrieval_mode = "vector"` in `config.toml` (or pass `--mode vector`
to `query` for a one-off) to disable the FTS5 arm and fall back to pure
cosine ranking.

The FTS5 index is populated incrementally as chunks are embedded -- no
separate indexing step. If you ever need to rebuild it from scratch (e.g.
after restoring an old database backup), run `book-rag reindex-fts`.

## Setup (manual, run once, in order)

Everything here is local. Run these in order, in the directory you want to
use as your library (where `config.toml` and `book.db` will live):

1. **Install Python dependencies:**

   ```
   uv sync
   ```

   Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`); it pins
   Python 3.12 and installs everything else for you.

2. **Pull the LLM model** (used for section summaries and chunk context):

   ```
   ollama pull cogito:8b
   ```

3. **Pull the embedding model:**

   ```
   ollama pull nomic-embed-text
   ```

   These are the defaults in `config.toml` -- if you've changed
   `llm_model` / `embedding_model` there, pull whatever you set instead.

4. **Confirm Ollama is running** at `http://localhost:11434` (`ollama serve`,
   or just have the Ollama app open). `book-rag` checks for required models
   on every run and refuses to proceed (with the exact `ollama pull ...`
   command) if Ollama is unreachable or a model is missing -- it never pulls
   one for you.

5. **Ingest at least one book** (PDF, EPUB, markdown, or plain text):

   ```
   uv run book-rag ingest path/to/book.pdf [--name slug]
   ```

   This is slow (every section gets an LLM summary, every chunk gets LLM
   context and an embedding) but fully resumable -- interrupting it with
   Ctrl-C is fine, re-running the same command picks up where it left off
   instead of starting over. See **How resume works** below.

Once you've done this once, the library is ready to query from the CLI
(`uv run book-rag query "..."`) and ready to connect to Claude.

## Usage

```
uv run book-rag ingest "<path-to-book.[pdf|epub|md|txt]>" [--name "Display Name"] [--force]
uv run book-rag list
uv run book-rag delete <slug> [--yes]
uv run book-rag query "<your question>" [--book <slug>] [--author <name>] [--type <pdf|epub|markdown|text>] [--mode hybrid|vector] [--expand]
uv run book-rag status
uv run book-rag reindex-fts [--book <slug>]
uv run book-rag serve-mcp
```

- **`ingest`** picks a parser by file extension (see **Supported source
  types** above), extracts chapters and sections, chunks each section,
  generates a per-section situating summary and per-chunk context with the
  LLM model, and embeds and FTS5-indexes every chunk. The book's display
  name (and the slug it's stored under) defaults to title metadata from the
  file itself where available, falling back to the filename; override it
  with `--name`. Author and source type are recorded automatically.
  Re-running `ingest` on the same file is idempotent and resumable. If a
  different file would collide with an existing slug, ingest refuses unless
  you pass `--force` to replace it.
- **`list`** shows every book in the library with its author, source type,
  page count, chapter count, section count, and chunk status breakdown.
- **`delete <slug>`** permanently removes a book and everything under it
  (chapters, sections, chunks, embeddings, FTS5 rows) in one transaction. It
  prompts for confirmation unless you pass `--yes`. **This command is
  CLI-only and is never exposed to Claude or the MCP server.**
- **`query`** embeds your question and ranks chunks with hybrid (vector +
  FTS5, RRF-fused) search by default across the whole library, printing the
  top matches with book/chapter/page citations. Scope or filter with
  `--book <slug>`, `--author <name>`, and/or `--type <pdf|epub|markdown|text>`
  (composable); override the retrieval mode for one query with
  `--mode vector` (debug/comparison only -- `config.toml`'s
  `retrieval_mode` is the persistent setting). Pass `--expand` to print each
  hit's full parent section text instead of just the chunk. It never calls
  a cloud LLM.
- **`status`** prints the configured models, the database path, and every
  book's chapter/section/chunk counts, including how many chunks are
  `pending` / `contextualized` / `embedded`.
- **`reindex-fts`** drops and repopulates the FTS5 keyword index from every
  currently-embedded chunk, library-wide or for one `--book <slug>`. The
  index is normally kept up to date incrementally as chunks are embedded;
  this is only needed to recover a hand-edited database or an old backup.
  **CLI-only.**
- **`serve-mcp`** starts the read-only MCP server over stdio -- this is what
  the `.mcpb` bundle (and the manual config below) both launch.

`ingest`, `delete`, and `reindex-fts` are all **CLI-only by design**: none
of them are wired into the MCP server, so an agent talking to Claude can
search and read your library but can never add to, remove from, or
reindex it.

## Connect to Claude (one-click via .mcpb)

**Prerequisite: finish Setup above first.** The `.mcpb` bundle only wires an
already-working `book-rag serve-mcp` into Claude Desktop's settings -- it
does not install Python, uv, Ollama, the models, or ingest any books. If you
install it before completing Setup, Claude Desktop will show the extension
as installed but the server will fail to start the moment it's invoked.

1. Build (or download) `book-rag.mcpb` -- see **Building the bundle** below
   if you need to build it yourself.
2. In Claude Desktop, go to **Settings → Extensions → Install Extension**
   and pick `book-rag.mcpb`.
3. When prompted for configuration, fill in:
   - **book-rag project directory** -- the absolute path to this repo clone
     (where you ran `uv sync`).
   - **Library directory** -- the absolute path to the directory containing
     your `config.toml` and `book.db` (where you ran `book-rag ingest`).
     This can be the same path as the project directory, or anywhere else.

That's the "one click" part: Claude Desktop generates the server config for
you from those two paths and starts `book-rag serve-mcp` itself.

### Manual alternative (no .mcpb)

You can wire the same server in by hand by adding it to
`claude_desktop_config.json` directly:

```json
{
  "mcpServers": {
    "book-rag": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/absolute/path/to/book-rag",
        "--directory",
        "/absolute/path/to/your/library",
        "book-rag",
        "serve-mcp"
      ]
    }
  }
}
```

`--project` points at this repo (so `uv` can find the `book-rag` entry
point and its synced environment); `--directory` is the directory
containing the `config.toml` and `book.db` for the library you want Claude
to search -- it can be anywhere, and is typically *not* this repo. Both
the bundled and the manual setup ultimately run the exact same command; the
bundle just collects the two paths through a settings UI instead of you
hand-editing JSON.

### The five tools

| tool | purpose |
|------|---------|
| `list_books()` | List every book (id, slug, name, author, source type, page count). |
| `get_book_outline(book_id)` | The chapter/section map for one book -- indices, titles, page_start, and an `approx_tokens` size hint per section. No section text. |
| `search_book(question, book_id=None, top_k=None, author=None, source_type=None)` | Hybrid (vector + FTS5) search the library, optionally scoped to one book and/or filtered by exact author or source type, for cited excerpts. |
| `get_section(book_id, chapter_index, section_index)` | Fetch a section's full text and summary -- the parent of a search hit, or a section located via `get_book_outline`. |
| `book_status()` | Chapter/section/chunk status counts for every book. |

There is no ingest, delete, or reindex tool, and no cloud LLM is ever
called from the server -- the only model it invokes is the local embedding
model, to embed search questions. The server always uses `config.toml`'s
`retrieval_mode` (hybrid by default); the CLI-only `--mode` override has no
MCP equivalent.

`get_book_outline` exists because `get_section` needs a `chapter_index` /
`section_index` pair that nothing else surfaces -- without it, Claude has
no way to resolve "the section about X" to a real index unless it happens
to come from a `search_book` hit. Auto-split sections (no native heading)
get a synthesized title -- `"Section 3 -- <snippet of its first words>..."`
-- so they're still identifiable in the outline even with no real title.

### Asking Claude

Once connected, there are two natural flows:

- **Browse**: ask Claude to call `list_books`, then `get_book_outline` for
  one of them to see its chapters and sections, then `get_section` with a
  specific `chapter_index` / `section_index` to read one in full.
- **Search**: just ask your question -- Claude will call `search_book` and
  cite chunks back to you. If a hit looks like it's missing surrounding
  context, ask Claude to pull the full section with `get_section` using the
  hit's `book_id` / `chapter_index` / `section_index`.

### Building the bundle

The bundle source lives in `mcpb/` (`manifest.json` plus a documentation
stub -- it ships no code or dependencies; see the `long_description` in the
manifest for why). To build `book-rag.mcpb` from it:

```
npm install -g @anthropic-ai/mcpb   # one-time; the official MCPB CLI
mcpb validate mcpb/manifest.json
mcpb pack mcpb book-rag.mcpb
```

Re-run `mcpb pack` after any change to `mcpb/manifest.json`.

## Where the index lives

Everything -- books, chapters, sections, chunks, generated context, and
embeddings -- is stored in a single SQLite database, `book.db` by default
(configurable via `db_path` in `config.toml`), in the directory you run
`book-rag` from. There is no separate checkpoint file: the database *is*
the checkpoint, and it's shared across every book in the library.

All books in one database must share the same embedding model: the model
used on the very first ingest is stamped into the database, and any later
ingest -- of any book -- with a different `embedding_model` fails loudly
rather than silently mixing incompatible vector spaces. To switch embedding
models, point `db_path` at a fresh file to start a new index.

Hybrid search requires SQLite's FTS5 extension, which `book-rag` checks for
on every `connect()` and fails loudly (not silently degrading to vector-only)
if it's missing. The official python.org installers and Homebrew's `sqlite3`
both ship with it; this has not been an issue in practice.

## How resume works

Every chunk has a `status` column that moves through
`pending -> contextualized -> embedded`. Each ingest stage only looks at
chunks (or sections) in the state it cares about:

- Section summaries are written one section at a time and are skipped once
  set, so a crash loses at most one in-flight summary.
- Contextualization commits **one chunk at a time**, so a crash loses at
  most one in-flight chunk.
- Embedding processes **one batch at a time** (sequential, no concurrency)
  and commits after each whole batch, so a crash loses at most one
  in-flight batch.

Re-running `ingest` on the same PDF re-extracts and re-loads (cheap and
idempotent -- rows are keyed by book/chapter/section/chunk index, so
existing rows are never duplicated or overwritten), then picks up
summarization, contextualization, and embedding exactly where they left
off. Nothing restarts from zero.

## Configuration

All settings live in `config.toml` in the working directory, with built-in
defaults so the pipeline runs with zero edits:

| key                 | default              | meaning                                   |
|---------------------|----------------------|--------------------------------------------|
| `llm_model`          | `cogito:8b`          | model used for section summaries and chunk context |
| `embedding_model`    | `nomic-embed-text`   | model used for embeddings (locked in per-database, see above) |
| `chunk_size`         | `600`                | target chunk size, in approx. tokens (chars/4) |
| `chunk_overlap`      | `80`                 | overlap between chunks, in approx. tokens |
| `section_split_tokens` | `3000`              | chapters longer than this (with no TOC sub-headings) are auto-split into sections of about this size |
| `embed_batch_size`   | `16`                 | chunks per batched embedding request      |
| `top_k`              | `5`                  | results returned by `query` / `search_book` |
| `db_path`            | `book.db`            | path to the shared SQLite library index   |
| `retrieval_mode`     | `hybrid`             | `hybrid` (vector + FTS5, RRF-fused) or `vector` (cosine only) |
| `candidate_pool_size` | `50`                | candidates each retrieval arm contributes before fusion |
| `rrf_k`              | `60`                 | the `k` constant in Reciprocal Rank Fusion (`score = Σ 1/(k+rank)`) |

## Running the tests

The test suite mocks Ollama entirely (a fake client returns deterministic
canned text and vectors) and uses temporary SQLite databases, so it runs
with no Ollama daemon and no models pulled:

```
uv run pytest
```
