"""Typer CLI: ingest, list, delete, query, status, reindex-fts, serve-mcp."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pymupdf
import typer

from . import db, ingest as ingest_mod, ollama_client, search as search_mod
from .config import load_config
from .sources import pdf as pdf_mod

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _verify_models(required: list[str]) -> None:
    try:
        ollama_client.ensure_models_available(required)
    except (ollama_client.OllamaUnavailableError, ollama_client.MissingModelsError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def ingest(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    name: Optional[str] = typer.Option(None, help="Display name / slug source for this book."),
    force: bool = typer.Option(
        False, help="Replace an existing book with the same slug but different content."
    ),
) -> None:
    """Extract, chunk, contextualize, and embed a PDF/EPUB/markdown/text file. Fully resumable."""
    config = load_config()
    _verify_models([config.llm_model, config.embedding_model])
    client = ollama_client.RealOllamaClient(num_ctx=config.num_ctx)
    conn = db.connect(config.db_path)
    try:
        book_id = ingest_mod.run_ingest(
            conn, client, config, source, name=name, force=force, progress=typer.echo
        )
        counts = db.counts_by_status(conn, book_id)
        typer.echo(f"Done. Status counts for this book: {counts}")
    except (RuntimeError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        conn.close()


def _parse_page_range(pages: str, page_count: int) -> tuple[int, int]:
    try:
        start_str, end_str = pages.split("-", 1)
        start, end = int(start_str), int(end_str)
    except ValueError as exc:
        raise typer.BadParameter(f"--pages must be START-END (e.g. 10-41), got {pages!r}") from exc
    if not (1 <= start <= end <= page_count):
        raise typer.BadParameter(
            f"--pages {pages} is out of range for a {page_count}-page document."
        )
    return start, end


@app.command(name="dump-toc-text")
def dump_toc_text(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    pages: Optional[str] = typer.Option(
        None, "--pages", help="Override as START-END (1-indexed, inclusive), e.g. 10-41."
    ),
    out: Path = typer.Option(Path("toc_dump.txt"), "--out", help="File to write the dumped text to."),
) -> None:
    """Diagnostic: run ONLY TOC-region detection and dump the raw page text it found.

    Tests R1 (region bounding) in isolation. Reads the PDF, never touches the
    database, never calls the LLM. Pass --pages to bypass detection entirely
    and dump an explicit range instead (useful once you know the real region
    and want its exact text for the next stage).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    doc = pymupdf.open(str(source))
    page_count = doc.page_count
    raw_pages_text = [page.get_text("text") for page in doc]
    doc.close()
    raw_pages_text = pdf_mod._strip_running_headers_footers(raw_pages_text)

    if pages is not None:
        start, end = _parse_page_range(pages, page_count)
    else:
        region = pdf_mod._detect_toc_pages(raw_pages_text)
        if region is None:
            typer.secho(
                "No TOC region detected (no 'Contents'/'Table of Contents' header found).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        start, end = region

    typer.echo(f"TOC region: pages {start}-{end} (of {page_count}) -- 1-indexed, inclusive")
    text = pdf_mod._join_pages(raw_pages_text, start, end)
    out.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote {len(text)} chars to {out}")


@app.command(name="extract-toc")
def extract_toc(
    source: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="A PDF path or a .txt TOC dump."
    ),
    pages: Optional[str] = typer.Option(
        None, "--pages", help="If `source` is a PDF, override the detected TOC region as START-END."
    ),
    out: Path = typer.Option(
        Path("toc_extracted.json"), "--out", help="File to write the extracted entries to."
    ),
) -> None:
    """Diagnostic: run ONLY the LLM TOC-extraction stage, before validation/offset resolution.

    Tests R2 in isolation, separating "the LLM misread the TOC" from "the
    text it received was incomplete." A PDF `source` is split into the same
    small, fixed-size, non-overlapping page batches the real pipeline uses
    (`_extract_toc_entries_in_batches`) -- so this mirrors `ingest` exactly,
    including how many LLM calls it makes and how a malformed batch is
    skipped rather than failing everything. A .txt `source` (e.g. a dump
    from `dump-toc-text`, possibly hand-edited) is sent as a single call
    instead, since it has no page structure to batch over -- useful for
    iterating on the prompt/text without Ollama ever re-reading the PDF.
    Never touches the database.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config()
    _verify_models([config.llm_model])
    client = ollama_client.RealOllamaClient(num_ctx=config.num_ctx)

    if source.suffix.lower() == ".txt":
        toc_text = source.read_text(encoding="utf-8")
        raw_response = client.generate(config.llm_model, pdf_mod._TOC_SYSTEM_PROMPT, toc_text)
        entries = pdf_mod._parse_llm_toc_response(raw_response)
        if entries is None:
            out.write_text(raw_response, encoding="utf-8")
            typer.secho(f"Response was not valid JSON. Wrote raw text to {out}.", fg=typer.colors.RED, err=True)
            typer.echo("--- raw response ---")
            typer.echo(raw_response)
            raise typer.Exit(code=1)
    else:
        doc = pymupdf.open(str(source))
        page_count = doc.page_count
        raw_pages_text = [page.get_text("text") for page in doc]
        doc.close()
        raw_pages_text = pdf_mod._strip_running_headers_footers(raw_pages_text)
        if pages is not None:
            start, end = _parse_page_range(pages, page_count)
        else:
            region = pdf_mod._detect_toc_pages(raw_pages_text)
            if region is None:
                typer.secho("No TOC region detected.", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            start, end = region
        n_pages = end - start + 1
        n_batches = -(-n_pages // pdf_mod._TOC_LLM_BATCH_PAGES)
        typer.echo(
            f"Using TOC region: pages {start}-{end} "
            f"({n_batches} batch(es) of up to {pdf_mod._TOC_LLM_BATCH_PAGES} pages)"
        )
        entries = pdf_mod._extract_toc_entries_in_batches(
            raw_pages_text, (start, end), client, config.llm_model
        )

    out.write_text(
        json.dumps([{"title": t, "declared_page": p} for t, p in entries], indent=2),
        encoding="utf-8",
    )
    typer.echo(f"Wrote {len(entries)} entries to {out}")
    for title, page in entries:
        typer.echo(f"  declared_page={page:>5}  {title}")


@app.command(name="resolve-toc")
def resolve_toc(
    toc_json: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Raw LLM TOC output, e.g. from extract-toc."
    ),
    pdf_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Diagnostic: run ONLY validation + offset resolution against a structured TOC.

    Tests R3 (offset resolution) and R4 (validation) in isolation from
    extraction. Prints pass/fail and the reason for any rejection. Never
    touches the database, never calls the LLM.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = toc_json.read_text(encoding="utf-8")
    entries = pdf_mod._parse_llm_toc_response(raw)
    if entries is None:
        typer.secho("Could not parse TOC JSON (not valid JSON, or not a list).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Parsed {len(entries)} entries:")
    for title, page in entries:
        typer.echo(f"  declared_page={page:>5}  {title}")

    if not pdf_mod._validate_declared_entries(entries):
        typer.secho("REJECTED at declared-entries validation (see reason above).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    doc = pymupdf.open(str(pdf_path))
    page_count = doc.page_count
    raw_pages_text = [page.get_text("text") for page in doc]
    doc.close()
    raw_pages_text = pdf_mod._strip_running_headers_footers(raw_pages_text)

    region = pdf_mod._detect_toc_pages(raw_pages_text)
    search_start = min(region[1] + 1, page_count) if region else 1
    bounds = pdf_mod._resolve_chapter_pages(entries, raw_pages_text, page_count, search_start)
    if bounds is None:
        typer.secho(
            "REJECTED: could not resolve a physical page for one or more entries.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if not pdf_mod._validate_resolved_bounds(bounds, page_count):
        typer.secho("REJECTED at resolved-bounds validation (see reason above).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nACCEPTED: {len(bounds)} resolved chapter(s):")
    for title, page_start, page_end in bounds:
        typer.echo(f"  pages {page_start:>5}-{page_end:<5}  {title}")


@app.command(name="list")
def list_books() -> None:
    """List every book in the library."""
    config = load_config()
    conn = db.connect(config.db_path)
    try:
        books = db.list_books(conn)
        if not books:
            typer.echo("No books yet -- run `ingest` first.")
            return
        for book in books:
            counts = db.counts_by_status(conn, book.id)
            chapters = db.chapter_count(conn, book.id)
            sections = db.section_count(conn, book.id)
            author = f" by {book.author}" if book.author else ""
            typer.echo(
                f'{book.slug}  "{book.name}"{author}  [{book.source_type}]  '
                f"{book.page_count}pp, {chapters} chapter(s), {sections} section(s)  "
                f"chunks={counts}"
            )
    finally:
        conn.close()


@app.command()
def delete(
    slug: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Permanently remove a book and everything under it. CLI-only -- never exposed to the MCP server."""
    config = load_config()
    conn = db.connect(config.db_path)
    try:
        book = db.get_book_by_slug(conn, slug)
        if book is None:
            typer.secho(f"No book with slug '{slug}'.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        if not yes:
            confirmed = typer.confirm(
                f"Delete '{book.name}' ({slug}) and all its chapters/sections/chunks?"
            )
            if not confirmed:
                typer.echo("Aborted.")
                return
        db.delete_book(conn, book.id)
        typer.echo(f"Deleted '{slug}'.")
    finally:
        conn.close()


@app.command(name="query")
def query_cmd(
    question: str = typer.Argument(...),
    book: Optional[str] = typer.Option(
        None, "--book", help="Scope the search to one book's slug."
    ),
    author: Optional[str] = typer.Option(None, "--author", help="Filter by exact book author."),
    source_type: Optional[str] = typer.Option(
        None, "--type", help="Filter by source type (pdf, epub, markdown, text)."
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", help="Retrieval mode: 'hybrid' (default) or 'vector' (debug override)."
    ),
    expand: bool = typer.Option(False, "--expand", help="Include the full parent section text."),
) -> None:
    """Embed a question and print the top-k cited chunks across the library (or one book)."""
    config = load_config()
    _verify_models([config.embedding_model])
    client = ollama_client.RealOllamaClient(num_ctx=config.num_ctx)
    conn = db.connect(config.db_path)
    try:
        book_id = None
        if book is not None:
            row = db.get_book_by_slug(conn, book)
            if row is None:
                typer.secho(f"No book with slug '{book}'.", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            book_id = row.id
        results = search_mod.search(
            conn,
            client,
            config.embedding_model,
            question,
            book_id=book_id,
            top_k=config.top_k,
            expand=expand,
            author=author,
            source_type=source_type,
            mode=mode or config.retrieval_mode,
            candidate_pool_size=config.candidate_pool_size,
            rrf_k=config.rrf_k,
        )
        typer.echo(search_mod.format_results(results))
    finally:
        conn.close()


@app.command()
def status() -> None:
    """Show the library's books, chapter/section/chunk status, and configured models."""
    config = load_config()
    typer.echo(f"DB path: {config.db_path}")
    typer.echo(f"LLM model: {config.llm_model}")
    typer.echo(f"Embedding model: {config.embedding_model}")

    if not Path(config.db_path).exists():
        typer.echo("No database yet -- run `ingest` first.")
        return

    conn = db.connect(config.db_path)
    try:
        books = db.list_books(conn)
        if not books:
            typer.echo("No books yet -- run `ingest` first.")
            return
        for book in books:
            counts = db.counts_by_status(conn, book.id)
            total = sum(counts.values())
            typer.echo(f'\n{book.slug} -- "{book.name}" ({book.page_count}pp)')
            typer.echo(f"  Chapters: {db.chapter_count(conn, book.id)}")
            typer.echo(f"  Sections: {db.section_count(conn, book.id)}")
            typer.echo(f"  Chunks: {total} total")
            for s in ("pending", "contextualized", "embedded"):
                typer.echo(f"    {s}: {counts.get(s, 0)}")
    finally:
        conn.close()


@app.command(name="reindex-fts")
def reindex_fts(
    book: Optional[str] = typer.Option(
        None, "--book", help="Only rebuild this book's slug (default: the whole library)."
    ),
) -> None:
    """Rebuild the FTS5 keyword index from every embedded chunk. CLI-only."""
    config = load_config()
    conn = db.connect(config.db_path)
    try:
        book_id = None
        if book is not None:
            row = db.get_book_by_slug(conn, book)
            if row is None:
                typer.secho(f"No book with slug '{book}'.", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            book_id = row.id
        n = db.rebuild_fts_index(conn, book_id)
        typer.echo(f"Reindexed {n} chunk(s).")
    finally:
        conn.close()


@app.command(name="serve-mcp")
def serve_mcp() -> None:
    """Start the read-only MCP server over stdio, for use as a Claude tool."""
    from . import mcp_server

    mcp_server.run()


def main() -> None:
    app()
