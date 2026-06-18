"""Typer CLI: ingest, list, delete, query, status, reindex-fts, serve-mcp."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import db, ingest as ingest_mod, ollama_client, search as search_mod
from .config import load_config

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
    client = ollama_client.RealOllamaClient()
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
    client = ollama_client.RealOllamaClient()
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
