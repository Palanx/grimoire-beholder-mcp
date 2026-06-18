"""Documentation stub for the grimoire-beholder-mcp MCP server entry point.

This file is not executed. The bundle's manifest.json launches the server
via `mcp_config` by shelling out to `uv run --project <project_dir>
--directory <library_dir> grimoire-beholder serve-mcp` in the user's own,
already-`uv sync`'d clone of the grimoire-beholder-mcp repository -- see ../../README.md.
grimoire-beholder-mcp's runtime dependencies (mcp, numpy, pymupdf, ebooklib, lxml,
ollama, tenacity, typer) include compiled extensions that cannot be
portably vendored into a single cross-platform .mcpb, so this bundle
intentionally ships no code and relies entirely on the user's prepared
environment.
"""
