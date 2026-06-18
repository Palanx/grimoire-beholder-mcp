"""Shared Ollama access: model-availability checks and an injectable client.

`ensure_models_available` never pulls a model on its own -- it fails
loudly with the exact `ollama pull` command(s) needed, so the user stays
in control of what gets downloaded.

Every generation/embedding call goes through the `OllamaClient` protocol,
with `RealOllamaClient` as the sole production implementation. Routing all
access through this one seam means the pipeline (contextualize, embed,
search) never imports `ollama` directly -- tests inject a fake client and
run the whole pipeline with no daemon running and no models pulled.

`RealOllamaClient` wraps calls with tenacity retries and runs cogito in
STANDARD mode: the system prompt never contains the "Enable deep thinking
subroutine." phrase that switches cogito into extended reasoning. As a
defensive measure (in case a model still emits one anyway), any
<think>...</think> block is stripped before the text is returned, so
reasoning traces never leak into stored context or summaries.
"""

from __future__ import annotations

import re
from typing import Protocol

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RETRY = retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama daemon can't be reached at all."""


class MissingModelsError(RuntimeError):
    """Raised when one or more required models are not pulled."""


def ensure_models_available(required: list[str]) -> None:
    """Fail loudly with the exact `ollama pull` command for any missing model."""
    try:
        listing = ollama.list()
    except Exception as exc:  # connection refused, daemon not running, etc.
        raise OllamaUnavailableError(
            f"Could not reach Ollama at http://localhost:11434 ({exc}). "
            "Start it first, e.g. `ollama serve` or open the Ollama app."
        ) from exc

    installed = {m.model for m in listing.models}
    missing = [m for m in required if not _model_present(installed, m)]
    if missing:
        lines = "\n".join(f"  ollama pull {m}" for m in missing)
        raise MissingModelsError(f"Missing required Ollama model(s). Run:\n{lines}")


def _model_present(installed: set[str], wanted: str) -> bool:
    if wanted in installed:
        return True
    if ":" not in wanted:
        return f"{wanted}:latest" in installed
    return False


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class OllamaClient(Protocol):
    """The pipeline's only window onto a model -- generate and embed.

    Anything implementing these two methods can stand in for Ollama. In
    particular, tests inject a fake implementation so the rest of the
    pipeline (contextualize, embed, search) runs deterministically with no
    Ollama daemon and no models pulled.
    """

    def generate(self, model: str, system: str, prompt: str) -> str:
        """Run one generation and return the model's response text."""
        ...

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        """Embed a batch of strings in one request."""
        ...


class RealOllamaClient:
    """Talks to a local Ollama daemon, with retries and <think> stripping."""

    @_RETRY
    def generate(self, model: str, system: str, prompt: str) -> str:
        """Run one standard-mode generation, with <think> blocks stripped defensively."""
        response = ollama.generate(model=model, system=system, prompt=prompt, think=False)
        return _strip_think(response.response)

    @_RETRY
    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        """Embed a batch of strings in one request."""
        response = ollama.embed(model=model, input=inputs)
        return response.embeddings
