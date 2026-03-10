"""
apps/ollama_analysis.py

Satellite image analysis using a local Ollama vision model.

The model is pulled automatically if it is not already present on the user's
machine.  Analysis is streamed token-by-token so the UI can show a live
progress indicator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VISION_MODEL = "llava"

_PROMPT = (
    "You are analysing a satellite image. "
    "Describe in detail what you see: terrain type, vegetation, water bodies, "
    "urban structures, and any other notable features. "
    "Be specific and concise."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_is_available(model: str) -> bool:
    """Return True if *model* is already present in the local Ollama library."""
    try:
        local_models = ollama.list()
        return any(m.model.startswith(model) for m in local_models.models)
    except Exception:
        return False


def ensure_model(model: str = VISION_MODEL) -> None:
    """Pull *model* from the Ollama registry if it is not already local.

    Raises:
        ollama.ResponseError: If the pull request fails (e.g. model not found).
    """
    if _model_is_available(model):
        return
    for _ in ollama.pull(model, stream=True):
        pass  # exhaust the generator; progress is shown by the caller


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe_satellite_image(
    image_path: Path,
    model: str = VISION_MODEL,
) -> Iterator[str]:
    """Stream a natural-language description of a satellite image.

    Ensures the requested vision model is available locally, then streams the
    response token-by-token.

    Args:
        image_path: Path to the PNG/JPEG satellite image on disk.
        model:      Ollama model tag to use (must support vision).

    Yields:
        Text chunks as they arrive from the model.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        ollama.ResponseError: If the model or Ollama server is unavailable.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ensure_model(model)

    stream = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": _PROMPT,
                "images": [str(image_path)],
            }
        ],
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token
