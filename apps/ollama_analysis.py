"""
apps/ollama_analysis.py

Satellite image analysis using a local Ollama vision model.

The model is pulled automatically if it is not already present on the user's
machine.  Analysis is streamed token-by-token so the UI can show a live
progress indicator.

All AI settings (models, prompts, temperatures) are loaded from
``models.yaml`` in the project root directory.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator

import ollama  # pylint: disable=import-error
import yaml

# ---------------------------------------------------------------------------
# Configuration — loaded from models.yaml
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "models.yaml"


def _load_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_cfg = _load_config()

VISION_MODEL: str = _cfg["vision"]["model"]
TEXT_MODEL: str = _cfg["text"]["model"]

_PROMPT: str = _cfg["vision"]["prompt"]
_RISK_PROMPT_TEMPLATE: str = _cfg["text"]["prompt_template"]

_VISION_TEMPERATURE: float = float(_cfg["vision"].get("temperature", 0.2))
_TEXT_TEMPERATURE: float = float(_cfg["text"].get("temperature", 0.3))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_is_available(model: str) -> bool:
    """Return True if *model* is already present in the local Ollama library."""
    try:
        local_models = ollama.list()
        return any(m.model.startswith(model) for m in local_models.models)
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def ensure_model(model: str = VISION_MODEL) -> None:
    """Pull *model* from the Ollama registry if it is not already local.

    Raises:
        ollama.ResponseError: If the pull request fails (e.g. model not found).
    """
    if _model_is_available(model):
        print(f"[ollama] Model '{model}' already available locally.")
        return
    print(f"[ollama] Model '{model}' not found locally — pulling now…")
    for _ in ollama.pull(model, stream=True):
        pass
    print(f"[ollama] Model '{model}' pull complete.")


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

    print(f"[step 1/2] Checking vision model '{model}'…")
    ensure_model(model)

    print(f"[step 2/2] Sending image to '{model}' for description…")
    t0 = time.time()
    token_count = 0

    stream = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": _PROMPT,
                "images": [str(image_path)],
            }
        ],
        options={"temperature": _VISION_TEMPERATURE},
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            token_count += 1
            if token_count == 1:
                print("[step 2/2] First token received — model is responding.")
            yield token

    print(f"[step 2/2] Description complete ({token_count} tokens, {time.time()-t0:.1f}s).")


def assess_environmental_risk(
    description: str,
    model: str = TEXT_MODEL,
) -> Iterator[str]:
    """Stream an environmental risk assessment derived from an image description.

    The model autonomously generates the most relevant environmental risk
    questions for the described area and answers them, then provides an overall
    verdict.  Ensures the text model is available locally before calling it.

    Args:
        description: Natural-language description of the satellite image.
        model:       Ollama text model tag to use.

    Yields:
        Text chunks as they arrive from the model.

    Raises:
        ollama.ResponseError: If the model or Ollama server is unavailable.
    """
    print(f"[step 1/2] Checking text model '{model}'…")
    ensure_model(model)

    print(f"[step 2/2] Sending description to '{model}' for risk assessment…")
    t0 = time.time()
    token_count = 0

    prompt = _RISK_PROMPT_TEMPLATE.format(description=description)

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": _TEXT_TEMPERATURE},
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            token_count += 1
            if token_count == 1:
                print("[step 2/2] First token received — model is responding.")
            yield token

    print(f"[step 2/2] Risk assessment complete ({token_count} tokens, {time.time()-t0:.1f}s).")


def extract_risk_verdict(assessment: str) -> str:
    """Parse the OVERALL VERDICT line from the risk assessment text.

    Returns one of: ``'AT RISK'``, ``'NOT AT RISK'``, or ``'UNCERTAIN'``.
    """
    for line in assessment.splitlines():
        upper = line.upper()
        if "OVERALL VERDICT" in upper or "OVERALL:" in upper:
            if "NOT AT RISK" in upper:
                return "NOT AT RISK"
            if "AT RISK" in upper:
                return "AT RISK"
            if "UNCERTAIN" in upper:
                return "UNCERTAIN"
    return "UNCERTAIN"
