"""
apps/db.py

Persistence layer for the AI classification pipeline.

Each completed run (satellite image → description → risk assessment) is
appended as a new row to ``database/images.csv`` in the project root.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "database" / "images.csv"

_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "zoom",
    "image_size_px",
    "image_path",
    "image_model",
    "image_prompt",
    "image_description",
    "text_model",
    "text_prompt",
    "text_description",
    "danger",
]


def find_existing_run(
    latitude: float,
    longitude: float,
    zoom: int,
    image_size_px: int,
) -> Optional[Dict[str, str]]:
    """Return the most recent cached row matching the given image settings, or None.

    Matches on latitude, longitude, zoom, and image_size_px.  If a match is
    found the stored image file must also still exist on disk; otherwise the
    row is skipped.

    Args:
        latitude:     Centre latitude.
        longitude:    Centre longitude.
        zoom:         Zoom level.
        image_size_px: Pixel size of the image.

    Returns:
        The matching row as a dict, or ``None`` if no valid cache entry exists.
    """
    if not DB_PATH.exists():
        return None

    match = None
    with open(DB_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                if (
                    float(row["latitude"]) == latitude
                    and float(row["longitude"]) == longitude
                    and int(row["zoom"]) == zoom
                    and int(row["image_size_px"]) == image_size_px
                    and Path(row["image_path"]).exists()
                ):
                    match = row  # keep iterating — last match wins (most recent)
            except (KeyError, ValueError):
                continue

    return match


def log_run(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    latitude: float,
    longitude: float,
    zoom: int,
    image_size_px: int,
    image_path: str,
    image_model: str,
    image_prompt: str,
    image_description: str,
    text_model: str,
    text_prompt: str,
    text_description: str,
    danger: str,
) -> None:
    """Append one row to the images.csv database.

    Creates the file with headers if it does not exist yet.

    Args:
        latitude:          Centre latitude used to fetch the satellite image.
        longitude:         Centre longitude used to fetch the satellite image.
        zoom:              Zoom level used when fetching the image.
        image_size_px:     Pixel size of the fetched image.
        image_path:        Absolute path to the saved satellite image file.
        image_model:       Ollama model tag used for image description.
        image_prompt:      Prompt sent to the vision model.
        image_description: Full text description produced by the vision model.
        text_model:        Ollama model tag used for risk assessment.
        text_prompt:       Prompt sent to the text model (with description filled in).
        text_description:  Full risk assessment text produced by the text model.
        danger:            Parsed verdict: ``'AT RISK'``, ``'NOT AT RISK'``, or ``'UNCERTAIN'``.
    """
    DB_PATH.parent.mkdir(exist_ok=True)

    write_header = not DB_PATH.exists() or DB_PATH.stat().st_size == 0

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latitude": latitude,
        "longitude": longitude,
        "zoom": zoom,
        "image_size_px": image_size_px,
        "image_path": image_path,
        "image_model": image_model,
        "image_prompt": image_prompt,
        "image_description": image_description,
        "text_model": text_model,
        "text_prompt": text_prompt,
        "text_description": text_description,
        "danger": danger,
    }

    with open(DB_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
