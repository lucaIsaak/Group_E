"""
apps/satellite.py

Satellite tile imagery fetching from ESRI World Imagery (free, no API key).

Tiles follow the standard Slippy Map (XYZ) convention:
    https://server.arcgisonline.com/.../tile/{z}/{y}/{x}
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

TILE_SIZE_PX = 256
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"

_HEADERS = {"User-Agent": "Mozilla/5.0 (Project Okavango / educational use)"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convert geographic coordinates to Slippy Map tile indices.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        zoom: Zoom level (0–18).

    Returns:
        Tuple (tile_x, tile_y).
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return tile_x, tile_y


def _fetch_tile(tile_x: int, tile_y: int, zoom: int) -> Image.Image:
    """
    Download a single 256 × 256 PNG tile from ESRI World Imagery.

    Args:
        tile_x: Tile column index.
        tile_y: Tile row index.
        zoom: Zoom level.

    Returns:
        PIL Image of the tile.

    Raises:
        requests.HTTPError: If the server returns an error status.
    """
    url = ESRI_TILE_URL.format(z=zoom, y=tile_y, x=tile_x)
    response = requests.get(url, headers=_HEADERS, timeout=15)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_satellite_image(
    lat: float,
    lon: float,
    zoom: int,
    size_px: int,
) -> Path:
    """
    Fetch a satellite image centred on the given coordinates.

    Downloads tiles from ESRI World Imagery, stitches them into a single
    image, and saves the result as a PNG in the images directory.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        zoom: Zoom level (1–18).
        size_px: Output image size in pixels. Must be a positive multiple
                 of 256 (e.g. 256, 512, 1024).

    Returns:
        Path to the saved PNG file.

    Raises:
        ValueError: If size_px is not a positive multiple of 256.
        requests.HTTPError: If a tile download fails.
    """
    if size_px <= 0 or size_px % TILE_SIZE_PX != 0:
        raise ValueError(
            f"size_px must be a positive multiple of {TILE_SIZE_PX}, got {size_px}."
        )

    IMAGES_DIR.mkdir(exist_ok=True)

    output_path = IMAGES_DIR / f"sat_{lat:.4f}_{lon:.4f}_z{zoom}_{size_px}px.png"
    if output_path.exists():
        return output_path

    tiles_per_side = size_px // TILE_SIZE_PX
    offset = tiles_per_side // 2
    center_x, center_y = _lat_lon_to_tile(lat, lon, zoom)

    canvas = Image.new("RGB", (size_px, size_px))
    for row in range(tiles_per_side):
        for col in range(tiles_per_side):
            tile_x = center_x - offset + col
            tile_y = center_y - offset + row
            tile = _fetch_tile(tile_x, tile_y, zoom)
            canvas.paste(tile, (col * TILE_SIZE_PX, row * TILE_SIZE_PX))

    canvas.save(output_path)
    return output_path
