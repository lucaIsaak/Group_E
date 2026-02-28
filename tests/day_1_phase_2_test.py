# tests/test_main.py
import sys
from pathlib import Path

# Add the project root (one level up from /tests) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import download_project_datasets, merge_map_with_datasets, OkavangoData
import pytest
import pandas as pd
import geopandas as gpd

def test_download_function():
    """Test Function 1: Downloads a single small file."""
    test_urls = {"test.csv": "https://ourworldindata.org/grapher/annual-change-forest-area.csv?v=1&csvType=full"}
    paths = download_project_datasets(test_urls)
    assert "test.csv" in paths
    assert paths["test.csv"].exists()

def test_merge_function():
    """Test Function 2: Merges dummy data with a GeoDataFrame."""
    world = gpd.GeoDataFrame({
        'ADM0_A3': ['USA', 'BRA'],
        'geometry': [None, None]
    })
    data = {
        'test': pd.DataFrame({
            'code': ['USA', 'BRA'],
            'year': [2020, 2020],
            'val': [1, 2]
        })
    }
    merged = merge_map_with_datasets(world, data)
    assert 'val' in merged.columns
    assert not merged['val'].isna().all()