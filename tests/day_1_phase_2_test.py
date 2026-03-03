"""Unit tests for the OkavangoData data processing pipeline."""

import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

# Add the project root to the system path to locate main.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import OkavangoData  # pylint: disable=import-error, wrong-import-position

def test_download_function():
    """Test Function 1: Downloads a single small file using the class method."""
    # Create a 'blank' instance of the class without running __init__
    dummy_app = OkavangoData.__new__(OkavangoData)

    test_urls = {
        "test_forest.csv": 
        "https://ourworldindata.org/grapher/annual-change-forest-area.csv?v=1&csvType=full"
        }

    # Call the method on our dummy app
    paths = dummy_app.download_project_datasets(test_urls)

    assert "test_forest.csv" in paths
    assert paths["test_forest.csv"].exists()

def test_merge_function():
    """Test Function 2: Merges dummy data with a GeoDataFrame using the class method."""
    dummy_app = OkavangoData.__new__(OkavangoData)

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

    merged = dummy_app.merge_map_with_datasets(world, data)

    assert 'val' in merged.columns
    assert not merged['val'].isna().all()
