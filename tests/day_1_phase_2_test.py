import sys
from pathlib import Path

# Add the project root (one level up from /tests) to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import geopandas as gpd

# Local imports
from main import download_project_datasets, merge_map_with_datasets

def test_download_function():
    """
    Test Function 1: Verifies that the download function successfully 
    fetches a file from a valid URL and saves it to the local directory.
    """
    test_filename = "test_forest_area.csv"
    test_urls = {
        test_filename: "https://ourworldindata.org/grapher/annual-change-forest-area.csv?v=1&csvType=full"
    }
    
    # Execute download
    paths = download_project_datasets(test_urls)
    
    # Assertions
    assert test_filename in paths, "Filename missing from returned paths dictionary."
    assert paths[test_filename].exists(), "Downloaded file does not exist on disk."
    
    # Cleanup after test to maintain a clean environment
    paths[test_filename].unlink(missing_ok=True)


def test_merge_function():
    """
    Test Function 2: Verifies that dummy tabular data merges correctly 
    with a dummy GeoDataFrame based on country codes and the latest year.
    """
    # Create dummy geographic data
    world = gpd.GeoDataFrame({
        'ADM0_A3': ['USA', 'BRA'],
        'geometry': [None, None]
    })
    
    # Create dummy tabular data representing the downloaded CSVs
    # Note: USA has data for 2019 and 2020 to test the 'latest year' logic
    data = {
        'test_metric': pd.DataFrame({
            'code': ['USA', 'BRA', 'USA'],
            'year': [2019, 2020, 2020], 
            'val': [1, 2, 3]
        })
    }
    
    # Execute merge
    merged = merge_map_with_datasets(world, data)
    
    # Assertions
    assert 'val' in merged.columns, "Merged column 'val' is missing from the output."
    assert not merged['val'].isna().all(), "All merged values are NaN, indicating a merge failure."
    
    # Verify that the logic successfully picked the most recent year (2020) for USA, which has val=3
    usa_val = merged.loc[merged['ADM0_A3'] == 'USA', 'val'].values[0]
    assert usa_val == 3, f"Expected 3 (most recent year), but got {usa_val}"