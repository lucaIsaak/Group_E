import requests
from pathlib import Path
from typing import Dict
from pydantic import HttpUrl, validate_call, ConfigDict
import pandas as pd
import geopandas as gpd

# PEP 8 constant for the download directory
DOWNLOAD_DIR = Path("downloads_")

@validate_call
def download_project_datasets(datasets: Dict[str, HttpUrl]) -> Dict[str, Path]:
    """
    Downloads all required datasets into the downloads_ directory.
    
    Args:
        datasets: A dictionary mapping the desired output file name 
                  (e.g., 'forest_area.csv') to its URL.
                  
    Returns:
        A dictionary mapping the file name to its saved local Path.
    """
    # Ensure the directory exists
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    saved_files: Dict[str, Path] = {}
    
    # Headers to prevent Our World in Data from blocking the request
    headers = {'User-Agent': 'Okavango Hackathon Data Fetcher/1.0'}
    
    for file_name, url in datasets.items():
        save_path = DOWNLOAD_DIR / file_name
        
        try:
            # Pydantic HttpUrl needs to be cast to a string for requests
            response = requests.get(str(url), headers=headers, timeout=30)
            response.raise_for_status() 
            
            # Save the file to the downloads_ directory
            with open(save_path, "wb") as f:
                f.write(response.content)
                
            saved_files[file_name] = save_path
            print(f"Success: Downloaded {file_name}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to download {file_name}. Reason: {e}")
            
    return saved_files



# Pydantic needs special permission to validate complex objects like DataFrames
@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def merge_map_with_datasets(
    world_map: gpd.GeoDataFrame, 
    datasets: Dict[str, pd.DataFrame]
) -> gpd.GeoDataFrame:
    """
    Merges a GeoPandas map dataframe with multiple pandas dataframes containing OWID data.
    Ensures the map is the left dataframe and filters OWID data for the most recent year.
    
    Args:
        world_map: The GeoDataFrame containing country polygons.
        datasets: A dictionary mapping dataset names to their pandas DataFrames.
        
    Returns:
        A single merged GeoDataFrame.
    """
    # Create a copy so we don't accidentally modify the original map in place
    merged_map = world_map.copy()
    
    for name, df in datasets.items():
        # CHANGE 1: Get the MOST RECENT data
        # OWID datasets are time-series. We only want the latest year per country.
        if 'Year' in df.columns and 'Code' in df.columns:
            # Find the index of the maximum year for each country code
            idx_latest = df.groupby('Code')['Year'].idxmax()
            latest_data = df.loc[idx_latest]
        else:
            latest_data = df
            
        # CHANGE 2: Merge on ISO Codes, NOT Country Names
        # The map is on the left (merged_map.merge)
        merged_map = merged_map.merge(
            latest_data, 
            how='left', 
            left_on='ADM0_A3',  # Natural Earth's 3-letter country code column
            right_on='Code',    # Our World in Data's 3-letter country code column
            suffixes=('', f'_{name}') # Prevents errors if datasets have overlapping column names
        )
        
    return merged_map