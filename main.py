import requests
from pathlib import Path
from typing import Dict
from pydantic import HttpUrl, validate_call, ConfigDict
import pandas as pd
import geopandas as gpd

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"

@validate_call
def download_project_datasets(datasets: Dict[str, HttpUrl]) -> Dict[str, Path]:
    """
    Downloads required datasets into the designated downloads directory.

    Args:
        datasets (Dict[str, HttpUrl]): A dictionary where keys are the desired 
            filenames and values are the Pydantic-validated URLs to download.

    Returns:
        Dict[str, Path]: A dictionary mapping the filenames to their saved 
            local file paths.
    """
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    saved_files: Dict[str, Path] = {}
    headers = {'User-Agent': 'Okavango Hackathon Data Fetcher/1.0'}
    
    for file_name, url in datasets.items():
        save_path = DOWNLOAD_DIR / file_name
        try:
            # str(url) is required because Pydantic HttpUrl is an object, not a plain string
            response = requests.get(str(url), headers=headers, timeout=30)
            response.raise_for_status() 
            with open(save_path, "wb") as f:
                f.write(response.content)
            saved_files[file_name] = save_path
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_name}: {e}")
            
    return saved_files

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def merge_map_with_datasets(
    world_map: gpd.GeoDataFrame, 
    datasets: Dict[str, pd.DataFrame]
) -> gpd.GeoDataFrame:
    """
    Merges geographic map data with external datasets, ensuring only the 
    most recent data available per country is used.

    Args:
        world_map (gpd.GeoDataFrame): The base map containing country geometries.
        datasets (Dict[str, pd.DataFrame]): A dictionary of datasets to merge.

    Returns:
        gpd.GeoDataFrame: The merged GeoDataFrame containing geometries and 
            the latest available metrics for each country.
    """
    merged_map = world_map.copy()
    
    for name, df in datasets.items():
        # Work on a copy to avoid Pandas SettingWithCopy warnings
        clean_df = df.copy()
        clean_df.columns = [str(c).lower() for c in clean_df.columns]
        
        # Ensure 'code' and 'year' exist to find the most recent data
        if 'code' in clean_df.columns and 'year' in clean_df.columns:
            # Find the index of the maximum year for each country code
            idx_latest = clean_df.groupby('code')['year'].idxmax()
            latest_data = clean_df.loc[idx_latest]
            
            # ADM0_A3 is the standard 3-letter country code in Natural Earth datasets
            merged_map = merged_map.merge(
                latest_data, 
                how='left', 
                left_on='ADM0_A3', 
                right_on='code',
                suffixes=('', f'_{name}') 
            )
            
    return merged_map

class OkavangoData:
    """
    Main class to handle data downloading, processing, and merging for Project Okavango.
    
    Attributes:
        downloaded_paths (Dict[str, Path]): Local paths of downloaded files.
        datasets (Dict[str, pd.DataFrame]): Loaded tabular datasets.
        world_map (gpd.GeoDataFrame): The base geographic map.
        merged_gdf (gpd.GeoDataFrame): The final merged dataset ready for plotting.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, dataset_urls: Dict[str, HttpUrl]):
        """
        Initializes the OkavangoData class by downloading, loading, and merging data.

        Args:
            dataset_urls (Dict[str, HttpUrl]): URLs of the datasets to process.
        """
        # Phase 2: Execute Function 1 (Download)
        self.downloaded_paths: Dict[str, Path] = download_project_datasets(dataset_urls)
        
        self.datasets: Dict[str, pd.DataFrame] = {}
        
        # Phase 2: Read datasets into corresponding dataframes and set as attributes
        for name, path in self.downloaded_paths.items():
            if name.endswith(".csv"):
                df = pd.read_csv(path)
                self.datasets[name] = df
                
                # Dynamically create attributes (e.g., self.forest_area_df)
                attr_name = name.replace(".csv", "_df").replace("-", "_")
                setattr(self, attr_name, df)
        
        # Load the geographic map
        map_path = self.downloaded_paths.get("ne_110m_admin_0_countries.zip")
        if not map_path:
            raise FileNotFoundError("World map dataset missing from downloads.")
            
        self.world_map: gpd.GeoDataFrame = gpd.read_file(map_path)

        # Phase 2: Execute Function 2 (Merge)
        self.merged_gdf: gpd.GeoDataFrame = merge_map_with_datasets(
            self.world_map, 
            self.datasets
        )

    def get_data(self) -> gpd.GeoDataFrame:
        """
        Retrieves the fully processed and merged geographic dataframe.

        Returns:
            gpd.GeoDataFrame: The merged dataset for Streamlit visualization.
        """
        return self.merged_gdf