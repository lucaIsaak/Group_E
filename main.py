#  source .venv/bin/activate
import requests
from pathlib import Path
from typing import Dict
from pydantic import HttpUrl, validate_call, ConfigDict
import pandas as pd
import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "downloads"

class OkavangoData:
    """Main class to handle data processing for Project Okavango."""

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, dataset_urls: Dict[str, HttpUrl]):
        self.datasets: Dict[str, pd.DataFrame] = {}
        
        # Execute Phase 2: Function 1 (Now called with self.)
        self.downloaded_paths: Dict[str, Path] = self.download_project_datasets(dataset_urls)
        
        # Read datasets into attributes as required
        for name, path in self.downloaded_paths.items():
            if name.endswith(".csv"):
                df = pd.read_csv(path)
                self.datasets[name] = df
                
                # Attribute creation (e.g., self.annual_deforestation_csv)
                attr_name = name.replace(".csv", "_df").replace("-", "_")
                setattr(self, attr_name, df)
        
        map_path = self.downloaded_paths.get("ne_110m_admin_0_countries.zip")
        if not map_path:
            raise FileNotFoundError("World map dataset missing.")
            
        self.world_map: gpd.GeoDataFrame = gpd.read_file(map_path)

        # Execute Phase 2: Function 2 (Now called with self.)
        self.merged_gdf: gpd.GeoDataFrame = self.merge_map_with_datasets(
            self.world_map, 
            self.datasets
        )

    # --- FUNCTION 1 INTEGRATED AS A METHOD ---
    @validate_call
    def download_project_datasets(self, datasets: Dict[str, HttpUrl]) -> Dict[str, Path]:
        """Downloads required datasets into the downloads directory."""
        DOWNLOAD_DIR.mkdir(exist_ok=True)
        saved_files: Dict[str, Path] = {}
        headers = {'User-Agent': 'Okavango Hackathon Data Fetcher/1.0'}
        
        for file_name, url in datasets.items():
            save_path = DOWNLOAD_DIR / file_name
            try:
                response = requests.get(str(url), headers=headers, timeout=30)
                response.raise_for_status() 
                with open(save_path, "wb") as f:
                    f.write(response.content)
                saved_files[file_name] = save_path
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_name}: {e}")
                
        return saved_files

    # --- FUNCTION 2 INTEGRATED AS A METHOD ---
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def merge_map_with_datasets(
        self,
        world_map: gpd.GeoDataFrame, 
        datasets: Dict[str, pd.DataFrame]
    ) -> gpd.GeoDataFrame:
        """Merges map with datasets using the most recent data available."""
        merged_map = world_map.copy()
        
        for name, df in datasets.items():
            df.columns = [c.lower() for c in df.columns]
            
            if 'code' in df.columns and 'year' in df.columns:
                idx_latest = df.groupby('code')['year'].idxmax()
                latest_data = df.loc[idx_latest]
                
                merged_map = merged_map.merge(
                    latest_data, 
                    how='left', 
                    left_on='ADM0_A3', 
                    right_on='code',
                    suffixes=('', f'_{name}') 
                )
                
        return merged_map

    def get_data(self) -> gpd.GeoDataFrame:
        """Returns the processed GeoDataFrame."""
        return self.merged_gdf