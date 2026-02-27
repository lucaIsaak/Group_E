import requests
from pathlib import Path
from typing import Dict
from pydantic import HttpUrl, validate_call

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