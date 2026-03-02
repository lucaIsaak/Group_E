# Project Okavango - Group E

Welcome to the Okavango Hackathon project repository for Group E! 

This project provides a lightweight, interactive data analysis tool for environmental protection. It automatically fetches the most recent global environmental datasets, merges them with geographical data, and presents them in an interactive dashboard.

## Group Members
* Ruben Vetter: 70844@novasbe.pt
* Luca Isaak: 70197@novasbe.pt
* Lennart Stenzel: 70485@novasbe.pt

---

## 1. How to start using your code 
**(Installation & Usage)**

To run this project, you will need Python installed on your machine along with a few key data science libraries.

**Step 1: Clone the repository**
```bash
git clone [https://github.com/lucaIsaak/Group_E.git](https://github.com/lucaIsaak/Group_E.git)
cd Group_E
```

**Step 2: Install the required dependencies**
*(Assuming you have a virtual environment activated)*
```bash
pip install streamlit pandas geopandas pydantic requests pytest matplotlib
```

**Step 3: Run the application**
We use Streamlit for our front-end application. To launch the dashboard, run the following command from the root of the project:
```bash
streamlit run apps/main_app.py
```
*Example Usage:* Once the browser window opens, use the dropdown menu at the top to select a dataset (e.g., "Annual change in forest area"). The app will automatically download the required data, extract the most recent year for each country, and display the choropleth map alongside the Top 5 and Bottom 5 countries.

---

## 2. What our modules and functions are doing
**(Architecture & Logic)**

We chose to separate our data processing logic from our UI logic to keep the codebase clean, modular, and easy to debug. All data calls are validated using `pydantic` strict typing.

### `main.py` (Data Engine)
This module handles all the heavy lifting for data ingestion and manipulation.
* `download_project_datasets(datasets)`: Downloads the required CSVs from *Our World in Data* and the shapefile from *Natural Earth* into a local `/downloads` directory.
* `merge_map_with_datasets(world_map, datasets)`: Merges the tabular data with the spatial GeoDataFrame. Crucially, it uses Pandas `idxmax()` logic to dynamically find and filter for the **most recent data available** per country, ensuring no years are hardcoded.
* `OkavangoData` (Class): The central data handler. Upon initialization, it triggers the downloads, reads the CSVs into dynamic attributes, loads the world map, and executes the merge.

### `apps/main_app.py` (Streamlit Dashboard)
This module contains the front-end code. 
* It initializes `OkavangoData` using `@st.cache_resource` so the data isn't re-downloaded every time the user clicks a button. 
* It maps the selected dropdown options to the correct underlying DataFrame columns and uses `matplotlib` to render the maps and the Top/Bottom 5 bar charts.

---

## 3. The expected results and how you test your code
**(Testing & Workflow)**

### Expected Results
When running the application, you should expect a web dashboard that successfully displays a global map colored by the selected metric (e.g., Share of land that is protected). Below the map, you will see two horizontal bar charts highlighting the 5 countries with the highest values (green) and the 5 with the lowest values (red). The data displayed will always represent the latest available year for each specific country.

### How to Test
We have included a test suite using `pytest` to ensure the core data logic is robust and to help others debug the workflow. 

To run the tests, simply execute the following command in the root directory:
```bash
pytest tests/
```

**What the tests cover (`tests/test_main.py`):**
1.  **Network/Download Logic:** Verifies that `download_project_datasets` can successfully reach out to the internet, download a CSV file, and save it to the local disk.
2.  **Merge & Temporal Logic:** Creates dummy spatial and tabular data (with multiple years of data for a single country) to verify that `merge_map_with_datasets` successfully joins the dataframes *and* correctly isolates the most recent year.

---
*License: MIT License*