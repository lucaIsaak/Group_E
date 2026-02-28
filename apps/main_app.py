import sys
from pathlib import Path

# Add the project root to the sys.path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from main import OkavangoData # Now this will work
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Datasets as defined in project instructions
DATASET_CONFIG = {
    "annual_change_forest_area.csv": "https://ourworldindata.org/grapher/annual-change-forest-area.csv?v=1&csvType=full&useColumnShortNames=true",
    "annual_deforestation.csv": "https://ourworldindata.org/grapher/annual-deforestation.csv?v=1&csvType=full&useColumnShortNames=true",
    "protected_land.csv": "https://ourworldindata.org/grapher/terrestrial-protected-areas.csv?v=1&csvType=full&useColumnShortNames=true",
    "degraded_land.csv": "https://ourworldindata.org/grapher/share-degraded-land.csv?v=1&csvType=full&useColumnShortNames=true",
    "red_list_index.csv": "https://ourworldindata.org/grapher/red-list-index.csv?v=1&csvType=full&useColumnShortNames=true",
    "ne_110m_admin_0_countries.zip": "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
}

@st.cache_resource
def get_processed_data():
    """Initialize the OkavangoData class once and cache the result."""
    return OkavangoData(DATASET_CONFIG)
# Load the data using the Class from main.py
okavango = get_processed_data()
gdf = okavango.get_data()

st.title("🌍 Project Okavango: Environmental Data Tool")

# 1. Map Selection Logic
# Extracting the numerical columns added during the merge
display_columns = [col for col in gdf.columns if col not in gdf.geometry.name and col not in ['admin', 'code', 'year']]
selected_metric = st.selectbox("Select a map to plot:", display_columns)

# 2. Plotting the World Map
st.subheader(f"Global Distribution: {selected_metric}")
fig_map, ax_map = plt.subplots(1, 1, figsize=(15, 8))
gdf.plot(
    column=selected_metric, 
    ax=ax_map, 
    legend=True, 
    cmap='viridis', 
    missing_kwds={'color': 'lightgrey'}
)
ax_map.set_axis_off()
st.pyplot(fig_map)

# 3. Top 5 / Bottom 5 Analysis
st.divider()
st.subheader(f"Top 5 and Bottom 5 Countries: {selected_metric}")

# Prepare the data for charting
chart_data = gdf[['admin', selected_metric]].dropna().sort_values(by=selected_metric, ascending=False)

if not chart_data.empty:
    top_5 = chart_data.head(5)
    bottom_5 = chart_data.tail(5)
    combined_data = pd.concat([top_5, bottom_5])

    fig_chart, ax_chart = plt.subplots(figsize=(10, 6))
    
    # Visual distinction between top and bottom
    colors = ['#2ecc71'] * 5 + ['#e74c3c'] * 5 
    
    ax_chart.barh(combined_data['admin'], combined_data[selected_metric], color=colors)
    ax_chart.set_xlabel("Value")
    ax_chart.invert_yaxis()  # Best values on top
    
    st.pyplot(fig_chart)
else:
    st.warning("No data available to generate charts for this selection.")