"""
apps/main_app.py

Streamlit application for Project Okavango. Displays interactive maps and 
statistical charts based on environmental data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

# Add project root to path so we can import from main.py
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from main import OkavangoData  # noqa: E402


def build_dataset_config() -> Dict[str, str]:
    """
    Builds the configuration dictionary containing URLs for the datasets.

    Returns:
        Dict[str, str]: A dictionary mapping filenames to their download URLs.
    """
    base_params = "csvType=full&useColumnShortNames=true"

    return {
        "annual_change_forest_area.csv": (
            f"https://ourworldindata.org/grapher/annual-change-forest-area.csv?{base_params}"
        ),
        "annual_deforestation.csv": (
            f"https://ourworldindata.org/grapher/annual-deforestation.csv?{base_params}"
        ),
        "protected_land.csv": (
            f"https://ourworldindata.org/grapher/terrestrial-protected-areas.csv?{base_params}"
        ),
        "degraded_land.csv": (
            f"https://ourworldindata.org/grapher/share-degraded-land.csv?{base_params}"
        ),
        "red_list_index.csv": (
            f"https://ourworldindata.org/grapher/red-list-index.csv?{base_params}"
        ),
        "ne_110m_admin_0_countries.zip": (
            "https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip"
        ),
    }


@st.cache_resource
def get_processed_data(_dataset_config: Dict[str, str]) -> OkavangoData:
    """
    Initializes the OkavangoData class, caching the result to prevent 
    re-downloading and re-merging on every Streamlit interaction.

    Args:
        _dataset_config (Dict[str, str]): Dictionary of dataset URLs. 
            (Prefixed with '_' to tell Streamlit not to hash it for caching).

    Returns:
        OkavangoData: The instantiated data processing class.
    """
    return OkavangoData(_dataset_config)


def find_country_column(columns: list[str]) -> Optional[str]:
    """
    Identifies the column name containing country names in the dataset.

    Args:
        columns (list[str]): List of column names from the GeoDataFrame.

    Returns:
        Optional[str]: The name of the country column if found, else None.
    """
    candidates = ("ADMIN", "admin", "NAME", "name")
    for col in candidates:
        if col in columns:
            return col
    return None


def main() -> None:
    """Main execution function for the Streamlit app."""
    st.set_page_config(page_title="Project Okavango", layout="wide")
    st.title("🌍 Project Okavango: Environmental Data Tool")

    dataset_config = build_dataset_config()

    with st.spinner("Loading and processing datasets (Fetching most recent data)..."):
        okavango = get_processed_data(dataset_config)
        gdf = okavango.get_data()

    country_col = find_country_column(list(gdf.columns))
    if country_col is None:
        st.error("Country column not found in the geographic data.")
        st.stop()

    dataset_to_metric: Dict[str, str] = {
        "Annual change in forest area": "net_change_forest_area",
        "Annual deforestation": "_1d_deforestation",
        "Share of land that is protected": "er_lnd_ptld_zs",
        "Share of land that is degraded": "_15_3_1__ag_lnd_dgrd",
        "Red List Index (5th dataset)": "_15_5_1__er_rsk_lst",
    }

    dataset_name = st.selectbox(
        "Select dataset (one map at a time):",
        list(dataset_to_metric.keys()),
    )

    metric_col = dataset_to_metric[dataset_name]

    # Note: gdf is already filtered to the most recent available year per country 
    # via the OkavangoData class logic.
    gdf_plot = gdf.copy()

    # ---------- MAP ----------
    st.subheader(f"World Map: {dataset_name}")
    st.caption("Displaying the most recent available data for each country.")

    fig_map, ax_map = plt.subplots(figsize=(15, 8))
    gdf_plot.plot(
        column=metric_col,
        ax=ax_map,
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "No Data"},
    )
    ax_map.set_axis_off()
    st.pyplot(fig_map, clear_figure=True)

    # ---------- TOP/BOTTOM ----------
    st.divider()
    st.subheader(f"Top 5 vs Bottom 5 Countries: {dataset_name}")

    chart_data = (
        gdf_plot[[country_col, metric_col]]
        .dropna(subset=[metric_col])
        .astype({metric_col: float})
    )

    if chart_data.empty:
        st.warning("No tabular data available to plot for this metric.")
        st.stop()

    chart_desc = chart_data.sort_values(by=metric_col, ascending=False)
    top_5 = chart_desc.head(5)
    bottom_5 = chart_data.sort_values(by=metric_col, ascending=True).head(5)

    left_col, right_col = st.columns(2)

    # Detect formatting style based on dataset name
    is_index_dataset = "Index" in dataset_name
    is_percentage = "Share" in dataset_name

    def format_value(v: float) -> str:
        """Formats the metric values for chart labels based on the dataset type."""
        if is_index_dataset:
            return f"{v:.3f}"
        if is_percentage:
            return f"{v:.2f}%"
        return f"{v:,.0f}"

    def apply_axis_format(ax: plt.Axes) -> None:
        """Applies specific formatting to the x-axis depending on the dataset."""
        if not is_index_dataset and not is_percentage:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x:,.0f}")
            )

    # ---- TOP 5 CHART ----
    with left_col:
        st.markdown("### ✅ Top 5")
        fig_top, ax_top = plt.subplots(figsize=(7, 4))
        ax_top.barh(
            top_5[country_col].astype(str),
            top_5[metric_col],
            color="green",
        )
        ax_top.invert_yaxis()
        apply_axis_format(ax_top)

        for i, value in enumerate(top_5[metric_col].tolist()):
            ax_top.text(value, i, f" {format_value(value)}", va="center")

        if is_index_dataset:
            min_val = chart_data[metric_col].min()
            max_val = chart_data[metric_col].max()
            ax_top.set_xlim(min_val * 0.999, max_val * 1.001)

        st.pyplot(fig_top, clear_figure=True)

    # ---- BOTTOM 5 CHART ----
    with right_col:
        st.markdown("### ❌ Bottom 5")
        fig_bottom, ax_bottom = plt.subplots(figsize=(7, 4))
        ax_bottom.barh(
            bottom_5[country_col].astype(str),
            bottom_5[metric_col],
            color="red",
        )
        ax_bottom.invert_yaxis()
        apply_axis_format(ax_bottom)

        for i, value in enumerate(bottom_5[metric_col].tolist()):
            ax_bottom.text(value, i, f" {format_value(value)}", va="center")

        if is_index_dataset:
            min_val = chart_data[metric_col].min()
            max_val = chart_data[metric_col].max()
            ax_bottom.set_xlim(min_val * 0.999, max_val * 1.001)

        st.pyplot(fig_bottom, clear_figure=True)

if __name__ == "__main__":
    main()