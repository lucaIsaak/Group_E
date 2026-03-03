"""
apps/main_app.py
Frontend Streamlit dashboard for Project Okavango.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import FuncFormatter

# Ensure the apps folder can see the main.py file in the root directory
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

# Tell Pylint to ignore the dynamic import
from main import OkavangoData  # pylint: disable=import-error, wrong-import-position


def build_dataset_config() -> Dict[str, str]:
    """Builds and returns the dictionary of dataset URLs."""
    base_params = "csvType=full&useColumnShortNames=true"

    return {
        "annual_change_forest_area.csv": (
            "https://ourworldindata.org/grapher/annual-change-forest-area.csv?"
            f"{base_params}"
        ),
        "annual_deforestation.csv": (
            "https://ourworldindata.org/grapher/annual-deforestation.csv?"
            f"{base_params}"
        ),
        "protected_land.csv": (
            "https://ourworldindata.org/grapher/terrestrial-protected-areas.csv?"
            f"{base_params}"
        ),
        "degraded_land.csv": (
            "https://ourworldindata.org/grapher/share-degraded-land.csv?"
            f"{base_params}"
        ),
        "red_list_index.csv": (
            "https://ourworldindata.org/grapher/red-list-index.csv?"
            f"{base_params}"
        ),
        "ne_110m_admin_0_countries.zip": (
            "https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip"
        ),
    }


# Cache the initialization so the class doesn't re-download data on every click!
@st.cache_resource
def get_processed_data(dataset_config: Dict[str, str]) -> OkavangoData:
    """Initializes and caches the OkavangoData class."""
    return OkavangoData(dataset_config)


def find_country_column(columns: list[str]) -> Optional[str]:
    """Finds the column name representing countries in the dataframe."""
    candidates = ("ADMIN", "admin", "NAME", "name")
    for col in candidates:
        if col in columns:
            return col
    return None


def main() -> None:
    """Main Streamlit application layout and logic."""
    # Tell Pylint it's okay for a Streamlit app to have a lot of variables
    # pylint: disable=too-many-locals, too-many-statements

    st.set_page_config(page_title="Project Okavango", layout="wide")
    st.title("🌍 Project Okavango: Environmental Data Tool")

    dataset_config = build_dataset_config()

    with st.spinner("Loading and processing datasets..."):
        # This triggers your class __init__, downloading and merging everything
        okavango = get_processed_data(dataset_config)
        gdf = okavango.get_data()

    country_col = find_country_column(list(gdf.columns))
    if country_col is None:
        st.error("Country column not found in the merged map.")
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

    # Because OkavangoData already handles the most recent year logic,
    # we can use the dataframe exactly as it comes out of the class!
    gdf_plot = gdf.copy()
    st.caption("Displaying the most recent available data per country.")

    # ---------- MAP ----------
    st.subheader(f"World Map: {dataset_name}")

    fig_map, ax_map = plt.subplots(figsize=(15, 8))

    # Check if the metric actually exists in the dataframe before plotting
    if metric_col not in gdf_plot.columns:
        st.error(
            f"Could not find data column '{metric_col}' in the merged dataset."
            "Check your CSV column names!"
        )
        st.stop()

    gdf_plot.plot(
        column=metric_col,
        ax=ax_map,
        legend=True,
        missing_kwds={"color": "lightgrey"},
    )
    ax_map.set_axis_off()
    st.pyplot(fig_map, clear_figure=True)

    # ---------- TOP/BOTTOM ----------
    st.divider()
    st.subheader(f"Top 5 vs Bottom 5 Countries: {dataset_name}")

    chart_data = (
        gdf_plot[[country_col, metric_col]]
        .dropna()
        .astype({metric_col: float})
    )

    if chart_data.empty:
        st.warning("No data available for this metric.")
        st.stop()

    chart_desc = chart_data.sort_values(by=metric_col, ascending=False)
    top_5 = chart_desc.head(5)
    bottom_5 = chart_data.sort_values(by=metric_col, ascending=True).head(5)

    left_col, right_col = st.columns(2)

    # Detect formatting style
    is_index_dataset = "Index" in dataset_name
    is_percentage = "Share" in dataset_name

    def format_value(v: float) -> str:
        if is_index_dataset:
            return f"{v:.3f}"
        if is_percentage:
            return f"{v:.2f}%"
        return f"{v:,.0f}"

    def apply_axis_format(ax):
        if not is_index_dataset and not is_percentage:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x:,.0f}")
            )

    # ---- TOP ----
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

    # ---- BOTTOM ----
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
