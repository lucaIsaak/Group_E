"""
apps/main_app.py
Minimal Streamlit dashboard for Project Okavango.

Run:
    streamlit run apps/main_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib.ticker import FuncFormatter

# Ensure apps/ can import from project root
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from main import OkavangoData  # noqa: E402  pylint: disable=wrong-import-position


def build_dataset_config() -> Dict[str, str]:
    """Build and return dataset URLs."""
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


@st.cache_resource
def get_processed_data(dataset_config: Dict[str, str]) -> OkavangoData:
    """Initialize and cache OkavangoData (avoids re-download each rerun)."""
    return OkavangoData(dataset_config)


def find_country_column(columns: list[str]) -> Optional[str]:
    """Pick a country label column from Natural Earth."""
    for col in ("ADMIN", "admin", "NAME", "name", "NAME_EN"):
        if col in columns:
            return col
    return None


def latest_year_with_metric_data(df: pd.DataFrame, metric_col: str) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Return (df_filtered_to_latest_year_with_metric_data, year).

    This prevents blank maps when the absolute latest year has NaNs for this metric.
    """
    if "year" not in df.columns:
        return df.copy(), None

    year_series = pd.to_numeric(df["year"], errors="coerce")
    metric_series = pd.to_numeric(df[metric_col], errors="coerce")

    valid = df.loc[metric_series.notna()].copy()
    if valid.empty:
        return df.copy(), None

    latest_year = int(pd.to_numeric(valid["year"], errors="coerce").max())
    return df.loc[year_series == latest_year].copy(), latest_year


def format_axis_int(x: float, _: int) -> str:
    """Format ticks without scientific notation."""
    return f"{x:,.0f}"


def apply_axis_format(ax: plt.Axes) -> None:
    """Avoid scientific notation for bar charts."""
    ax.xaxis.set_major_formatter(FuncFormatter(format_axis_int))


def build_map(df: pd.DataFrame, country_col: str, metric_col: str, dataset_name: str):
    """Build Plotly choropleth (colored)."""
    plot_df = df.copy()

    # Required for Plotly choropleth
    if "code" not in plot_df.columns:
        raise KeyError("Missing required 'code' column (ISO-3).")

    plot_df["code"] = plot_df["code"].astype(str)
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")

    fig = px.choropleth(
        plot_df,
        locations="code",
        locationmode="ISO-3",
        color=metric_col,
        hover_name=country_col,
        color_continuous_scale="Viridis",
        title=f"World Map: {dataset_name}",
    )

    # visible borders
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(40,40,40,0.8)")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=560)
    return fig


def compute_top_bottom(df: pd.DataFrame, country_col: str, metric_col: str, n: int = 5):
    """Compute top and bottom N by metric."""
    data = df[[country_col, metric_col]].copy()
    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    data = data.dropna(subset=[country_col, metric_col])

    top_n = data.sort_values(metric_col, ascending=False).head(n)
    bottom_n = data.sort_values(metric_col, ascending=True).head(n)
    return top_n, bottom_n


def main() -> None:
    """Streamlit app."""
    st.set_page_config(page_title="Project Okavango (Minimal)", layout="wide")
    st.title("🌍 Project Okavango: Minimal Dashboard")

    dataset_to_metric: Dict[str, str] = {
        "Annual change in forest area": "net_change_forest_area",
        "Annual deforestation": "_1d_deforestation",
        "Share of land that is protected": "er_lnd_ptld_zs",
        "Share of land that is degraded": "_15_3_1__ag_lnd_dgrd",
        "Red List Index (5th dataset)": "_15_5_1__er_rsk_lst",
    }

    dataset_name = st.selectbox("Select dataset:", list(dataset_to_metric.keys()))
    metric_col = dataset_to_metric[dataset_name]

    dataset_config = build_dataset_config()

    with st.spinner("Loading data..."):
        okavango = get_processed_data(dataset_config)
        gdf = okavango.get_data()

    country_col = find_country_column(list(gdf.columns))
    if country_col is None:
        st.error("Country column not found (expected ADMIN/NAME/etc).")
        st.stop()

    if metric_col not in gdf.columns:
        st.error(f"Metric column '{metric_col}' not found in merged data.")
        st.stop()

    if "code" not in gdf.columns:
        st.error("Column 'code' (ISO-3) not found. Choropleth needs ISO-3 codes.")
        st.stop()

    # Use latest year with data for THIS metric
    gdf_year, year_used = latest_year_with_metric_data(gdf, metric_col)
    if year_used is not None:
        st.caption(f"Using most recent year with data: {year_used}")

    # If metric is empty -> warn
    gdf_year[metric_col] = pd.to_numeric(gdf_year[metric_col], errors="coerce")
    if gdf_year.dropna(subset=[metric_col]).empty:
        st.error("No metric values available to color the map for this dataset.")
        st.stop()

    # ---- MAP ----
    st.subheader("World Map")
    try:
        map_fig = build_map(gdf_year, country_col, metric_col, dataset_name)
    except KeyError as exc:
        st.error(str(exc))
        st.stop()

    st.plotly_chart(map_fig, use_container_width=True)

    # ---- TOP/BOTTOM ----
    st.subheader("Top 5 and Bottom 5 Countries (same selection as map)")

    top_5, bottom_5 = compute_top_bottom(gdf_year, country_col, metric_col, n=5)

    left, right = st.columns(2)

    with left:
        st.markdown("### ✅ Top 5")
        fig_top, ax_top = plt.subplots(figsize=(7, 4))
        ax_top.barh(top_5[country_col].astype(str), top_5[metric_col].astype(float))
        ax_top.invert_yaxis()
        ax_top.set_xlabel("Value")
        apply_axis_format(ax_top)
        st.pyplot(fig_top, clear_figure=True)

    with right:
        st.markdown("### ❌ Bottom 5")
        fig_bottom, ax_bottom = plt.subplots(figsize=(7, 4))
        ax_bottom.barh(bottom_5[country_col].astype(str), bottom_5[metric_col].astype(float))
        ax_bottom.invert_yaxis()
        ax_bottom.set_xlabel("Value")
        apply_axis_format(ax_bottom)
        st.pyplot(fig_bottom, clear_figure=True)


if __name__ == "__main__":
    main()