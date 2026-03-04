"""
apps/main_app.py
Interactive Streamlit dashboard for Project Okavango.

Features
- Interactive world map
- Click country to show KPIs
- Region/continent filter
- Dataset selector
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from main import OkavangoData  # noqa: E402


# ---------------------------------------------------------
# DATA CONFIG
# ---------------------------------------------------------

def build_dataset_config() -> Dict[str, str]:
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
    return OkavangoData(dataset_config)


# ---------------------------------------------------------
# COLUMN HELPERS
# ---------------------------------------------------------

def find_country_column(columns: list[str]) -> Optional[str]:
    for col in ("ADMIN", "admin", "NAME", "name", "NAME_EN"):
        if col in columns:
            return col
    return None


def find_region_column(columns: list[str]) -> Optional[str]:
    for col in ("CONTINENT", "continent", "REGION_UN", "region_un"):
        if col in columns:
            return col
    return None


# ---------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------

def latest_year_with_metric_data(
    df: pd.DataFrame,
    metric_col: str,
) -> Tuple[pd.DataFrame, Optional[int]]:

    if "year" not in df.columns:
        return df.copy(), None

    metric_series = pd.to_numeric(df[metric_col], errors="coerce")
    valid = df.loc[metric_series.notna()].copy()

    if valid.empty:
        return df.copy(), None

    latest_year = int(valid["year"].max())

    return df.loc[df["year"] == latest_year].copy(), latest_year


# ---------------------------------------------------------
# MAP
# ---------------------------------------------------------

def build_map(
    df: pd.DataFrame,
    country_col: str,
    metric_col: str,
    dataset_name: str,
):

    fig = px.choropleth(
        df,
        locations="code",
        locationmode="ISO-3",
        color=metric_col,
        hover_name=country_col,
        hover_data={metric_col: ":,.2f"},
        color_continuous_scale="Viridis",
        title=f"World Map: {dataset_name}",
    )

    fig.update_traces(
        marker_line_width=0.5,
        marker_line_color="rgba(40,40,40,0.8)",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=560,
    )

    return fig


# ---------------------------------------------------------
# CLICK HANDLING
# ---------------------------------------------------------

def get_selection_iso3(selection_event):

    if selection_event is None:
        return None

    sel = getattr(selection_event, "selection", None)

    if not sel:
        return None

    points = sel.get("points", [])

    if not points:
        return None

    return points[0].get("location")


# ---------------------------------------------------------
# KPI CALCULATION
# ---------------------------------------------------------

def compute_country_kpis(
    df_year,
    country_col,
    metric_col,
    selected_iso3,
):

    data = df_year[[country_col, "code", metric_col]].copy()

    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    data = data.dropna()

    row = data[data["code"] == selected_iso3]

    if row.empty:
        return {"found": False}

    value = float(row.iloc[0][metric_col])
    country = row.iloc[0][country_col]

    ranked = data.sort_values(metric_col, ascending=False).reset_index(drop=True)

    rank = int(ranked.index[ranked["code"] == selected_iso3][0]) + 1
    total = len(ranked)

    return {
        "found": True,
        "country": country,
        "value": value,
        "rank": rank,
        "total": total,
        "world_avg": ranked[metric_col].mean(),
        "world_min": ranked[metric_col].min(),
        "world_max": ranked[metric_col].max(),
    }


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

def main():

    st.set_page_config(
        page_title="Project Okavango",
        layout="wide",
    )

    st.title("🌍 Project Okavango: Interactive Dashboard")

    dataset_to_metric = {
        "Annual change in forest area": "net_change_forest_area",
        "Annual deforestation": "_1d_deforestation",
        "Share of land protected": "er_lnd_ptld_zs",
        "Share of degraded land": "_15_3_1__ag_lnd_dgrd",
        "Red List Index": "_15_5_1__er_rsk_lst",
    }

    dataset_name = st.selectbox(
        "Select dataset",
        list(dataset_to_metric.keys()),
    )

    metric_col = dataset_to_metric[dataset_name]

    dataset_config = build_dataset_config()

    with st.spinner("Loading data..."):

        okavango = get_processed_data(dataset_config)
        gdf = okavango.get_data()

    country_col = find_country_column(list(gdf.columns))
    region_col = find_region_column(list(gdf.columns))

    if country_col is None:
        st.error("Country column not found")
        st.stop()

    gdf_year, year_used = latest_year_with_metric_data(gdf, metric_col)

    if year_used:
        st.caption(f"Using most recent year with data: {year_used}")

    # ---------------------------------------------------------
    # REGION FILTER
    # ---------------------------------------------------------

    if region_col:

        regions = sorted(gdf_year[region_col].dropna().unique())

        selected_regions = st.multiselect(
            "Filter by region",
            regions,
            default=regions,
        )

        gdf_year = gdf_year[gdf_year[region_col].isin(selected_regions)]

    if gdf_year.empty:
        st.warning("No countries match the selected region filter.")
        st.stop()

    # ---------------------------------------------------------
    # MAP
    # ---------------------------------------------------------

    st.subheader("World Map (click country)")

    map_fig = build_map(
        gdf_year,
        country_col,
        metric_col,
        dataset_name,
    )

    if "selected_iso3" not in st.session_state:
        st.session_state.selected_iso3 = None

    selection_event = st.plotly_chart(
        map_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    clicked = get_selection_iso3(selection_event)

    if clicked:
        st.session_state.selected_iso3 = clicked

    if st.button("Clear selection"):
        st.session_state.selected_iso3 = None

    # ---------------------------------------------------------
    # KPIs
    # ---------------------------------------------------------

    st.subheader("Country KPIs")

    iso3 = st.session_state.selected_iso3

    if iso3 is None:

        st.info("Click a country on the map to see KPIs")

        values = pd.to_numeric(gdf_year[metric_col], errors="coerce").dropna()

        c1, c2, c3 = st.columns(3)

        c1.metric("Countries with data", len(values))
        c2.metric("World average", f"{values.mean():,.2f}")
        c3.metric("World range", f"{values.min():,.2f} – {values.max():,.2f}")

        return

    kpis = compute_country_kpis(
        gdf_year,
        country_col,
        metric_col,
        iso3,
    )

    if not kpis["found"]:
        st.warning("Selected country has no data.")
        return

    st.markdown(f"### {kpis['country']}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Value", f"{kpis['value']:,.2f}")
    c2.metric("Rank", f"{kpis['rank']} / {kpis['total']}")
    c3.metric("World average", f"{kpis['world_avg']:,.2f}")
    c4.metric("World range", f"{kpis['world_min']:,.2f} – {kpis['world_max']:,.2f}")


if __name__ == "__main__":
    main()