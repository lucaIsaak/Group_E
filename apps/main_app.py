"""
apps/main_app.py
Interactive Streamlit dashboard for Project Okavango.

Features
- Dataset selector
- Region/continent filter with Select all + Clear
- Interactive world map (click country)
- KPIs always shown for current filtered set
- If no country selected: show Top 5 + Bottom 5 bar charts
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


def find_country_column(columns: list[str]) -> Optional[str]:
    for col in ("ADMIN", "admin", "NAME", "name", "NAME_EN"):
        if col in columns:
            return col
    return None


def find_region_column(columns: list[str]) -> Optional[str]:
    # Natural Earth commonly provides CONTINENT; REGION_UN is another good fallback.
    for col in ("CONTINENT", "continent", "REGION_UN", "region_un", "REGION_WB", "region_wb"):
        if col in columns:
            return col
    return None


def normalize_region_name(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "Unknown"
    s = str(x).strip()
    return s if s else "Unknown"


def latest_year_with_metric_data(
    df: pd.DataFrame, metric_col: str
) -> Tuple[pd.DataFrame, Optional[int]]:
    if "year" not in df.columns:
        return df.copy(), None

    metric_series = pd.to_numeric(df[metric_col], errors="coerce")
    valid = df.loc[metric_series.notna()].copy()
    if valid.empty:
        return df.copy(), None

    latest_year = int(pd.to_numeric(valid["year"], errors="coerce").max())
    return df.loc[pd.to_numeric(df["year"], errors="coerce") == latest_year].copy(), latest_year


def build_map(df: pd.DataFrame, country_col: str, metric_col: str, dataset_name: str):
    plot_df = df.copy()
    plot_df["code"] = plot_df["code"].astype(str)
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")

    fig = px.choropleth(
        plot_df,
        locations="code",
        locationmode="ISO-3",
        color=metric_col,
        hover_name=country_col,
        hover_data={"code": True, metric_col: ":,.3f"},
        color_continuous_scale="Viridis",
        title=f"World Map: {dataset_name}",
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(40,40,40,0.8)")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=560)
    return fig


def get_selection_iso3(selection_event) -> Optional[str]:
    if selection_event is None:
        return None
    sel = getattr(selection_event, "selection", None)
    if not sel:
        return None
    points = sel.get("points", []) if isinstance(sel, dict) else []
    if not points:
        return None
    iso3 = points[0].get("location")
    return str(iso3) if iso3 else None


def compute_set_kpis(df: pd.DataFrame, metric_col: str) -> dict:
    vals = pd.to_numeric(df[metric_col], errors="coerce").dropna().astype(float)
    if vals.empty:
        return {"has_data": False}

    return {
        "has_data": True,
        "countries_with_data": int(vals.shape[0]),
        "avg": float(vals.mean()),
        "median": float(vals.median()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def compute_country_kpis(
    df_year: pd.DataFrame,
    country_col: str,
    metric_col: str,
    selected_iso3: str,
) -> dict:
    tmp = df_year[[country_col, "code", metric_col]].copy()
    tmp["code"] = tmp["code"].astype(str)
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp = tmp.dropna(subset=[metric_col])

    row = tmp.loc[tmp["code"] == selected_iso3]
    if row.empty:
        return {"found": False}

    country_name = str(row.iloc[0][country_col])
    value = float(row.iloc[0][metric_col])

    ranked = tmp.sort_values(metric_col, ascending=False).reset_index(drop=True)
    rank_pos = int(ranked.index[ranked["code"] == selected_iso3][0]) + 1
    total = int(len(ranked))

    return {
        "found": True,
        "country": country_name,
        "iso3": selected_iso3,
        "value": value,
        "rank": rank_pos,
        "total": total,
    }


def compute_top_bottom(df: pd.DataFrame, country_col: str, metric_col: str, n: int = 5):
    data = df[[country_col, "code", metric_col]].copy()
    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    data = data.dropna(subset=[metric_col])

    top_n = data.sort_values(metric_col, ascending=False).head(n)
    bottom_n = data.sort_values(metric_col, ascending=True).head(n)
    return top_n, bottom_n


def main() -> None:
    st.set_page_config(page_title="Project Okavango", layout="wide")
    st.title("🌍 Project Okavango: Interactive Dashboard")

    dataset_to_metric: Dict[str, str] = {
        "Annual change in forest area": "net_change_forest_area",
        "Annual deforestation": "_1d_deforestation",
        "Share of land that is protected": "er_lnd_ptld_zs",
        "Share of land that is degraded": "_15_3_1__ag_lnd_dgrd",
        "Red List Index": "_15_5_1__er_rsk_lst",
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

    region_col = find_region_column(list(gdf.columns))
    if region_col is None:
        st.warning("Region/continent column not found. Region filtering disabled.")

    if metric_col not in gdf.columns:
        st.error(f"Metric column '{metric_col}' not found in merged data.")
        st.stop()

    if "code" not in gdf.columns:
        st.error("Column 'code' (ISO-3) not found. Choropleth needs ISO-3 codes.")
        st.stop()

    gdf_year, year_used = latest_year_with_metric_data(gdf, metric_col)
    if year_used is not None:
        st.caption(f"Using most recent year with data: {year_used}")

   # ---------- REGION FILTER ----------
    if region_col is not None:
        gdf_year[region_col] = gdf_year[region_col].apply(normalize_region_name)
        all_regions = sorted(
        [r for r in gdf_year[region_col].dropna().unique().tolist() if r != "Unknown"]
    )

    # Ensure state exists and is valid
    if "regions_filter" not in st.session_state:
        st.session_state.regions_filter = all_regions.copy()
    else:
        # If dataset/year changes, remove regions that no longer exist
        st.session_state.regions_filter = [
            r for r in st.session_state.regions_filter if r in all_regions
        ]
        # If empty (e.g., after switching dataset), reset to all
        if not st.session_state.regions_filter:
            st.session_state.regions_filter = all_regions.copy()

    # Buttons should update the SAME state that the widget uses
    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        if st.button("Select all regions", key="btn_select_all_regions"):
            st.session_state.regions_filter = all_regions.copy()
    with b2:
        if st.button("Clear regions", key="btn_clear_regions"):
            st.session_state.regions_filter = []

    # IMPORTANT: Use key=... and DO NOT pass default=...
    st.multiselect(
        "Filter by region/continent:",
        options=all_regions,
        key="regions_filter",
    )

    # Apply filter using the widget state
    gdf_year = gdf_year[gdf_year[region_col].isin(st.session_state.regions_filter)].copy()
    
    # ---------- MAP ----------
    st.subheader("World Map (click a country)")
    map_fig = build_map(gdf_year, country_col, metric_col, dataset_name)

    if "selected_iso3" not in st.session_state:
        st.session_state.selected_iso3 = None

    # If a country was selected but got filtered out, clear it
    if st.session_state.selected_iso3:
        if st.session_state.selected_iso3 not in set(gdf_year["code"].astype(str)):
            st.session_state.selected_iso3 = None

    selection_event = st.plotly_chart(
        map_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    clicked_iso3 = get_selection_iso3(selection_event)
    if clicked_iso3:
        st.session_state.selected_iso3 = clicked_iso3

    col_a, col_b = st.columns([1, 6])
    with col_a:
        if st.button("Clear country"):
            st.session_state.selected_iso3 = None

    # ---------- KPIs ALWAYS SHOWN ----------
    st.subheader("KPIs (based on current filters)")

    set_kpis = compute_set_kpis(gdf_year, metric_col)
    if not set_kpis["has_data"]:
        st.error("No KPI data available for current filters.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Countries with data", f"{set_kpis['countries_with_data']:,}")
    k2.metric("Average", f"{set_kpis['avg']:,.3f}")
    k3.metric("Median", f"{set_kpis['median']:,.3f}")
    k4.metric("Range", f"{set_kpis['min']:,.3f} to {set_kpis['max']:,.3f}")

    # ---------- COUNTRY KPIs (if selected) OR TOP/BOTTOM CHARTS (if not) ----------
    selected_iso3 = st.session_state.selected_iso3

    if selected_iso3:
        st.markdown("### Selected country")

        country_kpis = compute_country_kpis(gdf_year, country_col, metric_col, selected_iso3)
        if not country_kpis.get("found"):
            st.warning("Selected country has no data under current filters.")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Country", f"{country_kpis['country']} ({country_kpis['iso3']})")
        c2.metric("Value", f"{country_kpis['value']:,.3f}")
        c3.metric("Rank (within filtered set)", f"{country_kpis['rank']:,} / {country_kpis['total']:,}")

    else:
        st.markdown("### Top 5 and Bottom 5 countries (within current filters)")

        top_5, bottom_5 = compute_top_bottom(gdf_year, country_col, metric_col, n=5)

        left, right = st.columns(2)

        with left:
            st.markdown("#### ✅ Top 5")
            fig_top = px.bar(
                top_5.sort_values(metric_col, ascending=True),
                x=metric_col,
                y=country_col,
                orientation="h",
                hover_data={"code": True},
            )
            fig_top.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_top, use_container_width=True)

        with right:
            st.markdown("#### ❌ Bottom 5")
            fig_bottom = px.bar(
                bottom_5.sort_values(metric_col, ascending=True),
                x=metric_col,
                y=country_col,
                orientation="h",
                hover_data={"code": True},
            )
            fig_bottom.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_bottom, use_container_width=True)


if __name__ == "__main__":
    main()