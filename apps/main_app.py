"""
apps/main_app.py
Minimal Streamlit dashboard for Project Okavango (interactive selection + KPIs).

Run:
    streamlit run apps/main_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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


def latest_year_with_metric_data(
    df: pd.DataFrame, metric_col: str
) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Return (df_filtered_to_latest_year_with_metric_data, year).

    Prevents blank maps when absolute latest year has NaNs for this metric.
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


def build_map(df: pd.DataFrame, country_col: str, metric_col: str, dataset_name: str):
    """Build Plotly choropleth (colored) with click/selection enabled."""
    plot_df = df.copy()

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
        hover_data={"code": True, metric_col: ":,.3f"},
        color_continuous_scale="Viridis",
        title=f"World Map: {dataset_name}",
    )

    # visible borders
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(40,40,40,0.8)")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=560)
    return fig


def get_selection_iso3(selection_event) -> Optional[str]:
    """
    Extract ISO-3 code from Streamlit Plotly selection event.
    For choropleth, plotly point usually contains `location`.
    """
    if selection_event is None:
        return None

    # Newer Streamlit returns an object with `.selection`
    sel = getattr(selection_event, "selection", None)
    if not sel:
        return None

    points = sel.get("points", []) if isinstance(sel, dict) else []
    if not points:
        return None

    # Choropleth returns ISO3 in `location`
    iso3 = points[0].get("location")
    if iso3:
        return str(iso3)

    return None


def compute_country_kpis(
    df_year: pd.DataFrame,
    country_col: str,
    metric_col: str,
    selected_iso3: str,
) -> dict:
    """Compute KPI dictionary for the selected country for the current year slice."""
    tmp = df_year[[country_col, "code", metric_col]].copy()
    tmp["code"] = tmp["code"].astype(str)
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp = tmp.dropna(subset=[metric_col])

    row = tmp.loc[tmp["code"] == selected_iso3]
    if row.empty:
        return {"found": False}

    country_name = str(row.iloc[0][country_col])
    value = float(row.iloc[0][metric_col])

    # Ranking (higher = better) — you can flip this if your metric meaning differs
    ranked = tmp.sort_values(metric_col, ascending=False).reset_index(drop=True)
    rank_pos = int(ranked.index[ranked["code"] == selected_iso3][0]) + 1
    total = int(len(ranked))
    percentile = 100.0 * (1.0 - (rank_pos - 1) / max(total - 1, 1))

    world_avg = float(ranked[metric_col].mean())
    world_med = float(ranked[metric_col].median())
    world_min = float(ranked[metric_col].min())
    world_max = float(ranked[metric_col].max())

    return {
        "found": True,
        "country": country_name,
        "iso3": selected_iso3,
        "value": value,
        "rank": rank_pos,
        "total": total,
        "percentile": percentile,
        "world_avg": world_avg,
        "world_med": world_med,
        "world_min": world_min,
        "world_max": world_max,
        "diff_vs_avg": value - world_avg,
        "diff_vs_med": value - world_med,
    }


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

    gdf_year[metric_col] = pd.to_numeric(gdf_year[metric_col], errors="coerce")
    if gdf_year.dropna(subset=[metric_col]).empty:
        st.error("No metric values available to color the map for this dataset.")
        st.stop()

    # ---- MAP ----
    st.subheader("World Map (click a country)")
    map_fig = build_map(gdf_year, country_col, metric_col, dataset_name)

    # Persist selection across reruns
    if "selected_iso3" not in st.session_state:
        st.session_state.selected_iso3 = None

    # Streamlit Plotly selection (click/box/lasso)
    selection_event = st.plotly_chart(
        map_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    clicked_iso3 = get_selection_iso3(selection_event)
    if clicked_iso3:
        st.session_state.selected_iso3 = clicked_iso3

    # Optional: clear selection button
    col_a, col_b = st.columns([1, 6])
    with col_a:
        if st.button("Clear selection"):
            st.session_state.selected_iso3 = None

    # ---- KPIs (instead of bar charts) ----
    st.subheader("Country KPIs")

    selected_iso3 = st.session_state.selected_iso3
    if not selected_iso3:
        st.info("Click a country on the map to see KPIs here.")
        # Some global KPIs as a fallback
        valid_vals = gdf_year[metric_col].dropna().astype(float)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Countries with data", f"{len(valid_vals):,}")
        c2.metric("World average", f"{valid_vals.mean():,.3f}")
        c3.metric("World median", f"{valid_vals.median():,.3f}")
        c4.metric("World range", f"{valid_vals.min():,.3f} to {valid_vals.max():,.3f}")
        return

    kpis = compute_country_kpis(gdf_year, country_col, metric_col, selected_iso3)
    if not kpis.get("found"):
        st.warning(
            f"Selection '{selected_iso3}' has no data for the current year selection."
        )
        return

    st.markdown(f"### {kpis['country']} ({kpis['iso3']})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Value", f"{kpis['value']:,.3f}")
    c2.metric("Rank (higher is better)", f"{kpis['rank']:,} / {kpis['total']:,}")
    c3.metric("Percentile", f"{kpis['percentile']:.1f}%")
    c4.metric("Δ vs world avg", f"{kpis['diff_vs_avg']:,.3f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("World average", f"{kpis['world_avg']:,.3f}")
    c6.metric("World median", f"{kpis['world_med']:,.3f}")
    c7.metric("World min / max", f"{kpis['world_min']:,.3f} / {kpis['world_max']:,.3f}")

    # (Optional) show the exact row for transparency/debug
    with st.expander("Show selected row data"):
        show_df = gdf_year.loc[gdf_year["code"].astype(str) == selected_iso3, [country_col, "code", "year", metric_col]].copy()
        st.dataframe(show_df, use_container_width=True)


if __name__ == "__main__":
    main()