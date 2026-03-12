"""
apps/main_app.py

Interactive Streamlit dashboard for Project Okavango.

Features
- Dataset selector
- Region/continent filter with Select all + Clear
- Interactive world map (click country)
- KPIs always shown for current filtered set
- If no country selected: show Top 5 + Bottom 5 bar charts
- Deselecting on the map clears the selected country
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import base64

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------
# Import project code
# ---------------------------------------------------------------------

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from main import OkavangoData  # pylint: disable=wrong-import-position,import-error
from satellite import fetch_satellite_image  # pylint: disable=wrong-import-position,import-error,wrong-import-order
from ollama_analysis import (  # pylint: disable=wrong-import-position,import-error,wrong-import-order
    describe_satellite_image,
    assess_environmental_risk,
    extract_risk_verdict,
    VISION_MODEL,
    TEXT_MODEL,
    _PROMPT,
    _RISK_PROMPT_TEMPLATE,
)
from db import log_run, find_existing_run, DB_PATH  # pylint: disable=wrong-import-position,import-error,wrong-import-order
import folium  # pylint: disable=wrong-import-position,import-error,wrong-import-order
from streamlit_folium import st_folium  # pylint: disable=wrong-import-position,import-error,wrong-import-order

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BASE_PARAMS = "csvType=full&useColumnShortNames=true"

REGION_STATE_KEY = "regions_filter"
COUNTRY_STATE_KEY = "selected_iso3"

BTN_SELECT_ALL_KEY = "btn_select_all_regions"
BTN_CLEAR_REGIONS_KEY = "btn_clear_regions"

DEFAULT_MAP_HEIGHT = 560
DEFAULT_BAR_HEIGHT = 320

PAGE_MAP = "World Map"
PAGE_AI = "AI Workflow"
PAGE_HISTORY = "History"
PAGE_ABOUT = "About"
PAGE_KEY = "current_page"

P2_LAT_KEY = "p2_latitude"
P2_LON_KEY = "p2_longitude"
P2_ZOOM_KEY = "p2_zoom"
P2_LAST_CLICK_KEY = "p2_last_click"    # last click already processed, to ignore stale repeats
P2_STAGED_LAT_KEY = "p2_staged_lat"   # staging key: applied before widgets render
P2_STAGED_LON_KEY = "p2_staged_lon"
P2_SIZE_KEY = "p2_image_size"
P2_IMAGE_PATH_KEY = "p2_image_path"
P2_DESCRIPTION_KEY = "p2_description"
P2_RISK_ASSESSMENT_KEY = "p2_risk_assessment"
P2_RISK_VERDICT_KEY = "p2_risk_verdict"

@dataclass(frozen=True)
class SetKpis:
    """KPIs computed over the currently filtered set."""

    countries_with_data: int
    avg: float
    median: float
    min_val: float
    max_val: float


@dataclass(frozen=True)
class CountryKpis:
    """KPIs computed for a selected country within the filtered set."""

    country: str
    iso3: str
    value: float
    rank: int
    total: int


def build_dataset_config() -> Dict[str, str]:
    """Return dataset URLs used by OkavangoData."""
    return {
        "annual_change_forest_area.csv": (
            "https://ourworldindata.org/grapher/annual-change-forest-area.csv?"
            f"{BASE_PARAMS}"
        ),
        "annual_deforestation.csv": (
            "https://ourworldindata.org/grapher/annual-deforestation.csv?"
            f"{BASE_PARAMS}"
        ),
        "protected_land.csv": (
            "https://ourworldindata.org/grapher/terrestrial-protected-areas.csv?"
            f"{BASE_PARAMS}"
        ),
        "degraded_land.csv": (
            "https://ourworldindata.org/grapher/share-degraded-land.csv?"
            f"{BASE_PARAMS}"
        ),
        "red_list_index.csv": (
            "https://ourworldindata.org/grapher/red-list-index.csv?"
            f"{BASE_PARAMS}"
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


def find_country_column(columns: Sequence[str]) -> Optional[str]:
    """Pick a country label column from Natural Earth."""
    for col in ("ADMIN", "admin", "NAME", "name", "NAME_EN"):
        if col in columns:
            return col
    return None


def find_region_column(columns: Sequence[str]) -> Optional[str]:
    """Pick a continent/region column from Natural Earth (if present)."""
    for col in (
        "CONTINENT",
        "continent",
        "REGION_UN",
        "region_un",
        "REGION_WB",
        "region_wb",
    ):
        if col in columns:
            return col
    return None


def normalize_region_name(value: object) -> str:
    """Normalize region values for UI display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def latest_year_with_metric_data(
    df: pd.DataFrame, metric_col: str
) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Filter to the most recent year that has non-null metric data.

    Returns (df_filtered, year_used).
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


def ensure_metric_numeric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Ensure metric column is numeric (coerce errors to NaN)."""
    out = df.copy()
    out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    return out


def init_session_state(all_regions: List[str]) -> None:
    """Initialize session state keys if missing."""
    if REGION_STATE_KEY not in st.session_state:
        st.session_state[REGION_STATE_KEY] = all_regions.copy()
    if COUNTRY_STATE_KEY not in st.session_state:
        st.session_state[COUNTRY_STATE_KEY] = None


def render_region_filter(gdf_year: pd.DataFrame, region_col: Optional[str]) -> pd.DataFrame:
    """Render region filter UI and return filtered dataframe."""
    if region_col is None:
        return gdf_year

    df = gdf_year.copy()
    df[region_col] = df[region_col].apply(normalize_region_name)

    all_regions = sorted(
        [r for r in df[region_col].dropna().unique().tolist() if r != "Unknown"]
    )

    init_session_state(all_regions)

    current = [r for r in st.session_state[REGION_STATE_KEY] if r in all_regions]
    if not current:
        current = all_regions.copy()
    st.session_state[REGION_STATE_KEY] = current

    btn_col1, btn_col2, _ = st.columns([1, 1, 6])
    with btn_col1:
        if st.button("Select all regions", key=BTN_SELECT_ALL_KEY):
            st.session_state[REGION_STATE_KEY] = all_regions.copy()
    with btn_col2:
        if st.button("Clear regions", key=BTN_CLEAR_REGIONS_KEY):
            st.session_state[REGION_STATE_KEY] = []

    st.multiselect(
        "Filter by region/continent:",
        options=all_regions,
        key=REGION_STATE_KEY,
    )

    return df[df[region_col].isin(st.session_state[REGION_STATE_KEY])].copy()


def build_map(
    df: pd.DataFrame,
    country_col: str,
    metric_col: str,
    dataset_name: str,
):
    """Build Plotly choropleth."""
    plot_df = df.copy()

    if "code" not in plot_df.columns:
        raise KeyError("Column 'code' (ISO-3) missing. Choropleth needs ISO-3 codes.")

    plot_df["code"] = plot_df["code"].astype(str)
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")

    _scale = [[0.0, "#e05a5a"], [0.5, "#f5c842"], [1.0, "#3ecb8a"]]

    fig = px.choropleth(
        plot_df,
        locations="code",
        locationmode="ISO-3",
        color=metric_col,
        hover_name=country_col,
        hover_data={"code": True, metric_col: ":,.3f"},
        color_continuous_scale=_scale,
        title=f"World Map — {dataset_name}",
    )

    fig.update_traces(marker_line_width=0.4, marker_line_color="rgba(0,0,0,0.6)")
    fig.update_geos(
        showocean=True, oceancolor="#0d2535",
        showlakes=True, lakecolor="#0d2535",
        showrivers=True, rivercolor="#0d2535",
        showcountries=True, countrycolor="rgba(255,255,255,0.12)",
        showcoastlines=True, coastlinecolor="rgba(255,255,255,0.18)",
        bgcolor="#0a2420",
        landcolor="#132e28",
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 46, "b": 0},
        height=DEFAULT_MAP_HEIGHT,
        paper_bgcolor="#0a2420",
        font={"color": "#ffffff", "family": "Barlow Condensed, sans-serif"},
        title_font_color="#3ecb8a",
        coloraxis_colorbar={
            "tickfont": {"color": "#ffffff"},
            "title": {"font": {"color": "#7fbfb0"}},
            "bgcolor": "#0a2420",
            "bordercolor": "#1e4d42",
            "borderwidth": 1,
        },
    )
    return fig


def selection_is_empty(selection_event) -> bool:
    """
    Return True if we received a selection event but it contains no selected points.
    """
    if selection_event is None:
        return False

    selection = getattr(selection_event, "selection", None)
    if not selection:
        return True

    points = selection.get("points", []) if isinstance(selection, dict) else []
    return len(points) == 0


def get_selection_iso3(selection_event) -> Optional[str]:
    """Extract ISO-3 code from Streamlit Plotly selection event."""
    if selection_event is None:
        return None

    selection = getattr(selection_event, "selection", None)
    if not selection:
        return None

    points = selection.get("points", []) if isinstance(selection, dict) else []
    if not points:
        return None

    iso3 = points[0].get("location")
    return str(iso3) if iso3 else None


def clear_country_if_filtered_out(df: pd.DataFrame) -> None:
    """Clear selected country if it is no longer present after filtering."""
    selected = st.session_state.get(COUNTRY_STATE_KEY)
    if not selected:
        return

    available = set(df["code"].astype(str).tolist())
    if selected not in available:
        st.session_state[COUNTRY_STATE_KEY] = None


def compute_set_kpis(df: pd.DataFrame, metric_col: str) -> Optional[SetKpis]:
    """Compute KPIs over the current filtered set."""
    values = pd.to_numeric(df[metric_col], errors="coerce").dropna().astype(float)
    if values.empty:
        return None

    return SetKpis(
        countries_with_data=int(values.shape[0]),
        avg=float(values.mean()),
        median=float(values.median()),
        min_val=float(values.min()),
        max_val=float(values.max()),
    )


def compute_country_kpis(
    df_year: pd.DataFrame,
    country_col: str,
    metric_col: str,
    selected_iso3: str,
) -> Optional[CountryKpis]:
    """Compute KPIs for a selected country (rank is within filtered set)."""
    tmp = df_year[[country_col, "code", metric_col]].copy()
    tmp["code"] = tmp["code"].astype(str)
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp = tmp.dropna(subset=[metric_col])

    row = tmp.loc[tmp["code"] == selected_iso3]
    if row.empty:
        return None

    country_name = str(row.iloc[0][country_col])
    value = float(row.iloc[0][metric_col])

    ranked = tmp.sort_values(metric_col, ascending=False).reset_index(drop=True)
    rank_pos = int(ranked.index[ranked["code"] == selected_iso3][0]) + 1
    total = int(len(ranked))

    return CountryKpis(
        country=country_name,
        iso3=selected_iso3,
        value=value,
        rank=rank_pos,
        total=total,
    )


def compute_top_bottom(
    df: pd.DataFrame, country_col: str, metric_col: str, n: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_n, bottom_n) dataframes by metric value."""
    data = df[[country_col, "code", metric_col]].copy()
    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    data = data.dropna(subset=[metric_col])

    top_n = data.sort_values(metric_col, ascending=False).head(n)
    bottom_n = data.sort_values(metric_col, ascending=True).head(n)
    return top_n, bottom_n


def render_set_kpis(kpis: SetKpis) -> None:
    """Render KPIs for the current filtered set."""
    st.subheader("Summary statistics")
    st.caption(
        "These figures cover all countries currently visible on the map "
        "(based on your region filter and the selected dataset)."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Countries with data", f"{kpis.countries_with_data:,}",
                help="Number of countries that have a recorded value for this metric in the most recent available year.")
    col2.metric("Average", f"{kpis.avg:,.3f}",
                help="Mean value across all filtered countries.")
    col3.metric("Range", f"{kpis.min_val:,.3f} → {kpis.max_val:,.3f}",
                help="Lowest and highest recorded values within the current filter.")


def render_selected_country(kpis: CountryKpis) -> None:
    """Render KPIs for selected country."""
    st.markdown("### Selected country")
    col1, col2, col3 = st.columns(3)
    col1.metric("Country", f"{kpis.country} ({kpis.iso3})")
    col2.metric("Value", f"{kpis.value:,.3f}")
    col3.metric("Rank (within filtered set)", f"{kpis.rank:,} / {kpis.total:,}")


_CHART_LAYOUT = {
    "paper_bgcolor": "#0a2420",
    "plot_bgcolor": "#0a2420",
    "font": {"color": "#ffffff", "family": "Barlow Condensed, sans-serif"},
    "xaxis": {"gridcolor": "#1e4d42", "color": "#7fbfb0"},
    "yaxis": {"gridcolor": "#1e4d42", "color": "#ffffff"},
    "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
}


def render_top_bottom_charts(
    df: pd.DataFrame, country_col: str, metric_col: str
) -> None:
    """Render Top 5 and Bottom 5 horizontal bar charts."""
    st.markdown("### Top 5 and Bottom 5 countries (within current filters)")

    top_5, bottom_5 = compute_top_bottom(df, country_col, metric_col, n=5)
    left, right = st.columns(2)

    with left:
        st.markdown("#### Top 5")
        fig_top = px.bar(
            top_5.sort_values(metric_col, ascending=True),
            x=metric_col,
            y=country_col,
            orientation="h",
            hover_data={"code": True},
            color_discrete_sequence=["#3ecb8a"],
        )
        fig_top.update_layout(height=DEFAULT_BAR_HEIGHT, **_CHART_LAYOUT)
        st.plotly_chart(fig_top, use_container_width=True)

    with right:
        st.markdown("#### Bottom 5")
        fig_bottom = px.bar(
            bottom_5.sort_values(metric_col, ascending=True),
            x=metric_col,
            y=country_col,
            orientation="h",
            hover_data={"code": True},
            color_discrete_sequence=["#e05a5a"],
        )
        fig_bottom.update_layout(height=DEFAULT_BAR_HEIGHT, **_CHART_LAYOUT)
        st.plotly_chart(fig_bottom, use_container_width=True)


def _update_country_selection(selection_event) -> None:
    """Update the selected country in session state from a Plotly map event."""
    if selection_is_empty(selection_event):
        st.session_state[COUNTRY_STATE_KEY] = None
    else:
        clicked_iso3 = get_selection_iso3(selection_event)
        if clicked_iso3:
            st.session_state[COUNTRY_STATE_KEY] = clicked_iso3


def _render_kpis_above_map(
    gdf_year: pd.DataFrame, country_col: str, metric_col: str, set_kpis: SetKpis
) -> None:
    """Render set-level KPIs and, if a country is selected, its individual KPIs."""
    render_set_kpis(set_kpis)
    selected_iso3 = st.session_state.get(COUNTRY_STATE_KEY)
    if not selected_iso3:
        return
    country_kpis = compute_country_kpis(gdf_year, country_col, metric_col, selected_iso3)
    if country_kpis is None:
        st.warning("Selected country has no data under current filters.")
        return
    render_selected_country(country_kpis)


_ESRI_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


P2_MAP_KEY = "p2_map_key"  # incremented on reset to force full folium re-render


def _init_page2_state() -> None:
    """Initialise Page 2 session-state keys on first load."""
    defaults = {P2_LAT_KEY: 0.0, P2_LON_KEY: 0.0, P2_ZOOM_KEY: 12, P2_SIZE_KEY: "512 px",
                P2_MAP_KEY: 0}
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_folium_map() -> None:
    """Render the interactive satellite map and update state on click."""
    st.subheader("Location Preview")
    st.caption("Click on the map to set coordinates. Zoom slider is below.")

    lat = st.session_state[P2_LAT_KEY]
    lon = st.session_state[P2_LON_KEY]

    base_map = folium.Map(location=[0, 0], zoom_start=2, tiles=_ESRI_TILES,
                          attr="Esri World Imagery", max_zoom=18)
    pin_group = folium.FeatureGroup(name="pin")
    folium.Marker(
        location=[lat, lon],
        tooltip=f"{lat:.4f}, {lon:.4f}",
        icon=folium.Icon(color="red", icon="map-marker"),
    ).add_to(pin_group)

    _, map_col, _ = st.columns([1, 6, 1])
    with map_col:
        map_data = st_folium(
            base_map, height=440, use_container_width=True,
            returned_objects=["last_clicked"],
            center=[lat, lon],
            feature_group_to_add=pin_group,
            key=f"folium_map_{st.session_state[P2_MAP_KEY]}",
        )

    if map_data and map_data.get("last_clicked"):
        new_lat = round(map_data["last_clicked"]["lat"], 4)
        new_lon = round(map_data["last_clicked"]["lng"], 4)
        new_click = (new_lat, new_lon)
        if new_click != st.session_state.get(P2_LAST_CLICK_KEY):
            st.session_state[P2_LAST_CLICK_KEY] = new_click
            st.session_state[P2_STAGED_LAT_KEY] = new_lat
            st.session_state[P2_STAGED_LON_KEY] = new_lon
            st.rerun()

    if st.button("Reset pin", type="tertiary"):
        st.session_state[P2_STAGED_LAT_KEY] = 0.0
        st.session_state[P2_STAGED_LON_KEY] = 0.0
        st.session_state[P2_MAP_KEY] = st.session_state[P2_MAP_KEY] + 1
        st.session_state.pop(P2_LAST_CLICK_KEY, None)
        st.rerun()


def _render_coordinate_inputs() -> None:
    """Render manual lat/lon inputs."""
    st.subheader("Coordinates")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                        step=0.0001, format="%.4f",
                        help="Decimal degrees — negative values are South.",
                        key=P2_LAT_KEY)
    with col2:
        st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                        step=0.0001, format="%.4f",
                        help="Decimal degrees — negative values are West.",
                        key=P2_LON_KEY)


def _render_image_capture_settings() -> None:
    """Render zoom and resolution controls for the satellite image download."""
    st.markdown(
        '<p style="font-size:0.78rem;color:#7fbfb0;margin:0.2rem 0 0.6rem 0;'
        'text-transform:uppercase;letter-spacing:0.8px;">'
        'The settings below apply to the downloaded image, not the map preview.'
        '</p>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Image zoom level", min_value=1, max_value=18,
                  help="Higher zoom = more detail, smaller geographic area captured.",
                  key=P2_ZOOM_KEY)
    with col2:
        st.select_slider("Image resolution", options=["256 px", "512 px", "1024 px"],
                         help="Pixel dimensions of the saved satellite image.",
                         key=P2_SIZE_KEY)


def _render_fetch_and_describe() -> None:
    """Render the fetch button, satellite image, and AI description."""
    if st.button("Fetch Satellite Image", type="primary"):
        size_px = int(st.session_state[P2_SIZE_KEY].split()[0])
        cached = find_existing_run(st.session_state[P2_LAT_KEY],
                                   st.session_state[P2_LON_KEY],
                                   st.session_state[P2_ZOOM_KEY], size_px)
        if cached:
            st.info("Loaded from cache — skipping pipeline to save compute.")
            st.session_state[P2_IMAGE_PATH_KEY] = cached["image_path"]
            st.session_state[P2_DESCRIPTION_KEY] = cached["image_description"]
            st.session_state[P2_RISK_ASSESSMENT_KEY] = cached["text_description"]
            st.session_state[P2_RISK_VERDICT_KEY] = cached["danger"]
        else:
            with st.spinner("Downloading satellite imagery…"):
                try:
                    img_path = fetch_satellite_image(
                        st.session_state[P2_LAT_KEY], st.session_state[P2_LON_KEY],
                        st.session_state[P2_ZOOM_KEY], size_px)
                    st.session_state[P2_IMAGE_PATH_KEY] = str(img_path)
                    for key in (P2_DESCRIPTION_KEY, P2_RISK_ASSESSMENT_KEY, P2_RISK_VERDICT_KEY):
                        st.session_state.pop(key, None)
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Could not fetch satellite image: {exc}")

    if not st.session_state.get(P2_IMAGE_PATH_KEY):
        return

    title_col, clear_col = st.columns([5, 1])
    with title_col:
        st.subheader("Satellite Image & AI Analysis")
    with clear_col:
        st.markdown("<div style='padding-top:0.6rem;'>", unsafe_allow_html=True)
        if st.button("Clear results", type="tertiary", use_container_width=True):
            for key in (P2_IMAGE_PATH_KEY, P2_DESCRIPTION_KEY,
                        P2_RISK_ASSESSMENT_KEY, P2_RISK_VERDICT_KEY):
                st.session_state.pop(key, None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    img_col, desc_col = st.columns(2)
    with img_col:
        st.image(st.session_state[P2_IMAGE_PATH_KEY], use_container_width=True)
    with desc_col:
        st.markdown("**AI Description**")
        if st.session_state.get(P2_DESCRIPTION_KEY):
            st.markdown(st.session_state[P2_DESCRIPTION_KEY])
        else:
            st.caption("Click the button below to generate an AI description.")

        if st.button("Analyse with AI", type="secondary"):
            size_px = int(st.session_state[P2_SIZE_KEY].split()[0])
            cached = find_existing_run(st.session_state[P2_LAT_KEY],
                                       st.session_state[P2_LON_KEY],
                                       st.session_state[P2_ZOOM_KEY], size_px)
            if cached:
                st.info("Loaded from cache — skipping pipeline to save compute.")
                st.session_state[P2_DESCRIPTION_KEY] = cached["image_description"]
                st.session_state[P2_RISK_ASSESSMENT_KEY] = cached["text_description"]
                st.session_state[P2_RISK_VERDICT_KEY] = cached["danger"]
                st.rerun()
            img_path = Path(st.session_state[P2_IMAGE_PATH_KEY])
            spinner_msg = "Pulling model if needed and analysing image… this may take a minute."
            with st.spinner(spinner_msg):
                try:
                    full_description = st.write_stream(describe_satellite_image(img_path))
                    st.session_state[P2_DESCRIPTION_KEY] = full_description
                    for key in (P2_RISK_ASSESSMENT_KEY, P2_RISK_VERDICT_KEY):
                        st.session_state.pop(key, None)
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"AI analysis failed: {exc}")


def _is_q_line(line: str) -> bool:
    """Return True if line looks like 'Q1: ...'."""
    return len(line) >= 3 and line[0] == "Q" and line[1].isdigit() and line[2] == ":"


def _format_assessment_html(text: str) -> str:
    """Convert raw assessment text into styled HTML rows."""
    rows = []
    q_num = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith(("OVERALL VERDICT", "OVERALL:")):
            if "NOT AT RISK" in upper:
                bg, fg = "#1e4d42", "#3ecb8a"
            elif "AT RISK" in upper:
                bg, fg = "#4d1e1e", "#e05a5a"
            else:
                bg, fg = "#4d3e00", "#f5c842"
            rows.append(
                f'<div style="margin-top:1rem;padding:0.7rem 1rem;'
                f'background:{bg};border-radius:8px;'
                f'font-family:Barlow Condensed,sans-serif;font-weight:700;'
                f'font-size:1.05rem;color:{fg};">{line}</div>'
            )
        elif upper.startswith("SUMMARY"):
            rows.append(
                f'<p style="color:#7fbfb0;margin:0.4rem 0 0 0;'
                f'font-style:italic;font-size:0.9rem;">{line}</p>'
            )
        elif _is_q_line(line):
            q_num += 1
            rows.append(
                '<div style="display:flex;gap:1rem;margin:0.55rem 0;'
                'padding:0.5rem 0.8rem;background:#0a2420;border-radius:8px;'
                'border-left:3px solid #3ecb8a;">'
                '<span style="color:#3ecb8a;font-weight:800;'
                'font-family:Barlow Condensed,sans-serif;'
                f'min-width:1.4rem;font-size:1rem;">{q_num}</span>'
                f'<span style="font-size:0.9rem;">{line[3:].strip()}</span>'
                "</div>"
            )
        else:
            rows.append(
                f'<p style="margin:0.3rem 0;font-size:0.9rem;">{line}</p>'
            )
    return "\n".join(rows)


def _render_risk_assessment() -> None:
    """Render the environmental risk assessment section."""
    description = st.session_state.get(P2_DESCRIPTION_KEY)
    if st.session_state.get(P2_RISK_ASSESSMENT_KEY):
        st.subheader("Environmental Risk Assessment")
        st.markdown(
            _format_assessment_html(st.session_state[P2_RISK_ASSESSMENT_KEY]),
            unsafe_allow_html=True,
        )
        verdict = st.session_state.get(P2_RISK_VERDICT_KEY, "UNCERTAIN")
        if verdict == "AT RISK":
            st.error("AREA FLAGGED AS ENVIRONMENTALLY AT RISK")
        elif verdict == "NOT AT RISK":
            st.success("No significant environmental risk detected")
        else:
            st.warning("Environmental risk is uncertain — manual review recommended")
    elif description:
        st.subheader("Environmental Risk Assessment")
        with st.spinner("Assessing environmental risk… this may take a moment."):
            try:
                full_assessment = st.write_stream(assess_environmental_risk(description))
                verdict = extract_risk_verdict(full_assessment)
                st.session_state[P2_RISK_ASSESSMENT_KEY] = full_assessment
                st.session_state[P2_RISK_VERDICT_KEY] = verdict
                log_run(
                    latitude=st.session_state[P2_LAT_KEY],
                    longitude=st.session_state[P2_LON_KEY],
                    zoom=st.session_state[P2_ZOOM_KEY],
                    image_size_px=int(st.session_state[P2_SIZE_KEY].split()[0]),
                    image_path=st.session_state[P2_IMAGE_PATH_KEY],
                    image_model=VISION_MODEL, image_prompt=_PROMPT,
                    image_description=description, text_model=TEXT_MODEL,
                    text_prompt=_RISK_PROMPT_TEMPLATE.format(description=description),
                    text_description=full_assessment, danger=verdict,
                )
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Risk assessment failed: {exc}")


def render_page2() -> None:
    """Render Page 2: coordinate selection for the AI satellite-imagery workflow."""
    st.markdown(
        """
        <div class="ok-page-hero">
          <p class="ok-page-title">AI Workflow</p>
          <p class="ok-page-desc">
            Select a location on Earth — fetch satellite imagery and run AI environmental risk detection.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _init_page2_state()
    # Apply any click staged by the previous run BEFORE widgets are instantiated,
    # to avoid Streamlit's "cannot modify widget-owned key" error.
    if P2_STAGED_LAT_KEY in st.session_state:
        st.session_state[P2_LAT_KEY] = st.session_state.pop(P2_STAGED_LAT_KEY)
        st.session_state[P2_LON_KEY] = st.session_state.pop(P2_STAGED_LON_KEY)
    _render_coordinate_inputs()
    _render_folium_map()
    _render_image_capture_settings()
    _render_fetch_and_describe()
    _render_risk_assessment()


# ---------------------------------------------------------------------------
# Page 3 — History
# ---------------------------------------------------------------------------

def _verdict_badge(verdict: str) -> str:
    """Return a coloured HTML badge for a risk verdict."""
    colours = {
        "AT RISK":     ("#e05a5a", "#2a0a0a"),
        "NOT AT RISK": ("#3ecb8a", "#071917"),
        "UNCERTAIN":   ("#f5c842", "#1a1400"),
    }
    bg, fg = colours.get(verdict.strip().upper(), ("#7fbfb0", "#0a2420"))
    return (
        f'<span style="background:{bg};color:{fg};padding:0.2rem 0.7rem;'
        f'border-radius:20px;font-size:0.78rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.8px;white-space:nowrap;">'
        f'{verdict}</span>'
    )


def render_page3() -> None:
    """Render Page 3: browsable history of past AI analyses."""
    st.markdown(
        """
        <div class="ok-page-hero">
          <p class="ok-page-title">Analysis History</p>
          <p class="ok-page-desc">
            All past satellite AI analyses — most recent first.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        st.info("No analyses yet. Run the AI Workflow on a location to see results here.")
        return

    df = pd.read_csv(DB_PATH)
    if df.empty:
        st.info("No analyses yet.")
        return

    # Newest first
    df = df.iloc[::-1].reset_index(drop=True)

    _count_col, _btn_col = st.columns([6, 1])
    _count_col.caption(f"{len(df)} result{'s' if len(df) != 1 else ''} in database")
    with _btn_col:
        if st.button("Clear history", type="secondary", use_container_width=True):
            DB_PATH.write_text("")
            st.rerun()

    _BORDER_COLOUR = {"AT RISK": "#e05a5a", "NOT AT RISK": "#3ecb8a", "UNCERTAIN": "#f5c842"}

    for i, row in df.iterrows():
        verdict     = str(row.get("danger", "UNCERTAIN")).strip().upper()
        lat         = row.get("latitude",       "—")
        lon         = row.get("longitude",      "—")
        zoom        = row.get("zoom",           "—")
        size        = row.get("image_size_px",  "—")
        ts          = str(row.get("timestamp", ""))[:19].replace("T", "  ")
        img_path    = str(row.get("image_path",        ""))
        description = str(row.get("image_description", ""))
        assessment  = str(row.get("text_description",  ""))

        border = _BORDER_COLOUR.get(verdict, "#7fbfb0")

        # Compact card row — always visible
        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {border};
                background: #071917;
                border-radius: 0 8px 8px 0;
                padding: 0.85rem 1.2rem;
                margin-bottom: 0.25rem;
                display: flex;
                align-items: center;
                gap: 1.4rem;
                flex-wrap: wrap;
            ">
              {_verdict_badge(verdict)}
              <span style="font-size:0.95rem;font-weight:600;color:#ffffff;">
                Lat {lat}° &nbsp; Lon {lon}°
              </span>
              <span style="color:#5a9e8a;font-size:0.82rem;">
                Zoom {zoom} &nbsp;·&nbsp; {size} px
              </span>
              <span style="color:#3d7a6a;font-size:0.8rem;margin-left:auto;">
                {ts} UTC
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Expandable detail panel
        with st.expander("View details", expanded=False):
            img_col, detail_col = st.columns([1, 2])

            with img_col:
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    st.caption("Image file not found on disk.")

            with detail_col:
                if description:
                    st.markdown("**AI Description**")
                    st.markdown(description)
                if assessment:
                    st.markdown("**Risk Assessment**")
                    st.markdown(
                        _format_assessment_html(assessment),
                        unsafe_allow_html=True,
                    )

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)


def render_about() -> None:
    """Render the About page."""
    st.markdown(
        """
        <div class="ok-page-hero">
          <p class="ok-page-title">About</p>
          <p class="ok-page-desc">What this project is, how it works, and who built it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Mission ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#071917;border-radius:10px;padding:1.6rem 2rem;margin-bottom:1.5rem;">
          <p style="font-size:1.05rem;font-weight:600;color:#3ecb8a;
                    text-transform:uppercase;letter-spacing:1px;margin:0 0 0.6rem 0;">
            Our Mission
          </p>
          <p style="font-size:0.97rem;color:#d4f5e9;line-height:1.7;margin:0;">
            Project Okavango is an environmental intelligence platform built to make global
            deforestation and land-degradation data accessible and actionable. It combines
            open satellite imagery with local AI models to flag areas at environmental risk —
            helping researchers, policymakers, and curious citizens understand what is
            happening to our planet's ecosystems in near real-time.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── How to use ───────────────────────────────────────────────────────────
    st.markdown("### How to use this platform")

    col1, col2, col3 = st.columns(3)
    _card_style = (
        "background:#071917;border:1px solid #1e4d42;border-radius:10px;"
        "padding:1.3rem 1.4rem;height:100%;"
    )
    with col1:
        st.markdown(
            f"""<div style="{_card_style}">
              <p style="color:#3ecb8a;font-weight:700;font-size:0.8rem;
                        text-transform:uppercase;letter-spacing:1px;margin:0 0 0.5rem 0;">
                World Map
              </p>
              <p style="color:#d4f5e9;font-size:0.88rem;line-height:1.6;margin:0;">
                Choose one of five environmental datasets and explore how countries compare.
                Use the region filter to focus on a continent, click any country to see its
                individual ranking and value, and scroll down for the Top 5 / Bottom 5 breakdown.
              </p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div style="{_card_style}">
              <p style="color:#3ecb8a;font-weight:700;font-size:0.8rem;
                        text-transform:uppercase;letter-spacing:1px;margin:0 0 0.5rem 0;">
                AI Workflow
              </p>
              <p style="color:#d4f5e9;font-size:0.88rem;line-height:1.6;margin:0;">
                Click anywhere on the satellite map (or type coordinates manually) to select
                a location. Fetch the satellite image, then run the two-step AI pipeline:
                a vision model describes the terrain, and a language model delivers an
                environmental risk verdict — AT RISK, NOT AT RISK, or UNCERTAIN.
              </p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div style="{_card_style}">
              <p style="color:#3ecb8a;font-weight:700;font-size:0.8rem;
                        text-transform:uppercase;letter-spacing:1px;margin:0 0 0.5rem 0;">
                History
              </p>
              <p style="color:#d4f5e9;font-size:0.88rem;line-height:1.6;margin:0;">
                Every completed AI analysis is saved to a local database. The History page
                lets you browse all past results, compare verdicts across locations, and
                revisit the satellite images and AI assessments without re-running the
                (slow) models.
              </p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    st.markdown("### Data sources")
    datasets = [
        ("Annual change in forest area",    "FAO / Our World in Data", "Net yearly gain or loss of forest cover per country, in hectares."),
        ("Annual deforestation",            "FAO / Our World in Data", "Forest area permanently cleared each year. Does not offset regrowth."),
        ("Share of land that is protected", "World Bank",              "% of national territory under official protection (parks, reserves, etc.)."),
        ("Share of land that is degraded",  "UN SDG 15.3.1",           "% of land degraded by erosion, desertification, or soil loss."),
        ("Red List Index",                  "IUCN / Our World in Data","Extinction-risk index from 0.0 (worst) to 1.0 (all species safe)."),
    ]
    for name, source, desc in datasets:
        st.markdown(
            f"""
            <div style="display:flex;gap:1rem;align-items:flex-start;
                        padding:0.8rem 0;border-bottom:1px solid #1e4d42;">
              <div style="min-width:220px;font-weight:600;
                          color:#ffffff;font-size:0.88rem;">{name}</div>
              <div style="min-width:160px;color:#5a9e8a;
                          font-size:0.82rem;padding-top:1px;">{source}</div>
              <div style="color:#a8d8c8;font-size:0.85rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Tech stack ───────────────────────────────────────────────────────────
    st.markdown("### Technology")
    tech = [
        ("Streamlit",       "Web dashboard framework"),
        ("Ollama + llava",  "Local vision model — describes satellite imagery"),
        ("Ollama + llama3.2","Local language model — generates risk assessment"),
        ("ESRI World Imagery","Free satellite tile provider (no API key required)"),
        ("GeoPandas / Plotly","Geospatial data processing and interactive maps"),
        ("Our World in Data","Open environmental datasets (CC BY licence)"),
    ]
    t_cols = st.columns(2)
    for idx, (tech_name, tech_desc) in enumerate(tech):
        with t_cols[idx % 2]:
            st.markdown(
                f"""<div style="background:#071917;border:1px solid #1e4d42;
                               border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.6rem;">
                  <span style="color:#3ecb8a;font-weight:700;
                               font-size:0.85rem;">{tech_name}</span>
                  <span style="color:#7fbfb0;font-size:0.82rem;
                               margin-left:0.6rem;">{tech_desc}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Team ─────────────────────────────────────────────────────────────────
    st.markdown("### Team")
    st.markdown(
        """
        <div style="background:#071917;border:1px solid #1e4d42;border-radius:10px;
                    padding:1.2rem 1.6rem;display:inline-block;">
          <p style="color:#7fbfb0;font-size:0.88rem;margin:0;line-height:1.7;">
            Built by <strong style="color:#ffffff;">Group E</strong> as part of the
            <strong style="color:#ffffff;">Advanced Programming for Data Science</strong>
            course — Nova SBE, 2026.<br>
            Submitted to the hackathon track addressing UN Sustainable Development Goals
            <strong style="color:#3ecb8a;">SDG 2</strong>,
            <strong style="color:#3ecb8a;">SDG 13</strong>, and
            <strong style="color:#3ecb8a;">SDG 15</strong>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page1() -> None:
    """Render Page 1: interactive world-map dashboard."""
    st.markdown(
        """
        <div class="ok-page-hero">
          <p class="ok-page-title">World Map</p>
          <p class="ok-page-desc">
            Explore global environmental metrics — click a country for details.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    dataset_to_metric: Dict[str, str] = {
        "Annual change in forest area": "net_change_forest_area",
        "Annual deforestation": "_1d_deforestation",
        "Share of land that is protected": "er_lnd_ptld_zs",
        "Share of land that is degraded": "_15_3_1__ag_lnd_dgrd",
        "Red List Index": "_15_5_1__er_rsk_lst",
    }

    _METRIC_INFO: Dict[str, str] = {
        "Annual change in forest area": (
            "Net gain or loss of forest area per year, in hectares. "
            "Positive values mean forests are growing; negative values mean more forest is being lost than gained. "
            "Source: Our World in Data / FAO."
        ),
        "Annual deforestation": (
            "Total forest area permanently cleared each year, in hectares. "
            "This only counts losses — unlike the metric above, it does not offset gains. "
            "Higher values indicate greater destruction. Source: Our World in Data / FAO."
        ),
        "Share of land that is protected": (
            "Percentage of a country's total land area officially designated as protected — "
            "national parks, nature reserves, wildlife sanctuaries, etc. "
            "Higher values mean more land is shielded from industrial use. Source: World Bank."
        ),
        "Share of land that is degraded": (
            "Percentage of land affected by degradation: erosion, desertification, soil depletion, "
            "or loss of vegetation cover caused by human activity. "
            "Tracks UN Sustainable Development Goal 15.3.1. Source: Our World in Data / UN."
        ),
        "Red List Index": (
            "Measures the overall extinction risk across species groups (mammals, birds, amphibians, etc.). "
            "Ranges from 1.0 (all species at least concern) down to 0.0 (all species extinct). "
            "A declining index means biodiversity is worsening. Source: IUCN / Our World in Data."
        ),
    }

    _datasets_list = list(dataset_to_metric.keys())
    try:
        _ds_idx = int(st.query_params.get("ds", "0"))
        if _ds_idx < 0 or _ds_idx >= len(_datasets_list):
            _ds_idx = 0
    except (ValueError, TypeError):
        _ds_idx = 0
    _current_page = st.query_params.get("page", "world_map")
    _btns_html = "".join(
        f'<a href="?page={_current_page}&ds={i}" target="_self" '
        f'class="ok-dataset-btn {"ok-ds-active" if i == _ds_idx else "ok-ds-inactive"}">'
        f'{name}</a>'
        for i, name in enumerate(_datasets_list)
    )
    st.markdown(f'<div class="ok-dataset-row">{_btns_html}</div>', unsafe_allow_html=True)
    dataset_name = _datasets_list[_ds_idx]
    metric_col = dataset_to_metric[dataset_name]

    st.markdown(
        f"""
        <div style="background:#071917;border-left:3px solid #3ecb8a;
                    border-radius:0 8px 8px 0;padding:0.75rem 1.1rem;
                    margin:0.5rem 0 1.2rem 0;">
          <span style="font-size:0.82rem;color:#a8d8c8;line-height:1.5;">
            {_METRIC_INFO[dataset_name]}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading data..."):
        okavango = get_processed_data(build_dataset_config())
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

    gdf_year = render_region_filter(gdf_year, region_col)
    if gdf_year.empty:
        st.warning("No countries match the current region filter.")
        st.stop()

    gdf_year = ensure_metric_numeric(gdf_year, metric_col)
    if gdf_year.dropna(subset=[metric_col]).empty:
        st.error("No metric values available for the current filters.")
        st.stop()

    init_session_state(all_regions=[])
    clear_country_if_filtered_out(gdf_year)

    set_kpis = compute_set_kpis(gdf_year, metric_col)
    if set_kpis is None:
        st.error("No KPI data available for current filters.")
        st.stop()

    render_set_kpis(set_kpis)
    country_stats_slot = st.empty()

    st.subheader("World Map (click a country)")
    map_fig = build_map(gdf_year, country_col, metric_col, dataset_name)
    selection_event = st.plotly_chart(
        map_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    _update_country_selection(selection_event)

    selected_iso3 = st.session_state.get(COUNTRY_STATE_KEY)
    if selected_iso3:
        with country_stats_slot.container():
            country_kpis = compute_country_kpis(
                gdf_year, country_col, metric_col, selected_iso3
            )
            if country_kpis is None:
                st.warning("Selected country has no data under current filters.")
            else:
                render_selected_country(country_kpis)

    render_top_bottom_charts(gdf_year, country_col, metric_col)


_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Barlow+Condensed:wght@400;600;700;800"
    '&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">'
)

_LOGO_SVG_SOURCE = (
    '<svg width="38" height="38" viewBox="0 0 38 38" fill="none"'
    ' xmlns="http://www.w3.org/2000/svg">'
    '<polygon points="19,3 35,34 3,34" fill="#3ecb8a" fill-opacity="0.12"'
    ' stroke="#3ecb8a" stroke-width="1.6" stroke-linejoin="round"/>'
    '<line x1="19" y1="9" x2="19" y2="32" stroke="#3ecb8a" stroke-width="1.5" stroke-linecap="round"/>'
    '<line x1="19" y1="15" x2="13" y2="28" stroke="#3ecb8a" stroke-width="1.2" stroke-linecap="round" opacity="0.85"/>'
    '<line x1="19" y1="15" x2="25" y2="28" stroke="#3ecb8a" stroke-width="1.2" stroke-linecap="round" opacity="0.85"/>'
    '<line x1="19" y1="21" x2="10" y2="32" stroke="#3ecb8a" stroke-width="0.9" stroke-linecap="round" opacity="0.5"/>'
    '<line x1="19" y1="21" x2="28" y2="32" stroke="#3ecb8a" stroke-width="0.9" stroke-linecap="round" opacity="0.5"/>'
    '</svg>'
)
_LOGO_IMG = (
    '<img src="data:image/svg+xml;base64,'
    + base64.b64encode(_LOGO_SVG_SOURCE.encode()).decode()
    + '" width="38" height="38" style="display:block;flex-shrink:0;">'
)


def _inject_css() -> None:
    """Inject global custom CSS for a polished, green-themed UI."""
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(
        """<style>
        html,body,[class*="css"],.stApp,.stMarkdown,.stMarkdown p,
        button,input,select,label,textarea,
        [data-testid="stText"],[data-testid="stCaptionContainer"]
        { font-family:'Barlow',sans-serif !important; }
        h1,h2,h3,h4,h5,h6,.stMarkdown h1,.stMarkdown h2,
        .stMarkdown h3,.stMarkdown h4
        { font-family:'Barlow',sans-serif !important;
          font-weight:700 !important; letter-spacing:0.2px !important;
          text-transform:none !important; }
        .stApp,[data-testid="stAppViewContainer"],
        [data-testid="stMain"],[data-testid="block-container"]
        { background-color:#0d2e2a !important; color:#ffffff !important; }
        [data-testid="stHeader"] { display:none !important; }
        html,body { overflow-x:hidden !important; }
        [data-testid="stAppViewContainer"],[data-testid="stMain"]
        { overflow-x:hidden !important; max-width:100vw !important; }
        [data-testid="block-container"]
        { max-width:1200px !important; padding-left:2.5rem !important;
          padding-right:2.5rem !important; padding-top:0 !important;
          margin-left:auto !important; margin-right:auto !important; }
        h1,h2,h3,h4,h5,h6,p,label,span,.stMarkdown,.stMarkdown p
        { color:#ffffff !important; }
        .stCaption,[data-testid="stCaptionContainer"] p
        { color:#7fbfb0 !important; }
        section[data-testid="stSidebar"] { display:none !important; }

        /* ── Navbar ─────────────────────────────────────────────── */
        .ok-navbar {
            display:flex; align-items:center; justify-content:space-between;
            background:#071917;
            border-bottom:1px solid #1e4d42;
            padding:0 2.5rem;
            height:68px;
            position:sticky; top:0; z-index:1000;
            width:calc(100% + 5rem);
            margin-left:-2.5rem; margin-right:-2.5rem;
            margin-bottom:2rem;
            box-sizing:border-box;
        }
        .ok-brand { display:flex; align-items:center; gap:0.75rem; }
        .ok-brand-text { display:flex; flex-direction:column; line-height:1; }
        .ok-title {
            font-family:'Barlow Condensed',sans-serif !important;
            font-size:1.65rem !important; font-weight:800 !important;
            letter-spacing:2.5px !important; text-transform:uppercase !important;
            color:#ffffff !important; white-space:nowrap;
        }
        .ok-subtitle {
            font-size:0.68rem !important; color:#5a9e8a !important;
            text-transform:uppercase !important; letter-spacing:1.2px !important;
            margin-top:3px;
        }
        .ok-nav { display:flex; align-items:stretch; height:68px; gap:0; }
        .ok-navlink {
            display:flex; align-items:center; gap:0.45rem;
            padding:0 1.6rem;
            font-family:'Barlow',sans-serif !important;
            font-size:0.95rem !important; font-weight:600 !important;
            text-decoration:none !important;
            border-bottom:2px solid transparent;
            color:#7fbfb0 !important;
            text-transform:uppercase !important; letter-spacing:1px !important;
            transition:color 0.15s, border-color 0.15s;
            white-space:nowrap;
        }
        .ok-navlink:hover { color:#d4f5e9 !important; border-bottom-color:#3ecb8a60 !important; }
        .ok-nav-active { color:#ffffff !important; border-bottom-color:#3ecb8a !important; }
        .ok-nav-icon { font-size:0.95rem; }

        /* ── Page hero strip ──────────────────────────────────────── */
        .ok-page-hero {
            padding:1.4rem 0 1.2rem 0;
            border-bottom:1px solid #1e4d42;
            margin-bottom:1.8rem;
        }
        .ok-page-title {
            font-family:'Barlow Condensed',sans-serif !important;
            font-size:1.7rem !important; font-weight:800 !important;
            letter-spacing:1.5px !important; text-transform:uppercase !important;
            color:#ffffff !important; margin:0 !important;
        }
        .ok-page-desc {
            font-size:0.82rem !important; color:#7fbfb0 !important;
            margin:0.3rem 0 0 0 !important; letter-spacing:0.3px !important;
        }

        /* ── Buttons ─────────────────────────────────────────────── */
        .stButton button[kind="primary"]
        { background-color:#3ecb8a !important; color:#071917 !important;
          border:none !important; border-radius:6px !important;
          font-weight:700 !important; }
        .stButton button[kind="secondary"]
        { background-color:transparent !important; color:#3ecb8a !important;
          border:1.5px solid #3ecb8a !important; border-radius:6px !important; }
        .stButton button[kind="tertiary"]
        { color:#7fbfb0 !important; border:1px solid #1e4d42 !important;
          border-radius:6px !important; }

        /* ── Dataset selector buttons ────────────────────────────── */
        .ok-dataset-row {
            display:flex; flex-wrap:wrap; gap:0.5rem;
            margin:0.5rem 0 1.2rem 0;
        }
        .ok-dataset-btn {
            padding:0.45rem 1rem; border-radius:8px;
            font-size:0.85rem !important; font-weight:600 !important;
            text-decoration:none !important; cursor:pointer;
            transition:opacity 0.15s; white-space:nowrap;
            letter-spacing:0.2px; display:inline-block;
        }
        .ok-ds-active {
            background:#3ecb8a !important; color:#071917 !important;
            border:1.5px solid #3ecb8a !important;
        }
        .ok-ds-inactive {
            background:#071917 !important; color:#3ecb8a !important;
            border:1.5px solid #3ecb8a !important;
        }
        .ok-ds-inactive:hover { opacity:0.75; }

        /* ── Form controls ───────────────────────────────────────── */
        .stSelectbox>div>div,.stNumberInput>div>div>input
        { background-color:#0a2420 !important; color:#ffffff !important;
          border:1px solid #1e4d42 !important; border-radius:8px !important; }

        /* Multiselect container and tags */
        .stMultiSelect>div,
        .stMultiSelect [data-baseweb="select"],
        .stMultiSelect [data-baseweb="select"]>div
        { background-color:#0a2420 !important; border:1px solid #1e4d42 !important;
          border-radius:8px !important; }
        .stMultiSelect [data-baseweb="tag"]
        { background-color:#1e4d42 !important; border-radius:4px !important; }
        .stMultiSelect [data-baseweb="tag"] span,
        .stMultiSelect [data-baseweb="tag"] [role="presentation"]
        { color:#3ecb8a !important; }
        .stMultiSelect input
        { background-color:transparent !important; color:#ffffff !important; }

        /* ── KPI cards ───────────────────────────────────────────── */
        div[data-testid="metric-container"]
        { background:#071917 !important; border:1px solid #1e4d42 !important;
          border-radius:10px !important; padding:1rem 1.1rem !important; }
        [data-testid="stMetricValue"] { color:#ffffff !important; }
        [data-testid="stMetricLabel"] p { color:#7fbfb0 !important; }

        /* ── History expanders ───────────────────────────────────── */
        [data-testid="stExpander"] {
            background:#071917 !important;
            border:1px solid #1e4d42 !important;
            border-radius:0 0 8px 8px !important;
            border-top:none !important;
            margin-top:-1px !important;
            margin-bottom:0 !important;
        }
        [data-testid="stExpander"] summary {
            color:#7fbfb0 !important;
            font-size:0.82rem !important;
            padding:0.5rem 1.2rem !important;
        }
        [data-testid="stExpander"] summary:hover { color:#ffffff !important; }

        /* ── Misc ────────────────────────────────────────────────── */
        hr { border-color:#1e4d42 !important; margin:0.5rem 0 1.2rem 0; }
        [data-testid="stAlert"] { border-radius:10px !important; }
        </style>""",
        unsafe_allow_html=True,
    )


def _render_navbar(active_page: str) -> None:
    """Render the sticky top navigation bar with logo and page links."""
    world_active   = "ok-nav-active" if active_page == PAGE_MAP     else ""
    ai_active      = "ok-nav-active" if active_page == PAGE_AI      else ""
    history_active = "ok-nav-active" if active_page == PAGE_HISTORY else ""
    about_active   = "ok-nav-active" if active_page == PAGE_ABOUT   else ""
    st.markdown(
        f"""
        <div class="ok-navbar">
          <div class="ok-brand">
            {_LOGO_IMG}
            <div class="ok-brand-text">
              <span class="ok-title">Project Okavango</span>
              <span class="ok-subtitle">Environmental Intelligence &amp; AI Risk Detection</span>
            </div>
          </div>
          <nav class="ok-nav">
            <a href="?page=world_map" target="_self" class="ok-navlink {world_active}">World Map</a>
            <a href="?page=ai_workflow" target="_self" class="ok-navlink {ai_active}">AI Workflow</a>
            <a href="?page=history" target="_self" class="ok-navlink {history_active}">History</a>
            <a href="?page=about" target="_self" class="ok-navlink {about_active}">About</a>
          </nav>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Project Okavango",
        page_icon=None,
        layout="wide",
    )
    _inject_css()
    # Derive active page from URL query params (set by navbar anchor links)
    page_param = st.query_params.get("page", "world_map")
    if page_param == "ai_workflow":
        active_page = PAGE_AI
    elif page_param == "history":
        active_page = PAGE_HISTORY
    elif page_param == "about":
        active_page = PAGE_ABOUT
    else:
        active_page = PAGE_MAP
    _render_navbar(active_page)
    if active_page == PAGE_AI:
        render_page2()
    elif active_page == PAGE_HISTORY:
        render_page3()
    elif active_page == PAGE_ABOUT:
        render_about()
    else:
        render_page1()

if __name__ == "__main__":
    main()
