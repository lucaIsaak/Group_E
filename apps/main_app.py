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
from db import log_run, find_existing_run  # pylint: disable=wrong-import-position,import-error,wrong-import-order
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
BTN_CLEAR_COUNTRY_KEY = "btn_clear_country"

DEFAULT_MAP_HEIGHT = 560
DEFAULT_BAR_HEIGHT = 320

PAGE_MAP = "World Map"
PAGE_AI = "AI Workflow"
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
    st.subheader("KPIs (based on current filters)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Countries with data", f"{kpis.countries_with_data:,}")
    col2.metric("Average", f"{kpis.avg:,.3f}")
    col3.metric("Range", f"{kpis.min_val:,.3f} to {kpis.max_val:,.3f}")


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


def _init_page2_state() -> None:
    """Initialise Page 2 session-state keys on first load."""
    defaults = {P2_LAT_KEY: 0.0, P2_LON_KEY: 0.0, P2_ZOOM_KEY: 12, P2_SIZE_KEY: "512 px"}
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
        st.session_state[P2_LAT_KEY] = 0.0
        st.session_state[P2_LON_KEY] = 0.0
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

    st.subheader("Satellite Image & AI Analysis")
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
            rows.append(
                '<div style="margin-top:1rem;padding:0.7rem 1rem;'
                'background:#1e4d42;border-radius:8px;'
                'font-family:Barlow Condensed,sans-serif;font-weight:700;'
                f'font-size:1.05rem;color:#3ecb8a;">{line}</div>'
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
        "Enter coordinates manually or click on the satellite map below "
        "to select an area of interest."
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


def render_page1() -> None:
    """Render Page 1: interactive world-map dashboard."""
    dataset_to_metric: Dict[str, str] = {
        "Annual change in forest area": "net_change_forest_area",
        "Annual deforestation": "_1d_deforestation",
        "Share of land that is protected": "er_lnd_ptld_zs",
        "Share of land that is degraded": "_15_3_1__ag_lnd_dgrd",
        "Red List Index": "_15_5_1__er_rsk_lst",
    }

    dataset_name = st.selectbox("Select dataset:", list(dataset_to_metric.keys()))
    metric_col = dataset_to_metric[dataset_name]

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

    _render_kpis_above_map(gdf_year, country_col, metric_col, set_kpis)

    st.subheader("World Map (click a country)")
    map_fig = build_map(gdf_year, country_col, metric_col, dataset_name)
    selection_event = st.plotly_chart(
        map_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    _update_country_selection(selection_event)

    if st.button("Clear country", key=BTN_CLEAR_COUNTRY_KEY):
        st.session_state[COUNTRY_STATE_KEY] = None

    render_top_bottom_charts(gdf_year, country_col, metric_col)


_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Barlow+Condensed:wght@400;600;700;800"
    '&family=Barlow:wght@400;500;600&display=swap" rel="stylesheet">'
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
        { font-family:'Barlow Condensed',sans-serif !important;
          font-weight:800 !important; letter-spacing:0.5px !important;
          text-transform:uppercase !important; }
        .stApp,[data-testid="stAppViewContainer"],
        [data-testid="stMain"],[data-testid="block-container"]
        { background-color:#0d2e2a !important; color:#ffffff !important; }
        [data-testid="stHeader"] { background-color:#0d2e2a !important; }
        html,body { overflow-x:hidden !important; }
        [data-testid="stAppViewContainer"],[data-testid="stMain"]
        { overflow-x:hidden !important; max-width:100vw !important; }
        [data-testid="block-container"]
        { max-width:1100px !important; padding-left:2.5rem !important;
          padding-right:2.5rem !important;
          margin-left:auto !important; margin-right:auto !important; }
        h1,h2,h3,h4,h5,h6,p,label,span,.stMarkdown,.stMarkdown p
        { color:#ffffff !important; }
        .stCaption,[data-testid="stCaptionContainer"] p
        { color:#7fbfb0 !important; }
        section[data-testid="stSidebar"] { display:none !important; }
        .nav-btn button { border-radius:24px !important;
          font-weight:700 !important; font-size:0.95rem !important;
          transition:transform 0.1s ease,opacity 0.1s ease; }
        .nav-btn button:hover { transform:scale(1.04); }
        .stButton button[kind="primary"]
        { background-color:#3ecb8a !important; color:#0d2e2a !important;
          border:none !important; }
        .stButton button[kind="secondary"]
        { background-color:transparent !important; color:#3ecb8a !important;
          border:1.5px solid #3ecb8a !important; }
        .stButton button[kind="tertiary"]
        { color:#7fbfb0 !important; border:1px solid #1e4d42 !important; }
        .stSelectbox>div>div,.stNumberInput>div>div>input,
        .stSlider [data-testid="stSlider"]
        { background-color:#0a2420 !important; color:#ffffff !important;
          border:1px solid #1e4d42 !important; border-radius:8px !important; }
        .stMultiSelect>div
        { background-color:#0a2420 !important;
          border:1px solid #1e4d42 !important; }
        div[data-testid="metric-container"]
        { background:#0a2420 !important; border:1px solid #1e4d42 !important;
          border-radius:12px !important; padding:0.9rem 1rem !important; }
        [data-testid="stMetricValue"] { color:#ffffff !important; }
        [data-testid="stMetricLabel"] p { color:#7fbfb0 !important; }
        hr { border-color:#1e4d42 !important; margin:0.5rem 0 1rem 0; }
        [data-testid="stAlert"] { border-radius:10px !important; }
        </style>""",
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    """Render the project banner at the top of every page."""
    st.markdown(
        """
        <div style="padding:1.2rem 0 0.6rem 0;">
          <h1 style="color:#ffffff;margin:0;
                     font-family:'Barlow Condensed',sans-serif;
                     font-size:2.6rem;font-weight:800;
                     letter-spacing:2px;text-transform:uppercase;
                     line-height:1.1;">
            Project Okavango
          </h1>
          <p style="color:#7fbfb0;margin:0.3rem 0 0.7rem 0;
                    font-family:'Barlow',sans-serif;font-size:0.88rem;
                    letter-spacing:0.5px;text-transform:uppercase;">
            Environmental Intelligence &amp; AI Risk Detection
          </p>
          <div style="height:2px;background:#3ecb8a;
                      width:60px;border-radius:2px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_nav(page: str) -> None:
    """Render top navigation buttons and a divider."""
    c1, c2, _ = st.columns([1, 1, 5])
    with c1:
        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        if st.button(
            PAGE_MAP,
            use_container_width=True,
            type="primary" if page == PAGE_MAP else "secondary",
        ):
            st.session_state[PAGE_KEY] = PAGE_MAP
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        if st.button(
            PAGE_AI,
            use_container_width=True,
            type="primary" if page == PAGE_AI else "secondary",
        ):
            st.session_state[PAGE_KEY] = PAGE_AI
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.divider()


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Project Okavango",
        page_icon="O",
        layout="wide",
    )
    _inject_css()
    if PAGE_KEY not in st.session_state:
        st.session_state[PAGE_KEY] = PAGE_MAP
    _render_header()
    _render_nav(st.session_state[PAGE_KEY])
    if st.session_state[PAGE_KEY] == PAGE_AI:
        render_page2()
    else:
        render_page1()

if __name__ == "__main__":
    main()
