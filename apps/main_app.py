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

PAGE_MAP = "🌍 World Map"
PAGE_AI = "🤖 AI Workflow"

P2_LAT_KEY = "p2_latitude"
P2_LON_KEY = "p2_longitude"
P2_ZOOM_KEY = "p2_zoom"
P2_SIZE_KEY = "p2_image_size"


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
    fig.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0}, height=DEFAULT_MAP_HEIGHT)
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


def render_top_bottom_charts(
    df: pd.DataFrame, country_col: str, metric_col: str
) -> None:
    """Render Top 5 and Bottom 5 horizontal bar charts."""
    st.markdown("### Top 5 and Bottom 5 countries (within current filters)")

    top_5, bottom_5 = compute_top_bottom(df, country_col, metric_col, n=5)
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
        fig_top.update_layout(
            height=DEFAULT_BAR_HEIGHT, margin={"l": 0, "r": 0, "t": 10, "b": 0}
        )
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
        fig_bottom.update_layout(
            height=DEFAULT_BAR_HEIGHT, margin={"l": 0, "r": 0, "t": 10, "b": 0}
        )
        st.plotly_chart(fig_bottom, use_container_width=True)


def _update_country_selection(selection_event) -> None:
    """Update the selected country in session state from a Plotly map event."""
    if selection_is_empty(selection_event):
        st.session_state[COUNTRY_STATE_KEY] = None
    else:
        clicked_iso3 = get_selection_iso3(selection_event)
        if clicked_iso3:
            st.session_state[COUNTRY_STATE_KEY] = clicked_iso3


def _render_country_or_charts(
    gdf_year: pd.DataFrame, country_col: str, metric_col: str
) -> None:
    """Show per-country KPIs if one is selected, otherwise show Top/Bottom charts."""
    selected_iso3 = st.session_state.get(COUNTRY_STATE_KEY)
    if selected_iso3:
        country_kpis = compute_country_kpis(gdf_year, country_col, metric_col, selected_iso3)
        if country_kpis is None:
            st.warning("Selected country has no data under current filters.")
            return
        render_selected_country(country_kpis)
        return
    render_top_bottom_charts(gdf_year, country_col, metric_col)


def render_page2() -> None:
    """Render Page 2: coordinate selection for the AI satellite-imagery workflow."""
    st.title("🤖 AI Workflow: Environmental Risk Detection")
    st.markdown(
        "Select an area of interest. "
        "The app will fetch satellite imagery and analyse environmental risk."
    )

    st.subheader("Coordinates & Zoom")
    col1, col2, col3 = st.columns(3)
    with col1:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=0.0001,
            format="%.4f",
            help="Decimal degrees — negative values are South.",
        )
    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=0.0,
            step=0.0001,
            format="%.4f",
            help="Decimal degrees — negative values are West.",
        )
    with col3:
        zoom = st.slider(
            "Zoom level",
            min_value=1,
            max_value=18,
            value=12,
            help="Higher zoom = more detail, smaller area covered.",
        )

    image_size = st.select_slider(
        "Image resolution",
        options=["256 px", "512 px", "1024 px"],
        value="512 px",
        help="Width and height of the satellite image to download.",
    )

    st.subheader("Location Preview")
    preview_df = pd.DataFrame(
        {"lat": [latitude], "lon": [longitude], "label": ["Selected area"]}
    )
    fig_preview = px.scatter_geo(
        preview_df,
        lat="lat",
        lon="lon",
        hover_name="label",
        projection="natural earth",
    )
    fig_preview.update_traces(marker={"size": 15, "color": "red"})
    fig_preview.update_layout(
        height=350, margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )
    st.plotly_chart(fig_preview, use_container_width=True)

    st.button("🔍 Analyse Area", type="primary")

    st.session_state[P2_LAT_KEY] = latitude
    st.session_state[P2_LON_KEY] = longitude
    st.session_state[P2_ZOOM_KEY] = zoom
    st.session_state[P2_SIZE_KEY] = image_size


def render_page1() -> None:
    """Render Page 1: interactive world-map dashboard."""
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

    set_kpis = compute_set_kpis(gdf_year, metric_col)
    if set_kpis is None:
        st.error("No KPI data available for current filters.")
        st.stop()

    render_set_kpis(set_kpis)
    _render_country_or_charts(gdf_year, country_col, metric_col)


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Project Okavango", layout="wide")
    page = st.sidebar.radio("Navigation", [PAGE_MAP, PAGE_AI])
    if page == PAGE_AI:
        render_page2()
    else:
        render_page1()

if __name__ == "__main__":
    main()
