import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import os
import re
import time
import pandas as pd
from pyproj import Transformer
import geopandas as gpd
from folium.plugins import Fullscreen
from folium import MacroElement
from jinja2 import Template

# Streamlit page configuration
st.set_page_config(page_title="Halifax Urban Mobility Data Viewer", layout="wide")

# --- Timing for script execution (for debugging/performance monitoring) ---
script_start = time.time()

# --- Constants for Session State Keys ---
class AppSessionStateKeys:
    SELECTED_JUNCTION_TYPES = 'selected_junction_types'
    SELECTED_TRAFFIC_CONTROL_TYPES = 'selected_traffic_control_types'
    SELECTED_COLLISION_YEARS = 'selected_collision_years'
    SELECTED_COLLISION_CHARACTERISTICS = 'selected_collision_characteristics'
    SELECTED_TRAFFIC_CALMING_ASSET_CODES = 'selected_traffic_calming_asset_codes'
    SELECTED_STREET_LIGHT_USES = 'selected_street_light_uses'
    SELECTED_STREET_LIGHT_MATERIALS = 'selected_street_light_materials'
    SHOW_ALL_SELECTED_FEATURES = 'show_all_selected_features'
    MAP_ZOOM = 'map_zoom'
    MAP_CENTER = 'map_center'
    TOTAL_CHARACTERISTIC_COUNTS = 'total_characteristic_counts'
    # --- Add last rendered keys ---
    LAST_RENDERED_JUNCTION_TYPES = 'last_rendered_junction_types'
    LAST_RENDERED_TRAFFIC_CONTROL_TYPES = 'last_rendered_traffic_control_types'
    LAST_RENDERED_COLLISION_YEARS = 'last_rendered_collision_years'
    LAST_RENDERED_COLLISION_CHARACTERISTICS = 'last_rendered_collision_characteristics'
    LAST_RENDERED_TRAFFIC_CALMING_ASSET_CODES = 'last_rendered_traffic_calming_asset_codes'
    LAST_RENDERED_STREET_LIGHT_USES = 'last_rendered_street_light_uses'
    LAST_RENDERED_STREET_LIGHT_MATERIALS = 'last_rendered_street_light_materials'
    SELECTED_CENTRELINE_BUCKETS = 'selected_centreline_buckets'
    LAST_RENDERED_CENTRELINE_BUCKETS = 'last_rendered_centreline_buckets'
    CENTRELINE_BUCKET_COUNTS = 'centreline_bucket_counts'
    SELECTED_CENTRELINE_ST_CLASS = 'selected_centreline_st_class'
    LAST_RENDERED_CENTRELINE_ST_CLASS = 'last_rendered_centreline_class'
    ACTIVE_BASEMAP = 'active_basemap'

# --- Label dictionaries for UI and popups ---
JUNCTION_TYPE_LABELS = {
    0: "Non‐Intersection Junction",
    1: "Intersection",
    2: "Dead End",
    3: "Ferry Route Connection",
    4: "Outer Boundary Point",
    5: "Boulevard"
}
TRAFFIC_CONTROL_TYPE_LABELS = {
    1: "Intersection",
    6: "Signalized Intersection",
    7: "RA‐5 with Flashing Beacon",
    8: "Overhead Flashing Beacon",
    9: "RA‐5 without Flashing Beacon",
    10: "Rectangular Rapid Flashing Beacon",
    11: "Roundabout",
    12: "Lane Control",
    13: "All Way Stop",
    14: "Pedestrian Half Signals",
    15: "Median Mounted Flashing Beacon"
}
TRAFFIC_CALMING_ASSETCODE_LABELS = {
    "SPDHMP": "Speed Humps",
    "SPDTBL": "Speed Tables",
    "RSDINT": "Raised Intersections",
    "RSDCRW": "Raised Crosswalks",
    "SPDCSH": "Speed Cushions",
    "BMPOUT": "Concrete curb variations: Bump-outs",
    "CTRMED": "Concrete curb variations: Centre Island Medians",
    "CHICAN": "Concrete curb variations: Chicanes",
    "BUSPTF": "Bus Platforms",
    "BUSBMP": "Bus Stop Bump Outs / Bus Bulb",
    "TRFCIR": "Traffic Circle"
}

LIGHTUSE_LABELS = {
    "ROW": "ROW Street Light",
    "PRIV": "Private Light",
    "AREA": "Area Light",
    "FLOOD": "Flood Light",
    "PARK": "Park Light",
    "PARKING": "Parking Lot Light",
    "WALKWAY": "Walkway Light",
    "BOARDWALK": "Boardwalk Light"
}

# --- Collision characteristic filter keys and their column names ---
COLLISION_CHARACTERISTIC_FILTERS = {
    'non_fatal': 'NON_FATAL_',
    'fatal_injury': 'FATAL_INJU',
    'young_driver': 'YOUNG_DEMO',
    'pedestrian_involved': 'PEDESTRIAN',
    'aggressive_driving': 'AGRESSIVE_',
    'distracted_driving': 'DISTRACTED',
    'impaired_driving': 'IMPAIRED_D',
    'bicycle_collision': 'BICYCLE_CO',
    'intersection_related': 'INTERSECTI'
}

# --- Helper function for generating filter controls ---
def generate_filter_control(label_text, label_color, options_list, format_func, 
                            session_state_key_selected_values, multiselect_widget_key, 
                            default_selected_values=None, help_text=None):
    """
    Generates a styled label and a multiselect widget for filtering.
    """
    if default_selected_values is None:
        default_selected_values = []
    
    col1, col2 = st.columns([1.5, 4])
    with col1:
        st.markdown(
            f'<div style="background-color:{label_color}; color:white; padding:6px 6px; border-radius:6px; font-weight:bold; text-align:center; font-size:13px;">{label_text}</div>',
            unsafe_allow_html=True
        )
    with col2:
        selected_values = st.multiselect(
            label_text, # Use the actual label_text for accessibility
            options=options_list,
            default=st.session_state.get(session_state_key_selected_values, default_selected_values),
            format_func=format_func,
            key=multiselect_widget_key,
            label_visibility="collapsed", # Keep hidden as the markdown label is used
            help=help_text
        )
        st.session_state[session_state_key_selected_values] = selected_values

# --- Helper function for adding generic point layers to map ---
def add_generic_point_layer(map_object, data_gdf, layer_name_prefix, color, radius,
                            tooltip_popup_generator_func, unique_filter_tuple_for_name, show_layer=True):
    """
    Adds a generic point-based feature layer to the Folium map.
    """
    if not data_gdf.empty:
        count = len(data_gdf)
        feature_group_name_parts = [layer_name_prefix] + list(map(str, unique_filter_tuple_for_name))
        feature_group_name = "_".join(filter(None, feature_group_name_parts))
        
        fg = folium.FeatureGroup(name=feature_group_name, show=show_layer)
        for _, row in data_gdf.iterrows():
            popup_text = tooltip_popup_generator_func(row)
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                tooltip=popup_text
            ).add_to(fg)
        fg.add_to(map_object)
        return count
    return 0

# --- Data loading and caching functions for all datasets ---
@st.cache_resource(show_spinner=True)
def load_junctions_shapefile():
    gdf = gpd.read_file("Street junctions/Street_Junctions_trimmed.shp")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

@st.cache_resource(show_spinner=True)
def load_traffic_controls_shapefile():
    gdf = gpd.read_file("traffic control locations/Traffic_Control_Locations_trimmed.shp")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

@st.cache_resource(show_spinner=True)
def load_traffic_calming_shapefile():
    gdf = gpd.read_file("Traffic calming infrastructure/Traffic_Calming_Infrastructure_trimmed.shp")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

@st.cache_resource(show_spinner=True)
def load_street_lights_shapefile():
    gdf = gpd.read_file("streetlights/Street_Lights_trimmed.shp")
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    for col in ['LIGHTUSE', 'MAT', 'SETBACK']:
        if col in gdf.columns:
            gdf[col] = gdf[col].replace('', 'UNKN').fillna('UNKN')
    return gdf

@st.cache_resource(show_spinner=True)
def load_centrelines_shapefile():
    gdf = gpd.read_file("street centrelines/Street_Network_trimmed.shp")
    gdf.columns = [col.lower() for col in gdf.columns]
    # Project to UTM zone 20N (EPSG:26920) for accurate length in meters
    gdf_metric = gdf.to_crs(epsg=26920)
    gdf['length_m'] = gdf_metric.length
    # Back to WGS84 for Folium
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    bins = [0, 56, 90, 117, 150, 195, 251, 331, 450, 708, 42587]
    labels = [f"{bins[i]+1}–{bins[i+1]}m" for i in range(10)]
    gdf['length_bucket'] = pd.cut(gdf['length_m'], bins=bins, labels=labels, include_lowest=True, right=True)
    return gdf

# --- Load all datasets globally and cache them ---
load_start = time.time()
junctions_gdf = load_junctions_shapefile()
traffic_controls_gdf = load_traffic_controls_shapefile()
street_lights_gdf = load_street_lights_shapefile()
traffic_calming_gdf = load_traffic_calming_shapefile()
centrelines_gdf = load_centrelines_shapefile()
load_end = time.time()

# --- Collision data: available years and loading by year ---
def get_available_collision_years():
    folder = "traffic_collisions_by_year"
    years = []
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return years
    for fname in os.listdir(folder):
        if fname.startswith("collisions_") and fname.endswith(".shp"):
            try:
                year_str = fname.replace("collisions_", "").replace(".shp", "")
                if year_str.isdigit():
                    years.append(int(year_str))
            except ValueError:
                pass
    years.sort()
    return years

@st.cache_data
def get_all_collision_year_counts():
    counts = {}
    for year_val in get_available_collision_years():
        path = os.path.join("traffic_collisions_by_year", f"collisions_{year_val}.shp")
        if os.path.exists(path):
            try:
                gdf_year = gpd.read_file(path)
                counts[year_val] = len(gdf_year)
            except Exception:
                counts[year_val] = 0
        else:
            counts[year_val] = 0
    return counts

@st.cache_data
def get_all_collision_characteristic_counts():
    """
    Calculates the total count for each collision characteristic across all available years.
    """
    counts = {key: 0 for key in COLLISION_CHARACTERISTIC_FILTERS.keys()}
    all_years = get_available_collision_years()
    if not all_years:
        return counts

    all_dfs = []
    for year in all_years:
        path = os.path.join("traffic_collisions_by_year", f"collisions_{year}.shp")
        if os.path.exists(path):
            try:
                df = gpd.read_file(path)
                all_dfs.append(df)
            except Exception:
                st.warning(f"Could not load or process collision file for year {year}.")
    
    if not all_dfs:
        return counts

    combined_gdf = gpd.GeoDataFrame(pd.concat(all_dfs, ignore_index=True))

    for key, column_name in COLLISION_CHARACTERISTIC_FILTERS.items():
        if column_name in combined_gdf.columns:
            count = combined_gdf[column_name].fillna('N').astype(str).str.upper().isin(['Y', 'YES']).sum()
            counts[key] = int(count)
            
    return counts

# --- Session state initialization for all controls and filters ---
def initialize_session_state():
    defaults = {
        AppSessionStateKeys.SELECTED_JUNCTION_TYPES: [],
        AppSessionStateKeys.SELECTED_TRAFFIC_CONTROL_TYPES: [],
        AppSessionStateKeys.SELECTED_COLLISION_YEARS: [],
        AppSessionStateKeys.SELECTED_COLLISION_CHARACTERISTICS: [],
        AppSessionStateKeys.SELECTED_TRAFFIC_CALMING_ASSET_CODES: [],
        AppSessionStateKeys.SELECTED_STREET_LIGHT_USES: [],
        AppSessionStateKeys.SELECTED_STREET_LIGHT_MATERIALS: [],
        AppSessionStateKeys.SHOW_ALL_SELECTED_FEATURES: False,
        AppSessionStateKeys.MAP_ZOOM: 13,
        AppSessionStateKeys.MAP_CENTER: [44.649605, -63.592300],
        AppSessionStateKeys.TOTAL_CHARACTERISTIC_COUNTS: {},
        # --- Add last rendered keys ---
        AppSessionStateKeys.LAST_RENDERED_JUNCTION_TYPES: [],
        AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CONTROL_TYPES: [],
        AppSessionStateKeys.LAST_RENDERED_COLLISION_YEARS: [],
        AppSessionStateKeys.LAST_RENDERED_COLLISION_CHARACTERISTICS: [],
        AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CALMING_ASSET_CODES: [],
        AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_USES: [],
        AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_MATERIALS: [],
        AppSessionStateKeys.SELECTED_CENTRELINE_BUCKETS: [],
        AppSessionStateKeys.LAST_RENDERED_CENTRELINE_BUCKETS: [],
        AppSessionStateKeys.CENTRELINE_BUCKET_COUNTS: {label: 0 for label in centrelines_gdf['length_bucket'].cat.categories},
        AppSessionStateKeys.SELECTED_CENTRELINE_ST_CLASS: [],
        AppSessionStateKeys.LAST_RENDERED_CENTRELINE_ST_CLASS: [],
        AppSessionStateKeys.ACTIVE_BASEMAP: "OpenStreetMap",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    
    if not st.session_state.get(AppSessionStateKeys.TOTAL_CHARACTERISTIC_COUNTS):
        st.session_state[AppSessionStateKeys.TOTAL_CHARACTERISTIC_COUNTS] = get_all_collision_characteristic_counts()

    for key in COLLISION_CHARACTERISTIC_FILTERS.keys():
        if f"filter_collision_{key}" not in st.session_state:
            st.session_state[f"filter_collision_{key}"] = False
initialize_session_state()

# --- Cached filter functions for each dataset ---
@st.cache_data
def get_filtered_junction_data(selected_junction_types_tuple):
    if not selected_junction_types_tuple:
        return gpd.GeoDataFrame()
    return junctions_gdf[junctions_gdf['JUNCTION_T'].isin(selected_junction_types_tuple)]

@st.cache_data
def get_filtered_traffic_controls_data(selected_traffic_control_types_tuple):
    if not selected_traffic_control_types_tuple or 'CONTROL_TY' not in traffic_controls_gdf.columns:
        return gpd.GeoDataFrame()
    return traffic_controls_gdf[traffic_controls_gdf['CONTROL_TY'].isin(selected_traffic_control_types_tuple)]

@st.cache_data
def get_filtered_traffic_calming_data(selected_asset_codes_tuple):
    if not selected_asset_codes_tuple or 'ASSETCODE' not in traffic_calming_gdf.columns:
        return gpd.GeoDataFrame()
    return traffic_calming_gdf[traffic_calming_gdf['ASSETCODE'].isin(selected_asset_codes_tuple)]

@st.cache_data
def get_filtered_street_lights_data(selected_lightuse_tuple, selected_material_tuple):
    if not selected_lightuse_tuple and not selected_material_tuple:
        return gpd.GeoDataFrame()
    
    data_to_filter = street_lights_gdf

    if selected_lightuse_tuple:
        if 'LIGHTUSE' not in data_to_filter.columns: return gpd.GeoDataFrame()
        data_to_filter = data_to_filter[data_to_filter['LIGHTUSE'].isin(selected_lightuse_tuple)]
    
    if selected_material_tuple:
        if 'MAT' not in data_to_filter.columns: return gpd.GeoDataFrame()
        data_to_filter = data_to_filter[data_to_filter['MAT'].isin(selected_material_tuple)]

    return data_to_filter

@st.cache_data
def get_filtered_traffic_collisions_data(selected_years_tuple, active_boolean_filters):
    current_data_frames = []
    base_data_loaded = False
    if selected_years_tuple:
        for year in selected_years_tuple:
            path = os.path.join("traffic_collisions_by_year", f"collisions_{year}.shp")
            if os.path.exists(path):
                try:
                    df = gpd.read_file(path)
                    if 'Year' not in df.columns and 'ACCIDENT_D' in df.columns:
                        df['Year'] = pd.to_datetime(df['ACCIDENT_D'], errors='coerce').dt.year
                        df.dropna(subset=['Year'], inplace=True)
                        df['Year'] = df['Year'].astype(int)
                    current_data_frames.append(df)
                except Exception:
                    pass
        if current_data_frames:
            base_data_loaded = True
    elif any(active_boolean_filters.values()):
        for year in get_available_collision_years():
            path = os.path.join("traffic_collisions_by_year", f"collisions_{year}.shp")
            if os.path.exists(path):
                try:
                    df = gpd.read_file(path)
                    if 'Year' not in df.columns and 'ACCIDENT_D' in df.columns:
                        df['Year'] = pd.to_datetime(df['ACCIDENT_D'], errors='coerce').dt.year
                        df.dropna(subset=['Year'], inplace=True)
                        df['Year'] = df['Year'].astype(int)
                    current_data_frames.append(df)
                except Exception:
                    pass
        if current_data_frames:
            base_data_loaded = True
    if not base_data_loaded:
        return gpd.GeoDataFrame()
    final_data = gpd.GeoDataFrame(pd.concat(current_data_frames, ignore_index=True)) if current_data_frames else gpd.GeoDataFrame()
    if final_data.empty:
        return gpd.GeoDataFrame()
    if any(active_boolean_filters.values()):
        for filter_key, column_name in COLLISION_CHARACTERISTIC_FILTERS.items():
            if active_boolean_filters.get(filter_key, False):
                if column_name in final_data.columns:
                    condition = final_data[column_name].fillna('N').astype(str).str.upper().isin(['Y', 'YES'])
                    final_data = final_data[condition]
                    if final_data.empty:
                        return gpd.GeoDataFrame()
    return final_data

@st.cache_data
def get_filtered_centrelines_data(selected_buckets_tuple, selected_st_class_tuple):
    if not selected_buckets_tuple and not selected_st_class_tuple:
        return gpd.GeoDataFrame()
    
    # Initial data to filter
    data_to_filter = centrelines_gdf

    # Apply filters if they are provided
    if selected_buckets_tuple:
        if 'length_bucket' not in data_to_filter.columns: return gpd.GeoDataFrame()
        data_to_filter = data_to_filter[data_to_filter['length_bucket'].isin(selected_buckets_tuple)]
    
    if selected_st_class_tuple:
        if 'st_class' not in data_to_filter.columns: return gpd.GeoDataFrame()
        data_to_filter = data_to_filter[data_to_filter['st_class'].isin(selected_st_class_tuple)]

    return data_to_filter

# --- UI: Title and layout columns ---
st.title("Halifax Urban Mobility Data Viewer")
left_col, right_col = st.columns([1, 2])

# --- UI: Controls for all filters (left column) ---
with left_col:
    with st.form(key="filter_form"):
        # --- UI: Filter dropdowns using the helper function ---
        
        # Junctions
        junction_type_counts = junctions_gdf['JUNCTION_T'].value_counts().to_dict()
        all_junction_types = sorted([key for key in JUNCTION_TYPE_LABELS.keys() if junction_type_counts.get(key, 0) > 0])
        def format_junction_label_with_count(x):
            count = junction_type_counts.get(x, 0)
            return f"{x}: {JUNCTION_TYPE_LABELS[x]} ({count})"
        generate_filter_control(
            "Junctions", "#1976d2", all_junction_types, format_junction_label_with_count,
            AppSessionStateKeys.SELECTED_JUNCTION_TYPES, "junction_type_multiselect"
        )

        # Traffic Controls
        if 'CONTROL_TY' in traffic_controls_gdf.columns:
            control_type_counts = traffic_controls_gdf['CONTROL_TY'].value_counts().to_dict()
            all_control_types = sorted([key for key in TRAFFIC_CONTROL_TYPE_LABELS.keys() if control_type_counts.get(key, 0) > 0])
            def format_control_label_with_count(x):
                count = control_type_counts.get(x, 0)
                label = TRAFFIC_CONTROL_TYPE_LABELS.get(x, f"Unknown Type {x}")
                return f"{label} ({count})"
            generate_filter_control(
                "Traffic Controls", "#d32f2f", all_control_types, format_control_label_with_count,
                AppSessionStateKeys.SELECTED_TRAFFIC_CONTROL_TYPES, "traffic_control_type_multiselect"
            )
        else:
            st.warning("Column 'CONTROL_TY' not found in traffic controls data. Cannot display multiselect.")

        # Collisions (year)
        available_collision_years = get_available_collision_years()
        collision_year_counts = get_all_collision_year_counts() if available_collision_years else {}
        def format_year_label_with_count(year_val):
            count = collision_year_counts.get(year_val, 0)
            return f"{year_val} ({count})"
        if available_collision_years:
            generate_filter_control(
                "Collisions (year)", "#ff9800", available_collision_years, format_year_label_with_count,
                AppSessionStateKeys.SELECTED_COLLISION_YEARS, "traffic_collision_year_multiselect"
            )
        else:
            st.warning("No collision data files found in 'traffic_collisions_by_year/'. Year selection is unavailable.")
            st.session_state[AppSessionStateKeys.SELECTED_COLLISION_YEARS] = []
            
        # Collisions (type)
        # Counts for these are now pre-calculated and stored in session state
        total_characteristic_counts = st.session_state.get(AppSessionStateKeys.TOTAL_CHARACTERISTIC_COUNTS, {})
        def format_characteristic_label_with_count(key):
            label = key.replace('_', ' ').title()
            count = total_characteristic_counts.get(key, 0) # Use pre-calculated total counts
            return f"{label} ({count})"
        generate_filter_control(
            "Collisions (type)", "#ff9800", list(COLLISION_CHARACTERISTIC_FILTERS.keys()), 
            format_characteristic_label_with_count,
            AppSessionStateKeys.SELECTED_COLLISION_CHARACTERISTICS, 
            "collision_characteristic_multiselect",
            default_selected_values=st.session_state.get(AppSessionStateKeys.SELECTED_COLLISION_CHARACTERISTICS, [])
        )

        # Traffic Calming
        if not traffic_calming_gdf.empty and 'ASSETCODE' in traffic_calming_gdf.columns:
            asset_code_counts = traffic_calming_gdf['ASSETCODE'].value_counts().to_dict()
            all_asset_codes = sorted([key for key in TRAFFIC_CALMING_ASSETCODE_LABELS.keys() if asset_code_counts.get(key, 0) > 0])
            def format_asset_code_label_with_count(x):
                count = asset_code_counts.get(x, 0)
                label = TRAFFIC_CALMING_ASSETCODE_LABELS.get(x, f"Unknown Asset Code {x}")
                return f"{label} ({count})"
            generate_filter_control(
                "Traffic Calming", "#008080", all_asset_codes, format_asset_code_label_with_count,
                AppSessionStateKeys.SELECTED_TRAFFIC_CALMING_ASSET_CODES, "traffic_calming_asset_code_multiselect"
            )
        else:
            st.warning("Traffic calming data is not available or 'ASSETCODE' column is missing.")

        # Street Lights (Use)
        if not street_lights_gdf.empty and 'LIGHTUSE' in street_lights_gdf.columns:
            lightuse_counts = street_lights_gdf['LIGHTUSE'].value_counts().to_dict()
            all_lightuse_values = sorted(street_lights_gdf['LIGHTUSE'].unique())
            def format_lightuse_label_with_count(x):
                count = lightuse_counts.get(x, 0)
                label = LIGHTUSE_LABELS.get(x, x)
                return f"{label} ({count})"
            generate_filter_control(
                "Street Lights (Use)", "#DAA520", all_lightuse_values, format_lightuse_label_with_count,
                AppSessionStateKeys.SELECTED_STREET_LIGHT_USES, "street_light_use_multiselect",
                default_selected_values=st.session_state.get(AppSessionStateKeys.SELECTED_STREET_LIGHT_USES, [])
            )
        else:
            st.warning("Street lights data is not available or 'LIGHTUSE' column is missing.")

        # Street Lights (Material)
        if not street_lights_gdf.empty and 'MAT' in street_lights_gdf.columns:
            material_counts = street_lights_gdf['MAT'].value_counts().to_dict()
            all_material_values = sorted(street_lights_gdf['MAT'].unique())
            def format_material_label_with_count(x):
                count = material_counts.get(x, 0)
                return f"{x} ({count})"
            generate_filter_control(
                "Street Lights (Material)", "#DAA520", all_material_values, format_material_label_with_count,
                AppSessionStateKeys.SELECTED_STREET_LIGHT_MATERIALS, "street_light_material_multiselect",
                default_selected_values=st.session_state.get(AppSessionStateKeys.SELECTED_STREET_LIGHT_MATERIALS, [])
            )
        else:
            st.warning("Street lights data is not available or 'MAT' column is missing.")
            
        # Street Centrelines (segment length)
        centreline_bucket_counts = centrelines_gdf['length_bucket'].value_counts().to_dict()
        all_centreline_buckets = [b for b in centrelines_gdf['length_bucket'].cat.categories if centreline_bucket_counts.get(b, 0) > 0]
        def format_centreline_bucket_label_with_count(x):
            count = centreline_bucket_counts.get(x, 0)
            return f"{x} ({count})"
        generate_filter_control(
            "Street Segments (Length)", "#1565c0", all_centreline_buckets, format_centreline_bucket_label_with_count,
            AppSessionStateKeys.SELECTED_CENTRELINE_BUCKETS, "centreline_bucket_multiselect"
        )

        # Street Centrelines (street segmentclassification)
        if 'st_class' in centrelines_gdf.columns:
            st_class_counts = centrelines_gdf['st_class'].value_counts(dropna=True).to_dict()
            all_st_classes = sorted(st_class_counts.keys())
            def format_st_class_label_with_count(x):
                count = st_class_counts.get(x, 0)
                return f"{x} ({count})"
            generate_filter_control(
                "Street Segments (Classification)", "#1565c0", all_st_classes, format_st_class_label_with_count,
                AppSessionStateKeys.SELECTED_CENTRELINE_ST_CLASS, "centreline_stclass_multiselect"
            )
        else:
            st.warning("Column 'st_class' not found in street centrelines data. Class filter is unavailable.")

        submitted = st.form_submit_button("Render")

    # --- Unified Render and Clear Map buttons ---
    if submitted:
        # Copy current selections to last rendered keys
        st.session_state[AppSessionStateKeys.LAST_RENDERED_JUNCTION_TYPES] = list(st.session_state[AppSessionStateKeys.SELECTED_JUNCTION_TYPES])
        st.session_state[AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CONTROL_TYPES] = list(st.session_state[AppSessionStateKeys.SELECTED_TRAFFIC_CONTROL_TYPES])
        st.session_state[AppSessionStateKeys.LAST_RENDERED_COLLISION_YEARS] = list(st.session_state[AppSessionStateKeys.SELECTED_COLLISION_YEARS])
        st.session_state[AppSessionStateKeys.LAST_RENDERED_COLLISION_CHARACTERISTICS] = list(st.session_state[AppSessionStateKeys.SELECTED_COLLISION_CHARACTERISTICS])
        st.session_state[AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CALMING_ASSET_CODES] = list(st.session_state[AppSessionStateKeys.SELECTED_TRAFFIC_CALMING_ASSET_CODES])
        st.session_state[AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_USES] = list(st.session_state[AppSessionStateKeys.SELECTED_STREET_LIGHT_USES])
        st.session_state[AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_MATERIALS] = list(st.session_state[AppSessionStateKeys.SELECTED_STREET_LIGHT_MATERIALS])
        st.session_state[AppSessionStateKeys.LAST_RENDERED_CENTRELINE_BUCKETS] = list(st.session_state[AppSessionStateKeys.SELECTED_CENTRELINE_BUCKETS])
        st.session_state[AppSessionStateKeys.LAST_RENDERED_CENTRELINE_ST_CLASS] = list(st.session_state[AppSessionStateKeys.SELECTED_CENTRELINE_ST_CLASS])
        st.session_state[AppSessionStateKeys.SHOW_ALL_SELECTED_FEATURES] = True
        
        # Update bucket counts (defer until Render)
        st.session_state[AppSessionStateKeys.CENTRELINE_BUCKET_COUNTS] = centrelines_gdf['length_bucket'].value_counts().to_dict()
        st.rerun()

    if st.button("Clear Map"):
        st.session_state[AppSessionStateKeys.SHOW_ALL_SELECTED_FEATURES] = False
        # Clear last rendered keys
        st.session_state[AppSessionStateKeys.LAST_RENDERED_JUNCTION_TYPES] = []
        st.session_state[AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CONTROL_TYPES] = []
        st.session_state[AppSessionStateKeys.LAST_RENDERED_COLLISION_YEARS] = []
        st.session_state[AppSessionStateKeys.LAST_RENDERED_COLLISION_CHARACTERISTICS] = []
        st.session_state[AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CALMING_ASSET_CODES] = []
        st.session_state[AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_USES] = []
        st.session_state[AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_MATERIALS] = []
        st.session_state[AppSessionStateKeys.LAST_RENDERED_CENTRELINE_BUCKETS] = []
        st.session_state[AppSessionStateKeys.LAST_RENDERED_CENTRELINE_ST_CLASS] = []
        st.session_state[AppSessionStateKeys.SELECTED_JUNCTION_TYPES] = []
        st.session_state[AppSessionStateKeys.SELECTED_TRAFFIC_CONTROL_TYPES] = []
        st.session_state[AppSessionStateKeys.SELECTED_COLLISION_YEARS] = []
        st.session_state[AppSessionStateKeys.SELECTED_COLLISION_CHARACTERISTICS] = []
        st.session_state[AppSessionStateKeys.SELECTED_TRAFFIC_CALMING_ASSET_CODES] = []
        st.session_state[AppSessionStateKeys.SELECTED_STREET_LIGHT_USES] = []
        st.session_state[AppSessionStateKeys.SELECTED_STREET_LIGHT_MATERIALS] = []
        st.session_state[AppSessionStateKeys.SELECTED_CENTRELINE_BUCKETS] = []
        st.session_state[AppSessionStateKeys.SELECTED_CENTRELINE_ST_CLASS] = []
        st.session_state[AppSessionStateKeys.CENTRELINE_BUCKET_COUNTS] = {label: 0 for label in centrelines_gdf['length_bucket'].cat.categories}
        st.session_state[AppSessionStateKeys.ACTIVE_BASEMAP] = "OpenStreetMap"
        st.rerun()

# --- UI: Map rendering and statistics (right column) ---
with right_col:
    t0 = time.time()
    map_center = st.session_state.get(AppSessionStateKeys.MAP_CENTER, [44.7, -63.65])
    map_zoom = st.session_state.get(AppSessionStateKeys.MAP_ZOOM, 13)
    
    # --- Add multiple tile layers for selection ---
    basemap_options = {
        "OpenStreetMap": "OpenStreetMap",
        "Light (Positron)": "CartoDB positron",
        "Dark (Dark Matter)": "CartoDB dark_matter",
    }
    # Add reverse mapping from folium tile name to display name
    folium_tile_to_display = {v: k for k, v in basemap_options.items()}

    # Get the last active basemap name from session state to set the default view
    active_basemap_name = st.session_state.get(AppSessionStateKeys.ACTIVE_BASEMAP, "OpenStreetMap")
    # Defensively fallback to OSM if the saved name is somehow invalid
    if active_basemap_name not in basemap_options:
        active_basemap_name = "OpenStreetMap"

    # Initialize map with no default tiles
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles=None, max_bounds=False)

    # Add the active basemap first to set it as the default layer
    folium.TileLayer(
        tiles=basemap_options[active_basemap_name],
        name=active_basemap_name,
        overlay=False,
        control=True,
        show=True
    ).add_to(m)
    
    # Add the other basemaps to the layer control
    for name, tiles in basemap_options.items():
        if name != active_basemap_name:
            folium.TileLayer(
                tiles=tiles,
                name=name,
                overlay=False,
                control=True,
                show=False
            ).add_to(m)

    map_init_time = time.time() - t0
    
    # --- Add map layers using the helper function ---
    filter_start = time.time()
    junctions_count = controls_count = collisions_count = traffic_calming_count = street_lights_count = centrelines_count = 0
    
    show_features = st.session_state.get(AppSessionStateKeys.SHOW_ALL_SELECTED_FEATURES, False)

    # Junctions
    if show_features and st.session_state.get(AppSessionStateKeys.LAST_RENDERED_JUNCTION_TYPES):
        selected_types_tuple = tuple(sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_JUNCTION_TYPES, [])))
        filtered_junctions = get_filtered_junction_data(selected_types_tuple)
        def junction_tooltip_generator(row):
            label_key = row.get('JUNCTION_T')
            label = JUNCTION_TYPE_LABELS.get(label_key, f"Unknown Type {label_key}")
            return f"Junction Type: {label}<br>Lon: {row.geometry.x:.5f}<br>Lat: {row.geometry.y:.5f}"
        junctions_count = add_generic_point_layer(
            m, filtered_junctions, "JunctionsLayer", 'blue', 5, 
            junction_tooltip_generator, selected_types_tuple
        )

    # Traffic Controls
    if show_features and st.session_state.get(AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CONTROL_TYPES):
        selected_types_tuple = tuple(sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CONTROL_TYPES, [])))
        filtered_controls = get_filtered_traffic_controls_data(selected_types_tuple)
        def control_tooltip_generator(row):
            control_type_key = row.get('CONTROL_TY')
            label = TRAFFIC_CONTROL_TYPE_LABELS.get(control_type_key, f"Unknown Type {control_type_key}")
            return f"Control Type: {label}<br>Lon: {row.geometry.x:.5f}<br>Lat: {row.geometry.y:.5f}"
        controls_count = add_generic_point_layer(
            m, filtered_controls, "TrafficControlsLayer", 'red', 4,
            control_tooltip_generator, selected_types_tuple
        )

    # Collisions
    show_collisions_layer = show_features and \
        (st.session_state.get(AppSessionStateKeys.LAST_RENDERED_COLLISION_YEARS) or \
         st.session_state.get(AppSessionStateKeys.LAST_RENDERED_COLLISION_CHARACTERISTICS))
    if show_collisions_layer:
        selected_years_tuple = tuple(sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_COLLISION_YEARS, [])))
        active_boolean_filters = {
            key: (key in st.session_state.get(AppSessionStateKeys.LAST_RENDERED_COLLISION_CHARACTERISTICS, []))
            for key in COLLISION_CHARACTERISTIC_FILTERS.keys()
        }
        filtered_collisions = get_filtered_traffic_collisions_data(selected_years_tuple, active_boolean_filters)
        def collision_tooltip_generator(row):
            # Base info
            tooltip_text = f"Year: {row.get('Year', 'N/A')}<br>Date: {str(row.get('ACCIDENT_D', 'N/A'))[:10]}"
            
            # Characteristics
            char_cols = {
                'NON_FATAL_': 'Non-Fatal', 'FATAL_INJU': 'Fatal/Injury', 'YOUNG_DEMO': 'Young Driver', 
                'PEDESTRIAN': 'Pedestrian', 'AGRESSIVE_': 'Aggressive Driving', 'DISTRACTED': 'Distracted', 
                'IMPAIRED_D': 'Impaired', 'BICYCLE_CO': 'Bicycle', 'INTERSECTI': 'Intersection'
            }
            
            present_chars = []
            for col, label in char_cols.items():
                if str(row.get(col, 'N')).upper() in ['Y', 'YES']:
                    present_chars.append(label)
            
            if present_chars:
                tooltip_text += "<br>" + ", ".join(present_chars)
                
            # Coordinates
            tooltip_text += f"<br>Lon: {row.geometry.x:.5f}<br>Lat: {row.geometry.y:.5f}"
            
            return tooltip_text
        boolean_filter_names = sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_COLLISION_CHARACTERISTICS, []))
        collision_layer_id_tuple = selected_years_tuple + tuple(boolean_filter_names) 
        collisions_count = add_generic_point_layer(
            m, filtered_collisions, "TrafficCollisionsLayer", 'orange', 3,
            collision_tooltip_generator, collision_layer_id_tuple
        )

    # Traffic Calming
    if show_features and st.session_state.get(AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CALMING_ASSET_CODES):
        selected_asset_codes_tuple = tuple(sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_TRAFFIC_CALMING_ASSET_CODES, [])))
        filtered_traffic_calming = get_filtered_traffic_calming_data(selected_asset_codes_tuple)
        def calming_tooltip_generator(row):
            asset_code_key = row.get('ASSETCODE')
            label = TRAFFIC_CALMING_ASSETCODE_LABELS.get(asset_code_key, asset_code_key) # Use key as fallback
            return (f"Type: {label}<br>"
                    f"Install Year: {row.get('INSTYR', 'N/A')}<br>"
                    f"Location: {row.get('LOCATION', 'N/A')}<br>"
                    f"Lon: {row.geometry.x:.5f}<br>"
                    f"Lat: {row.geometry.y:.5f}")
        traffic_calming_count = add_generic_point_layer(
            m, filtered_traffic_calming, "TrafficCalmingLayer", 'teal', 3,
            calming_tooltip_generator, selected_asset_codes_tuple
        )

    # Street Lights
    show_street_lights_layer = show_features and \
        (st.session_state.get(AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_USES) or \
         st.session_state.get(AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_MATERIALS))
    if show_street_lights_layer:
        selected_uses_tuple = tuple(sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_USES, [])))
        selected_materials_tuple = tuple(sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_STREET_LIGHT_MATERIALS, [])))
        filtered_street_lights = get_filtered_street_lights_data(selected_uses_tuple, selected_materials_tuple)
        def streetlight_tooltip_generator(row):
            lightuse_code = row.get('LIGHTUSE', 'N/A')
            lightuse_label = LIGHTUSE_LABELS.get(lightuse_code, lightuse_code)
            return (f"Material: {row.get('MAT', 'N/A')}<br>"
                    f"Use: {lightuse_label}<br>"
                    f"Setback: {row.get('SETBACK', 'N/A')}<br>"
                    f"Install Year: {row.get('INSTYR', 'N/A')}<br>"
                    f"Lon: {row.geometry.x:.5f}<br>"
                    f"Lat: {row.geometry.y:.5f}")
        street_lights_layer_id_tuple = selected_uses_tuple + selected_materials_tuple
        street_lights_count = add_generic_point_layer(
            m, filtered_street_lights, "StreetLightsLayer", '#DAA520', 2.5,
            streetlight_tooltip_generator, street_lights_layer_id_tuple
        )
        
    # Street Centrelines
    show_centrelines_layer = show_features and \
        (st.session_state.get(AppSessionStateKeys.LAST_RENDERED_CENTRELINE_BUCKETS) or \
         st.session_state.get(AppSessionStateKeys.LAST_RENDERED_CENTRELINE_ST_CLASS))
    if show_centrelines_layer:
        selected_buckets_tuple = tuple(sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_CENTRELINE_BUCKETS, [])))
        selected_st_class_tuple = tuple(sorted(st.session_state.get(AppSessionStateKeys.LAST_RENDERED_CENTRELINE_ST_CLASS, [])))
        filtered_centrelines = get_filtered_centrelines_data(selected_buckets_tuple, selected_st_class_tuple)
        if not filtered_centrelines.empty:
            fg = folium.FeatureGroup(name="StreetCentrelinesLayer", show=True)
            for _, row in filtered_centrelines.iterrows():
                if row.geometry and row.geometry.geom_type == 'LineString':
                    folium.PolyLine(
                        locations=[(lat, lon) for lon, lat in row.geometry.coords],
                        color='#444', weight=4, opacity=0.8,
                        tooltip=f"Name: {row.get('full_name', 'N/A')}<br>From: {row.get('from_str', 'N/A')}<br>To: {row.get('to_str', 'N/A')}<br>Class: {row.get('st_class', 'N/A')}<br>Length: {row.get('length_m', 0):.1f}m"
                    ).add_to(fg)
                elif row.geometry and row.geometry.geom_type == 'MultiLineString':
                    for linestring in row.geometry.geoms:
                        folium.PolyLine(
                            locations=[(lat, lon) for lon, lat in linestring.coords],
                            color='#444', weight=4, opacity=0.8,
                            tooltip=f"Name: {row.get('full_name', 'N/A')}<br>From: {row.get('from_str', 'N/A')}<br>To: {row.get('to_str', 'N/A')}<br>Class: {row.get('st_class', 'N/A')}<br>Length: {row.get('length_m', 0):.1f}m"
                        ).add_to(fg)
            fg.add_to(m)
            centrelines_count = len(filtered_centrelines)
        
    filter_end = time.time()

    # Add controls at the end, so they are aware of all layers
    Fullscreen(
        position="topleft",
        title="Fullscreen",
        title_cancel="Exit Fullscreen",
        force_separate_button=False,
    ).add_to(m)
    folium.LayerControl(position='topleft').add_to(m)

    map_data = st_folium(m, width=900, height=600, returned_objects=['last_tile_layer'], key="folium_map")

    # Only persist the user's last selected basemap if available (not None)
    if map_data and map_data.get("last_tile_layer") is not None:
        new_basemap = map_data.get("last_tile_layer")
        # Map folium tile name back to display name if needed
        if new_basemap in folium_tile_to_display:
            new_basemap_display = folium_tile_to_display[new_basemap]
        else:
            new_basemap_display = new_basemap  # fallback, should be display name already
        st.session_state[AppSessionStateKeys.ACTIVE_BASEMAP] = new_basemap_display

    map_render_time = time.time() - filter_end
    total_points = junctions_count + controls_count + collisions_count + traffic_calming_count + street_lights_count + centrelines_count
    stats_lines = [f"**Total data points rendered:** {total_points}"]
    if junctions_count:
        stats_lines.append(f"- Junctions: {junctions_count}")
    if controls_count:
        stats_lines.append(f"- Traffic controls: {controls_count}")
    if collisions_count:
        stats_lines.append(f"- Collisions: {collisions_count}")
    if traffic_calming_count:
        stats_lines.append(f"- Traffic Calming: {traffic_calming_count}")
    if street_lights_count:
        stats_lines.append(f"- Street Lights: {street_lights_count}")
    if centrelines_count:
        stats_lines.append(f"- Street Centrelines: {centrelines_count}")
    st.markdown("\n".join(stats_lines))

    st.markdown(f"**Timing:** Data load: {load_end - load_start:.3f}s | Map init: {map_init_time:.3f}s | Filtering: {filter_end - filter_start:.3f}s | Map render: {map_render_time:.3f}s")

# --- Script execution time (for debugging/performance monitoring) ---
script_end = time.time()
st.write(f"Total script execution time: {script_end - script_start:.3f}s") 