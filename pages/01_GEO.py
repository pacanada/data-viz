import json
import streamlit as st
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

ROOT = Path(__file__).parent
TEMPLATES = ROOT / "templates"

st.set_page_config(layout="wide")

# Use filtered data if available (from the main explorer page), otherwise fall back to raw data.
if "filtered_data" in st.session_state and st.session_state["filtered_data"] is not None:
    data = st.session_state["filtered_data"]
else:
    data = st.session_state.get("data")

numeric_columns = st.session_state.get("numeric_columns", [])

# Try to infer categorical columns if not already stored
categorical_columns = st.session_state.get("categorical_columns")
if categorical_columns is None and data is not None:
    categorical_columns = [
        c
        for c in data.columns
        if pd.api.types.is_bool_dtype(data[c])
        or pd.api.types.is_categorical_dtype(data[c])
        or (data[c].dtype == object and data[c].nunique(dropna=True) <= 200)
    ]

if data is None or numeric_columns is None:
    st.info("Load data into session_state (or replace with your loader).")
    st.stop()

env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
template = env.get_template("geo.html")

# Persist geo UI choices
stored_ts = st.session_state.get("geo_timestamp_column")
stored_lat = st.session_state.get("geo_lat_column")
stored_lng = st.session_state.get("geo_lng_column")

def valid_default(col_name, options, fallback):
    if col_name in options:
        return col_name
    if fallback in options:
        return fallback
    return options[0]

# Ensure the session state has a valid starting value before creating widgets
st.session_state.setdefault(
    "geo_timestamp_column", valid_default(stored_ts, data.columns, "timestamp")
)
st.session_state.setdefault(
    "geo_lat_column", valid_default(stored_lat, data.columns, "latitude")
)
st.session_state.setdefault(
    "geo_lng_column", valid_default(stored_lng, data.columns, "longitude")
)

selected_timestamps_column = st.selectbox(
    "Timestamp column",
    options=data.columns,
    key="geo_timestamp_column",
)
selected_lat_column = st.selectbox(
    "Latitude column",
    options=data.columns,
    key="geo_lat_column",
)
selected_lng_column = st.selectbox(
    "Longitude column",
    options=data.columns,
    key="geo_lng_column",
)

stored_selected_numeric = st.session_state.get("geo_selected_numeric")
if stored_selected_numeric is None:
    st.session_state["geo_selected_numeric"] = []

selected_numeric = st.multiselect(
    "Metrics to visualize",
    numeric_columns,
    key="geo_selected_numeric",
    #default=numeric_columns[:3]   # sensible default
)
# Ensure the selected list is unique to avoid duplicate columns in downstream DataFrames
selected_numeric = list(dict.fromkeys(selected_numeric))

if "geo_display_mode" not in st.session_state:
    st.session_state["geo_display_mode"] = "Polyline"

display_mode = st.selectbox(
    "Map display",
    ["Polyline", "Points"],
    key="geo_display_mode",
)

if "geo_color_mode" not in st.session_state:
    st.session_state["geo_color_mode"] = "None"

color_mode = st.selectbox(
    "Color polyline by",
    ["None", "Numeric", "Categorical"],
    key="geo_color_mode",
)

color_column = None
stored_color_column = st.session_state.get("geo_color_column")
if color_mode == "Numeric" and numeric_columns:
    opts = numeric_columns
    default_col = stored_color_column if stored_color_column in opts else None
    color_column = st.selectbox(
        "Numeric column",
        opts,
        index=opts.index(default_col) if default_col in opts else 0,
        key="geo_color_column",
    )
elif color_mode == "Categorical" and categorical_columns:
    opts = categorical_columns
    default_col = stored_color_column if stored_color_column in opts else None
    color_column = st.selectbox(
        "Categorical column",
        opts,
        index=opts.index(default_col) if default_col in opts else 0,
        key="geo_color_column",
    )


color_config = {"mode": "none"}
if color_column:
    if color_mode == "Numeric":
        # Compute min/max for numeric scaling
        series = pd.to_numeric(data[color_column], errors="coerce")
        color_config = {
            "mode": "numeric",
            "column": color_column,
            "min": float(series.min(skipna=True)),
            "max": float(series.max(skipna=True)),
        }
    else:
        cats = data[color_column].astype(str).fillna("")
        color_config = {
            "mode": "categorical",
            "column": color_column,
            "categories": sorted(cats.dropna().unique().tolist()),
        }

cols = [selected_timestamps_column, selected_lat_column, selected_lng_column] + selected_numeric
if color_column and color_column not in cols:
    cols.append(color_column)

data_df = data[cols].copy()
# Ensure timestamp type is JSON serializable
if selected_timestamps_column in data_df.columns:
    data_df[selected_timestamps_column] = data_df[selected_timestamps_column].astype(str)

if color_column:
    data_df["color_val"] = data_df[color_column]
else:
    data_df["color_val"] = None

# Prepare data for the map template
data_dict = data_df.rename(columns={
    selected_timestamps_column: "timestamp",
    selected_lat_column: "lat",
    selected_lng_column: "lng",
})[["timestamp", "lat", "lng", "color_val"] + selected_numeric].to_dict(orient="records")

rendered = template.render(
    data_json=data_dict,           # IMPORTANT: pass JSON string
    numeric_columns_json=selected_numeric,  # Pass numeric columns as JSON string
    color_config_json=color_config,
    display_mode=display_mode,
    title="Voyage Simulation",
)

st.components.v1.html(rendered, height=900, scrolling=True)