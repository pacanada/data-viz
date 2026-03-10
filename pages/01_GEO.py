import json
import streamlit as st
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

ROOT = Path(__file__).parent
TEMPLATES = ROOT / "templates"

st.set_page_config(layout="wide")


data = st.session_state.get("data")  # or load from file / API
numeric_columns = st.session_state.get("numeric_columns")

if data is None or numeric_columns is None:
    st.info("Load data into session_state (or replace with your loader).")
    st.stop()

env = Environment(loader=FileSystemLoader(str(TEMPLATES)))
template = env.get_template("geo.html")

selected_timestamps_column = st.selectbox(
    "Timestamp column",
    options=data.columns,
    index = data.columns.get_loc("timestamp") if "timestamp" in data.columns else 0
)
selected_lat_column = st.selectbox(
    "Latitude column",
    options=data.columns,
    index = data.columns.get_loc("latitude") if "latitude" in data.columns else 0
)
selected_lng_column = st.selectbox(
    "Longitude column",
    options=data.columns,
    index = data.columns.get_loc("longitude") if "longitude" in data.columns else 0
)

selected_numeric = st.multiselect(
    "Metrics to visualize",
    numeric_columns,
    #default=numeric_columns[:3]   # sensible default
)

data[selected_timestamps_column] = data[selected_timestamps_column].astype(str)
data_dict = data.rename(columns={
    selected_timestamps_column: "timestamp",
    selected_lat_column: "lat",
    selected_lng_column: "lng",
})[["timestamp", "lat", "lng"] + selected_numeric].to_dict(orient="records")
#st.json(data_dict)  # for debugging
rendered = template.render(
    data_json=data_dict,           # IMPORTANT: pass JSON string
    numeric_columns_json=selected_numeric,  # Pass numeric columns as JSON string
    title="Voyage Simulation",
)

st.components.v1.html(rendered, height=900, scrolling=True)