import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Optional

st.set_page_config(page_title="Dataset Explorer", layout="wide")
#DEFAULT_FILENAME = os.getenv("FILENAME", None)  # or .parquet, .json, etc.
# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    if name.endswith(".json"):
        return pd.read_json(file)
    raise ValueError("Unsupported file type. Use CSV, Parquet, or JSON.")

def infer_datetime_cols(df: pd.DataFrame) -> List[str]:
    dt_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_cols.append(c)
        elif df[c].dtype == object:
            # Try parse a sample
            sample = df[c].dropna().head(200)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            if parsed.notna().mean() > 0.8:
                dt_cols.append(c)
    return dt_cols

def coerce_datetimes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df

def numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def categorical_cols(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        dtype = df[c].dtype

        if pd.api.types.is_bool_dtype(dtype):
            out.append(c)

        elif isinstance(dtype, pd.CategoricalDtype):
            out.append(c)

        elif isinstance(dtype, object):
            nunique = df[c].nunique(dropna=True)
            if nunique <= 200:
                out.append(c)

    return out

def text_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == object and df[c].nunique(dropna=True) > 200]

def apply_filters(df: pd.DataFrame, filters_state: dict) -> pd.DataFrame:
    out = df
    # Numeric ranges
    for col, (lo, hi) in filters_state.get("num_ranges", {}).items():
        out = out[out[col].between(lo, hi)]
    # Categorical selections
    for col, selected in filters_state.get("cat_selected", {}).items():
        if selected is not None and len(selected) > 0:
            out = out[out[col].isin(selected)]
    # Datetime range
    for col, (start, end) in filters_state.get("dt_ranges", {}).items():
        if start is not None:
            out = out[out[col] >= pd.to_datetime(start)]
        if end is not None:
            out = out[out[col] <= pd.to_datetime(end)]
    # Text contains
    for col, pattern in filters_state.get("text_contains", {}).items():
        if pattern:
            out = out[out[col].astype(str).str.contains(pattern, case=False, na=False)]
    # Null handling
    null_mode = filters_state.get("null_mode", "keep")
    null_cols = filters_state.get("null_cols", [])
    if null_mode == "drop_rows_any" and null_cols:
        out = out.dropna(subset=null_cols, how="any")
    elif null_mode == "drop_rows_all" and null_cols:
        out = out.dropna(subset=null_cols, how="all")
    return out

def null_mask_fig(df: pd.DataFrame, max_rows: int = 800) -> go.Figure:
    # Downsample for speed/legibility
    if len(df) > max_rows:
        df_view = df.sample(max_rows, random_state=42)
    else:
        df_view = df

    mask = df_view.isna().astype(int)  # 1 = null
    fig = px.imshow(
        mask.T,
        aspect="auto",
        labels=dict(x="Row (sampled)", y="Column", color="Null"),
    )
    fig.update_layout(height=min(700, 30 + 18 * len(df.columns)))
    return fig

def corr_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] < 2:
        return None
    corr = nums.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=False, aspect="auto", labels=dict(color="corr"))
    fig.update_layout(height=600)
    return fig

# ---------------------------
# Sidebar: Data Loading
# ---------------------------
st.title("🔎 Dataset Explorer (interactive + zoomable)")

with st.sidebar:
    st.header("1) Load data")
    uploaded = st.file_uploader("Upload CSV / Parquet / JSON", type=["csv", "parquet", "json"], )
    demo = st.selectbox("…or load a demo dataset", ["(none)", "Gapminder", "Iris", "Tips"])

    df = None
    if uploaded is not None:
        try:
            df = load_data(uploaded)
            st.session_state["data"] = df
        except Exception as e:
            st.error(f"Could not load file: {e}")
    elif demo != "(none)":
        if demo == "Gapminder":
            df = px.data.gapminder()
        elif demo == "Iris":
            df = px.data.iris()
        elif demo == "Tips":
            df = px.data.tips()

    st.divider()
    st.header("2) Settings")
    max_rows = st.slider("Max rows to use (downsample for speed)", 1_000, 300_000, 50_000, step=1_000)
    seed = st.number_input("Downsample seed", min_value=0, value=42, step=1)
    downsample = st.checkbox("Downsample if dataset is larger than max rows", value=True)

if df is None:
    st.info("Upload a dataset (CSV/Parquet/JSON) or select a demo dataset from the sidebar.")
    st.stop()

# Downsample (optional)
if downsample and len(df) > max_rows:
    df = df.sample(max_rows, random_state=int(seed)).reset_index(drop=True)

# Datetime detection + coercion toggles
dt_candidates = infer_datetime_cols(df)
with st.sidebar:
    st.header("3) Datetime parsing")
    dt_cols_selected = st.multiselect(
        "Columns to parse as datetime",
        options=list(df.columns),
        default=dt_candidates,
    )
df = coerce_datetimes(df, dt_cols_selected)

# ---------------------------
# Quick overview
# ---------------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Preview")
    st.caption(f"Rows: {len(df):,} • Columns: {df.shape[1]:,}")
    st.dataframe(df.head(200), width="stretch")

with right:
    st.subheader("Column summary")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "nunique": df.nunique(dropna=True),
    }).sort_values(["nulls", "nunique"], ascending=[False, False])
    st.dataframe(summary, width="stretch", height=420)

# ---------------------------
# Filtering
# ---------------------------
st.divider()
st.subheader("Filters (on-demand)")

num_cols = numeric_cols(df)
st.session_state["numeric_columns"] = num_cols
cat_cols = categorical_cols(df)
dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
txt_cols = text_cols(df)

filters_state = {"num_ranges": {}, "cat_selected": {}, "dt_ranges": {}, "text_contains": {}, "null_mode": "keep", "null_cols": []}

f1, f2, f3, f4 = st.columns(4, gap="large")

with f1:
    st.markdown("**Numeric**")
    for col in st.multiselect("Pick numeric columns", num_cols):
        s = df[col].dropna()
        if len(s) == 0:
            continue

        lo0, hi0 = float(s.min()), float(s.max())

        # Handle constant columns safely
        if lo0 == hi0:
            st.info(f"{col} has a constant value: {lo0}")
            filters_state["num_ranges"][col] = (lo0, hi0)
        else:
            lo, hi = st.slider(col, lo0, hi0, (lo0, hi0))
            filters_state["num_ranges"][col] = (lo, hi)

with f2:
    st.markdown("**Categorical**")
    for col in st.multiselect("Pick categorical columns", cat_cols):
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = sorted(vals)[:5000]
        chosen = st.multiselect(col, vals, default=[])
        if chosen:
            filters_state["cat_selected"][col] = chosen

with f3:
    st.markdown("**Datetime**")
    for col in st.multiselect("Pick datetime columns", dt_cols, default=dt_cols[:1]):
        s = df[col].dropna()
        if len(s) == 0:
            continue
        start0, end0 = s.min(), s.max()
        start = st.date_input(f"{col} start", value=start0.date())
        end = st.date_input(f"{col} end", value=end0.date())
        filters_state["dt_ranges"][col] = (start, end)

with f4:
    st.markdown("**Text search**")
    for col in st.multiselect("Pick text columns", txt_cols, default=txt_cols[:1]):
        pattern = st.text_input(f"{col} contains", value="")
        if pattern:
            filters_state["text_contains"][col] = pattern

with st.expander("Null handling", expanded=False):
    null_cols_pick = st.multiselect("Columns to consider for null rules", options=list(df.columns), default=[])
    null_mode = st.selectbox(
        "Null rule",
        ["keep", "drop_rows_any", "drop_rows_all"],
        format_func=lambda x: {
            "keep": "Keep rows (no null filtering)",
            "drop_rows_any": "Drop rows with ANY null in selected columns",
            "drop_rows_all": "Drop rows with ALL null in selected columns",
        }[x],
    )
    filters_state["null_cols"] = null_cols_pick
    filters_state["null_mode"] = null_mode

df_f = apply_filters(df, filters_state)

st.caption(f"Filtered rows: {len(df_f):,} / {len(df):,}")
st.dataframe(df_f.head(200), width="stretch")

# ---------------------------
# Visualizations
# ---------------------------
st.divider()
st.subheader("Visualizations (zoom + pan + hover)")

viz_tabs = st.tabs([
    "Histogram",
    "Timeseries",
    "Double axis",
    "Scatter",
    "Box/Violin",
    "Correlation",
    "Null mask",
])

# Histogram
with viz_tabs[0]:
    c1, c2, c3 = st.columns([2, 1, 1], gap="large")
    with c1:
        x = st.selectbox("X (numeric)", options=num_cols or list(df_f.columns))
    with c2:
        bins = st.slider("Bins", 5, 200, 40)
    with c3:
        color = st.selectbox("Color (optional)", options=["(none)"] + cat_cols)
    if x:
        fig = px.histogram(
            df_f,
            x=x,
            nbins=bins,
            color=None if color == "(none)" else color,
            marginal="box",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")

# Timeseries
with viz_tabs[1]:
    if not dt_cols:
        st.info("No datetime columns detected/parsed. Pick one in the sidebar (Datetime parsing).")
    else:
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1], gap="large")
        with c1:
            t = st.selectbox("Time column", options=dt_cols)
        with c2:
            y = st.multiselect("Y (numeric)", options=num_cols, default=num_cols[:1])
        with c3:
            agg = st.selectbox("Aggregation", ["none", "mean", "sum", "min", "max", "median"])
        with c4:
            freq = st.selectbox("Resample freq", ["(none)", "D", "W", "M", "Q", "Y"])
        group = st.selectbox("Group by (optional)", options=["(none)"] + cat_cols)

        d = df_f.dropna(subset=[t]).sort_values(t)
        if agg != "none" or freq != "(none)" or group != "(none)":
            keys = [t] + ([] if group == "(none)" else [group])
            d = d[keys + y].copy()

            if freq != "(none)":
                d["_tbin"] = d[t].dt.to_period(freq).dt.to_timestamp()
                t_use = "_tbin"
            else:
                t_use = t

            if group != "(none)":
                gb = d.groupby([t_use, group], dropna=False)
            else:
                gb = d.groupby([t_use], dropna=False)

            if agg == "none":
                # default to mean if user is binning/grouping
                agg_use = "mean"
            else:
                agg_use = agg

            d = gb[y].agg(agg_use).reset_index()
        else:
            t_use = t

        fig = go.Figure()
        if not y:
            st.warning("Pick at least one Y column.")
        else:
            if group == "(none)":
                for yy in y:
                    fig.add_trace(go.Scatter(x=d[t_use], y=d[yy], mode="lines", name=yy))
            else:
                # multiple lines per group per y (can get busy quickly)
                max_groups = 12
                groups = d[group].astype(str).unique().tolist()[:max_groups]
                d2 = d[d[group].astype(str).isin(groups)]
                for yy in y:
                    for g in groups:
                        dd = d2[d2[group].astype(str) == g]
                        fig.add_trace(go.Scatter(x=dd[t_use], y=dd[yy], mode="lines", name=f"{yy} • {g}"))

        fig.update_layout(height=560, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

# Double axis
with viz_tabs[2]:
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1], gap="large")
    with c1:
        x = st.selectbox("X", options=dt_cols + cat_cols + num_cols if (dt_cols + cat_cols + num_cols) else list(df_f.columns))
    with c2:
        y1 = st.selectbox("Y1 (left axis)", options=num_cols)
    with c3:
        y2 = st.selectbox("Y2 (right axis)", options=[c for c in num_cols if c != y1] or num_cols)
    with c4:
        mode = st.selectbox("Style", ["lines", "markers", "lines+markers"])

    if x and y1 and y2:
        d = df_f[[x, y1, y2]].dropna(subset=[x]).copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d[x], y=d[y1], name=y1, mode=mode, yaxis="y1"))
        fig.add_trace(go.Scatter(x=d[x], y=d[y2], name=y2, mode=mode, yaxis="y2"))

        fig.update_layout(
            height=560,
            xaxis=dict(title=x),
            yaxis=dict(title=y1),
            yaxis2=dict(title=y2, overlaying="y", side="right"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

# Scatter
with viz_tabs[3]:
    c1, c2, c3 = st.columns([2, 2, 2], gap="large")
    with c1:
        x = st.selectbox("X (numeric)", options=num_cols, key="sc_x")
    with c2:
        y = st.selectbox("Y (numeric)", options=[c for c in num_cols if c != x] or num_cols, key="sc_y")
    with c3:
        color = st.selectbox("Color (optional)", options=["(none)"] + cat_cols, key="sc_c")
    size = st.selectbox("Size (optional)", options=["(none)"] + num_cols, key="sc_s")
    def safe_hover_cols(df: pd.DataFrame, exclude: set, k: int = 10) -> list[str]:
        cols = []
        for c in df.columns:
            if c in exclude:
                continue
            s = df[c]
            if (
                pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_bool_dtype(s)
                #or pd.api.types.is_datetime64_any_dtype(s)
                or (s.dtype == object and s.dropna().astype(str).map(len).mean() < 80)
            ):
                cols.append(c)
            if len(cols) >= k:
                break
        return cols

    hover_cols = safe_hover_cols(df_f, exclude={x, y}, k=10)
    hover_data = {c: True for c in hover_cols}

    fig = px.scatter(
        df_f,
        x=x,
        y=y,
        color=None if color == "(none)" else color,
        size=None if size == "(none)" else size,
        hover_data=hover_data,
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")

# Box/Violin
with viz_tabs[4]:
    c1, c2, c3 = st.columns([2, 2, 2], gap="large")
    with c1:
        y = st.selectbox("Y (numeric)", options=num_cols, key="bx_y")
    with c2:
        x = st.selectbox("X (category, optional)", options=["(none)"] + cat_cols, key="bx_x")
    with c3:
        kind = st.selectbox("Plot", ["box", "violin"])
    if kind == "box":
        fig = px.box(df_f, x=None if x == "(none)" else x, y=y, points="outliers")
    else:
        fig = px.violin(df_f, x=None if x == "(none)" else x, y=y, box=True, points="outliers")
    fig.update_layout(height=560)
    st.plotly_chart(fig, width="stretch")

# Correlation
with viz_tabs[5]:
    fig = corr_heatmap(df_f)
    if fig is None:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")
    else:
        st.plotly_chart(fig, width="stretch")

# Null mask
with viz_tabs[6]:
    st.caption("Null mask: 1 = null (sampled if many rows). Useful to spot missingness patterns.")
    fig = null_mask_fig(df_f)
    st.plotly_chart(fig, width="stretch")

# ---------------------------
# Download filtered data
# ---------------------------
st.divider()
st.subheader("Export")
c1, c2 = st.columns([1, 2])
with c1:
    fmt = st.selectbox("Format", ["csv", "parquet"])
with c2:
    st.caption("Download the filtered dataset you created via the filters above.")
if fmt == "csv":
    data = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=data, file_name="filtered.csv", mime="text/csv")
else:
    # Parquet bytes
    import io
    buf = io.BytesIO()
    df_f.to_parquet(buf, index=False)
    st.download_button("Download filtered Parquet", data=buf.getvalue(), file_name="filtered.parquet", mime="application/octet-stream")