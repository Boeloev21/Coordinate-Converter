"""Coordinate converter: upload CSV/XYZ, pick source + target CRS, download result.

Run with:  streamlit run coord_converter_app.py
Needs:     pip install streamlit pandas numpy pyproj
"""
import csv
import io

import numpy as np
import pandas as pd
import pyproj
import streamlit as st
from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup

st.set_page_config(page_title="Coordinate Converter", layout="wide")

# ---------------------------------------------------------------- CRS presets
PRESETS = {
    "WGS84 geographic (lat/lon) - 4326": 4326,
    "Web Mercator (Pseudo-Mercator) - 3857": 3857,
    "Hartebeesthoek94 geographic - 4148": 4148,
    "Cape geographic - 4222": 4222,
    "WGS84 / UTM 34S - 32734": 32734,
    "WGS84 / UTM 35S - 32735": 32735,
    "WGS84 / UTM 36S - 32736": 32736,
}
for i, zone in enumerate(range(15, 35, 2)):
    PRESETS[f"Hartebeesthoek94 / Lo{zone} - {2046 + i}"] = 2046 + i
for zone, code in zip(range(15, 35, 2), [22275, 22277, 22279, 22281, 22283,
                                          22285, 22287, 22289, 22291, 22293]):
    PRESETS[f"Cape / Lo{zone} - {code}"] = code
CUSTOM = "Custom EPSG code..."


def axes_in_xy_order(crs: CRS):
    """Axes in the order pyproj returns them with always_xy=True."""
    ax = list(crs.axis_info)[:2]
    if len(ax) == 2 and ax[0].direction in ("north", "south") \
            and ax[1].direction in ("east", "west"):
        ax = ax[::-1]
    return ax


def axis_label(a):
    return f"{a.abbrev or a.name} ({a.direction})"


def crs_picker(label, key, default_name):
    names = list(PRESETS) + [CUSTOM]
    choice = st.selectbox(label, names, index=names.index(default_name), key=key)
    try:
        if choice == CUSTOM:
            code = st.text_input("EPSG code", value="4326", key=key + "_code")
            crs = CRS.from_user_input(f"EPSG:{code.strip()}")
        else:
            crs = CRS.from_epsg(PRESETS[choice])
    except Exception:
        st.error("Invalid EPSG code.")
        return None
    order = ", ".join(axis_label(a) for a in axes_in_xy_order(crs))
    st.caption(f"{crs.name} | coordinate order: **{order}** | units: "
               f"{crs.axis_info[0].unit_name}")
    return crs


# ---------------------------------------------------------------- file loading
@st.cache_data(show_spinner=False)
def load_table(raw: bytes, delimiter: str, has_header: bool) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="replace")
    seps = {"Comma": ",", "Semicolon": ";", "Tab": "\t", "Whitespace": r"\s+"}
    if delimiter == "Auto":
        try:
            d = csv.Sniffer().sniff(text[:4096], delimiters=",;\t ").delimiter
            sep = r"\s+" if d == " " else d
        except csv.Error:
            sep = r"\s+"
    else:
        sep = seps[delimiter]
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python",
                     header=0 if has_header else None)
    if not has_header:
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------- UI
st.title("Coordinate Converter")
st.write("Upload a CSV or XYZ file, choose the source and target coordinate "
         "systems, convert, and download the result.")

with st.sidebar:
    st.header("File options")
    delimiter = st.selectbox("Delimiter", ["Auto", "Comma", "Semicolon", "Tab", "Whitespace"])
    has_header = st.checkbox("First row is a header", value=True,
                             help="Untick for typical headerless .xyz files.")
    use_network = st.checkbox("Allow grid downloads (better datum accuracy)", value=False)
    pyproj.network.set_network_enabled(use_network)

uploaded = st.file_uploader("Input file", type=["csv", "xyz", "txt", "pts", "asc"])
if uploaded is None:
    st.info("Upload a file to get started.")
    st.stop()

if uploaded.name.lower().endswith(".xyz") and has_header:
    st.caption("Tip: .xyz files usually have no header row. Untick the option in the sidebar if needed.")

try:
    df = load_table(uploaded.getvalue(), delimiter, has_header)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.subheader("1. Input data")
st.write(f"{len(df):,} rows, {df.shape[1]} columns")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Coordinate systems")
c_src, c_dst = st.columns(2)
with c_src:
    src = crs_picker("Base (source) coordinate system", "src",
                     "Web Mercator (Pseudo-Mercator) - 3857")
with c_dst:
    dst = crs_picker("Target coordinate system", "dst", "Cape / Lo31 - 22291")

st.subheader("3. Columns")
cols = list(df.columns)
src_axes = [axis_label(a) for a in axes_in_xy_order(src)] if src else ["First", "Second"]
k1, k2, k3 = st.columns(3)
col1 = k1.selectbox(f"First coordinate: {src_axes[0]}", cols, index=0)
col2 = k2.selectbox(f"Second coordinate: {src_axes[1]}", cols, index=min(1, len(cols) - 1))
zcol = k3.selectbox("Z column (copied unchanged)", ["(none)"] + cols,
                    index=3 if len(cols) > 2 else 0)
keep_original = st.checkbox("Keep all original columns in the output", value=True)

if src is None or dst is None:
    st.stop()

if st.button("Convert", type="primary"):
    a = pd.to_numeric(df[col1], errors="coerce")
    b = pd.to_numeric(df[col2], errors="coerce")
    valid = a.notna() & b.notna()

    tg = TransformerGroup(src, dst, always_xy=True)
    if not tg.transformers:
        st.error("No transformation available between these systems. "
                 "Try enabling grid downloads in the sidebar.")
        st.stop()
    transformer = tg.transformers[0]

    out1, out2 = transformer.transform(a[valid].to_numpy(), b[valid].to_numpy())
    out1, out2 = np.asarray(out1, float), np.asarray(out2, float)
    bad = ~np.isfinite(out1) | ~np.isfinite(out2)
    out1[bad], out2[bad] = np.nan, np.nan

    dst_axes = axes_in_xy_order(dst)
    n1, n2 = f"{dst_axes[0].abbrev or 'C1'}_out", f"{dst_axes[1].abbrev or 'C2'}_out"
    res = pd.DataFrame(np.nan, index=df.index, columns=[n1, n2])
    res.loc[valid, n1], res.loc[valid, n2] = out1, out2

    if keep_original:
        base = df.copy()
    else:
        base = df[[zcol]].copy() if zcol != "(none)" else pd.DataFrame(index=df.index)
    result = pd.concat([base, res], axis=1)

    st.session_state["result"] = result
    st.session_state["info"] = (transformer.description, transformer.accuracy,
                                int((~valid).sum()), int(bad.sum()),
                                dst.is_geographic, uploaded.name)

if "result" in st.session_state:
    result = st.session_state["result"]
    desc, acc, n_invalid, n_failed, is_geo, fname = st.session_state["info"]

    st.subheader("4. Result")
    st.write(f"**Transformation used:** {desc}")
    st.write(f"**Stated accuracy:** {'unknown' if acc is None or acc < 0 else f'{acc} m'}")
    if n_invalid:
        st.warning(f"{n_invalid} row(s) had non-numeric coordinates and were left blank.")
    if n_failed:
        st.warning(f"{n_failed} point(s) could not be transformed and were left blank.")

    st.dataframe(result.head(50), use_container_width=True)

    csv_bytes = result.to_csv(index=False, float_format="%.9f" if is_geo else "%.4f").encode()
    stem = fname.rsplit(".", 1)[0]
    st.download_button("Download converted CSV", csv_bytes,
                       file_name=f"{stem}_converted.csv", mime="text/csv")
