"""Coordinate converter with automatic source-CRS detection.

Run with:  streamlit run coord_converter_app.py
Needs:     pip install -r requirements.txt
"""
import csv
import io
import re

import numpy as np
import pandas as pd
import pyproj
import streamlit as st
from pyproj import CRS
from pyproj.transformer import TransformerGroup

st.set_page_config(page_title="Coordinate Converter", layout="wide")

# ---------------------------------------------------------------- CRS presets
LO_ZONES = list(range(15, 35, 2))
HART_LO = {z: 2046 + i for i, z in enumerate(LO_ZONES)}
CAPE_LO = dict(zip(LO_ZONES, [22275, 22277, 22279, 22281, 22283,
                              22285, 22287, 22289, 22291, 22293]))
PRESETS = {
    "WGS84 geographic (lat/lon) - 4326": 4326,
    "Web Mercator (Pseudo-Mercator) - 3857": 3857,
    "Hartebeesthoek94 geographic - 4148": 4148,
    "Cape geographic - 4222": 4222,
    "WGS84 / UTM 33S - 32733": 32733,
    "WGS84 / UTM 34S - 32734": 32734,
    "WGS84 / UTM 35S - 32735": 32735,
    "WGS84 / UTM 36S - 32736": 32736,
}
for z, c in HART_LO.items():
    PRESETS[f"Hartebeesthoek94 / Lo{z} - {c}"] = c
for z, c in CAPE_LO.items():
    PRESETS[f"Cape / Lo{z} - {c}"] = c
CUSTOM = "Custom EPSG code..."
PLACEHOLDER = "-- select a coordinate system --"


def names_for(*codes):
    return [n for n, c in PRESETS.items() if c in codes]


def axes_in_xy_order(crs: CRS):
    """Axes in the order pyproj returns them with always_xy=True."""
    ax = list(crs.axis_info)[:2]
    if len(ax) == 2 and ax[0].direction in ("north", "south") \
            and ax[1].direction in ("east", "west"):
        ax = ax[::-1]
    return ax


def axis_label(a):
    return f"{a.abbrev or a.name} ({a.direction})"


def crs_picker(label, key, default_name=None, detected=None, candidates=()):
    """Select a CRS. `detected` pre-selects an auto-detected CRS; `candidates`
    are listed first. With no default, a placeholder forces a manual choice."""
    extra, default = [], default_name
    if detected is not None:
        epsg = detected.to_epsg(min_confidence=70)
        match = next((n for n, c in PRESETS.items() if c == epsg), None)
        if match:
            default = match
        else:
            extra = [f"Detected: {detected.name}"]
            default = extra[0]
    ordered = [n for n in candidates if n in PRESETS] + \
              [n for n in PRESETS if n not in candidates]
    names = extra + ordered + [CUSTOM]
    if default is None:
        names, default = [PLACEHOLDER] + names, PLACEHOLDER
    choice = st.selectbox(label, names, index=names.index(default), key=key)
    try:
        if choice == PLACEHOLDER:
            return None
        if extra and choice == extra[0]:
            crs = detected
        elif choice == CUSTOM:
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
def load_table(raw: bytes, delimiter: str, has_header: bool):
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    head = "\n".join(lines[:25])  # kept (with comments) for CRS detection
    body = "\n".join(l for l in lines if not l.lstrip().startswith(("#", "//")))
    seps = {"Comma": ",", "Semicolon": ";", "Tab": "\t", "Whitespace": r"\s+"}
    if delimiter == "Auto":
        try:
            d = csv.Sniffer().sniff(body[:4096], delimiters=",;\t ").delimiter
            sep = r"\s+" if d == " " else d
        except csv.Error:
            sep = r"\s+"
    else:
        sep = seps[delimiter]
    df = pd.read_csv(io.StringIO(body), sep=sep, engine="python",
                     header=0 if has_header else None)
    if not has_header:
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df, head


def guess_columns(df):
    low = {c.lower(): c for c in df.columns}
    lon = next((low[n] for n in ("lon", "long", "lng", "longitude") if n in low), None)
    lat = next((low[n] for n in ("lat", "latitude") if n in low), None)
    num = list(df.select_dtypes("number").columns) or list(df.columns)
    if lon and lat:
        first, second = lon, lat  # pyproj order for geographic CRSs
    else:
        first, second = num[0], (num[1] if len(num) > 1 else num[0])
    rest = [c for c in num if c not in (first, second)]
    return first, second, (rest[0] if rest else None)


# ---------------------------------------------------------------- CRS detection
def detect_from_metadata(head, columns, prj_text):
    """Explicit CRS information: .prj file, 'EPSG:xxxx' text, or Lo + datum names."""
    if prj_text:
        try:
            return CRS.from_wkt(prj_text), "the uploaded .prj file"
        except Exception:
            pass
    text = head + "\n" + " ".join(columns)
    m = re.search(r"(?:EPSG|SRID)\s*[:=_\-\s]?\s*(\d{4,6})", text, re.I)
    if m:
        try:
            return CRS.from_epsg(int(m.group(1))), "an EPSG code in the file"
        except Exception:
            pass
    low = text.lower()
    z = re.search(r"\blo\s*[-_ ]?\s*(15|17|19|21|23|25|27|29|31|33)\b", low)
    if z:
        zone = int(z.group(1))
        if "cape" in low:
            return CRS.from_epsg(CAPE_LO[zone]), "the Lo zone and datum named in the file"
        if "hartebeesthoek" in low or "hart94" in low:
            return CRS.from_epsg(HART_LO[zone]), "the Lo zone and datum named in the file"
    return None, None


def _range(col):
    v = pd.to_numeric(col, errors="coerce").to_numpy(float)
    v = v[np.isfinite(v)]
    return tuple(np.percentile(v, [1, 99])) if v.size else None


def _family(ra, rb):
    (alo, ahi), (blo, bhi) = ra, rb
    amax, bmax = max(abs(alo), abs(ahi)), max(abs(blo), abs(bhi))
    if amax <= 180 and bmax <= 90:
        return "geo"
    if amax <= 800_000 and blo >= 2_300_000 and bhi <= 4_400_000:
        return "lo"
    if alo >= 100_000 and ahi <= 900_000 and blo >= 5_800_000 and bhi <= 7_800_000:
        return "utm"
    if alo >= 1_500_000 and ahi <= 4_200_000 and blo >= -4_600_000 and bhi <= -2_300_000:
        return "merc_sa"
    if amax <= 20_037_509 and bmax <= 20_048_967 and amax > 1000:
        return "merc"
    return None


# family -> (status, EPSG if a single answer, candidate presets, message)
FAMILIES = {
    "geo": ("guess", 4326, names_for(4326, 4148, 4222),
            "Values look like degrees (lon/lat). Datum can't be told from values, so WGS84 was assumed."),
    "merc_sa": ("guess", 3857, names_for(3857),
                "Values match Web Mercator over southern Africa."),
    "lo": ("partial", None, names_for(*HART_LO.values(), *CAPE_LO.values()),
           "Values look like a South African Lo system (Y westing, X southing), but the zone "
           "and datum (Hartebeesthoek94 or Cape) can't be told from values. Please select it below."),
    "utm": ("partial", None, names_for(32733, 32734, 32735, 32736),
            "Values look like southern-hemisphere UTM in South Africa, but the zone can't be told "
            "from values. Please select it below."),
    "merc": ("partial", None, names_for(3857),
             "Values look like a global metric projection such as Web Mercator. Please confirm below."),
}


def detect_from_values(df, col1, col2):
    ra, rb = _range(df[col1]), _range(df[col2])
    if ra is None or rb is None:
        return dict(status="failed", crs=None, candidates=[],
                    note="The selected columns contain no numeric values.")
    fam = _family(ra, rb)
    if fam:
        status, code, cands, note = FAMILIES[fam]
        return dict(status=status, crs=CRS.from_epsg(code) if code else None,
                    candidates=cands, note=note)
    fam = _family(rb, ra)
    if fam:
        return dict(status="partial", crs=None, candidates=FAMILIES[fam][2],
                    note="The first and second coordinate columns look swapped. Swap them above "
                         "and detection will run again.")
    return dict(status="failed", crs=None, candidates=[],
                note=f"Values ({col1}: {ra[0]:.6g} to {ra[1]:.6g}; {col2}: {rb[0]:.6g} to "
                     f"{rb[1]:.6g}) don't match a known system.")


def detect_source_crs(head, df, col1, col2, prj_text):
    crs, how = detect_from_metadata(head, df.columns, prj_text)
    if crs is not None:
        return dict(status="detected", crs=crs, candidates=[],
                    note=f"Detected from {how}: {crs.name}.")
    return detect_from_values(df, col1, col2)


def show_detection(det):
    s, note = det["status"], det["note"]
    if s == "detected":
        st.success(note)
    elif s == "guess":
        st.warning(note + " Please verify.")
    elif s == "partial":
        st.warning(note)
    else:
        st.warning(note + " Could not detect automatically, please select manually.")


# ---------------------------------------------------------------- UI
st.title("Coordinate Converter")
st.write("Upload a CSV or XYZ file. The app tries to detect its coordinate system; "
         "if it can't, you choose it. Then pick a target system, convert and download.")

with st.sidebar:
    st.header("File options")
    delimiter = st.selectbox("Delimiter", ["Auto", "Comma", "Semicolon", "Tab", "Whitespace"])
    has_header = st.checkbox("First row is a header", value=True,
                             help="Untick for typical headerless .xyz files.")
    prj_file = st.file_uploader("Optional: .prj file with the CRS definition", type=["prj"])
    use_network = st.checkbox("Allow grid downloads (better datum accuracy)", value=False)
    pyproj.network.set_network_enabled(use_network)
prj_text = prj_file.getvalue().decode("utf-8", errors="replace") if prj_file else None

uploaded = st.file_uploader("Input file", type=["csv", "xyz", "txt", "pts", "asc"])
if uploaded is None:
    st.info("Upload a file to get started.")
    st.stop()

sig = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("sig") != sig:
    st.session_state.pop("result", None)
    st.session_state.pop("info", None)
    st.session_state["sig"] = sig

try:
    df, head = load_table(uploaded.getvalue(), delimiter, has_header)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

st.subheader("1. Input data")
st.write(f"{len(df):,} rows, {df.shape[1]} columns")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Columns")
cols = list(df.columns)
g1, g2, gz = guess_columns(df)
k1, k2, k3 = st.columns(3)
col1 = k1.selectbox("First coordinate column", cols, index=cols.index(g1), key=f"c1_{sig}")
col2 = k2.selectbox("Second coordinate column", cols, index=cols.index(g2), key=f"c2_{sig}")
zopts = ["(none)"] + cols
zcol = k3.selectbox("Z column (copied unchanged)", zopts,
                    index=zopts.index(gz) if gz else 0, key=f"cz_{sig}")
keep_original = st.checkbox("Keep all original columns in the output", value=True)

st.subheader("3. Coordinate systems")
det = detect_source_crs(head, df, col1, col2, prj_text)
c_src, c_dst = st.columns(2)
with c_src:
    show_detection(det)
    src = crs_picker("Base (source) coordinate system", f"src_{sig}_{col1}_{col2}",
                     detected=det["crs"], candidates=det["candidates"])
with c_dst:
    dst = crs_picker("Target coordinate system", "dst", default_name="Cape / Lo31 - 22291")

if src is None:
    st.info("Select the source coordinate system to continue.")
    st.stop()
if dst is None:
    st.stop()

# A value-based guess must be confirmed, unless the user picked something else.
need_confirm = det["status"] == "guess" and det["crs"] is not None and src == det["crs"]
confirmed = st.checkbox("I have checked that the source coordinate system is correct") \
    if need_confirm else True

if st.button("Convert", type="primary", disabled=not confirmed):
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
    st.session_state["result"] = pd.concat([base, res], axis=1)
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
    st.download_button("Download converted CSV", csv_bytes,
                       file_name=f"{fname.rsplit('.', 1)[0]}_converted.csv", mime="text/csv")
