#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import tempfile
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
from ftplib import FTP_TLS

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as pe

import requests


# ======================
# CONFIGURATION
# ======================
TXT_URL = os.environ.get("CURRENTMONTHURL", "").strip()
GEOJSON_SECRET = os.environ.get("GEOJSON_PASS", "").strip()

FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()

PLOT_WIDTH = 12
PLOT_HEIGHT = 8

GRID_N = 300
TOP_N = 30
TOP_FONTSIZE = 7.0

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 15

if not TXT_URL:
    raise RuntimeError("Environment variable CURRENTMONTHURL is not set.")

if not GEOJSON_SECRET:
    raise RuntimeError("Environment variable GEOJSON_PASS is not set.")

if not FTP_HOST or not FTP_USER or not FTP_PASS:
    raise RuntimeError("FTP_HOST / FTP_USER / FTP_PASS environment variables are not all set.")


# EXACT BOXES: same as yearly scripts
REGIONS = [
    {
        "key": "greece",
        "title": "Υπολογ. σωρευτικός υετός στην Ελλάδα (τρέχων μήνας)",
        "outfile": "monthlyppn.png",
        "lon_min": 19.0,  "lon_max": 30.0,
        "lat_min": 34.5,  "lat_max": 42.5
    },
    {
        "key": "attica",
        "title": "Υπολογ. σωρευτικός υετός Αττικής (τρέχων μήνας)",
        "outfile": "monthlyppn_attica.png",
        "lon_min": 22.7,  "lon_max": 25.0,
        "lat_min": 37.5,  "lat_max": 38.7
    },
    {
        "key": "negreece",
        "title": "Υπολογ. σωρευτικός υετός ΒΑ Ελλάδας (τρέχων μήνας)",
        "outfile": "monthlyppn_negreece.png",
        "lon_min": 22.0,  "lon_max": 26.6,
        "lat_min": 39.7,  "lat_max": 41.8,
        "top_n": 15,
        "top_loc": "bottom_right"
    },
    {
        "key": "crete",
        "title": "Υπολογ. σωρευτικός υετός Κρήτης (τρέχων μήνας)",
        "outfile": "monthlyppn_crete.png",
        "lon_min": 23.37, "lon_max": 26.4,
        "lat_min": 34.7,  "lat_max": 35.78,
        "top_n": 12
    },
]


# ======================
# HELPERS
# ======================
ABBREV_RULES = [
    (r"\bΣτρατιωτική\s+Σχολή\s+Ευελπίδων\b", "Στρατ. Σχ. Ευελπίδων"),
    (r"\bΙερά\s+Μονή\b", "Ι.Μ."),
    (r"\bΚαταφύγιο\b", "Καταφ."),
    (r"\bΟροπέδιο\b", "Οροπ."),
    (r"\bΝομισματοκοπείο\b", "Νομισμ."),
    (r"\bΖωολογικό\b", "Ζωολ."),
    (r"\bΠυροσβεστικού Σώματος\b", "Πυροσβ. Σώμ."),
    (r"\bΌρος\b", "Όρ."),
    (r"\bΆνω\b", "Ά."),
    (r"\bΚάτω\b", "Κ."),
    (r"\bΆγιος\b", "Άγ."),
    (r"\bΑγία\b", "Αγ."),
    (r"\bσταθμός\b", "στ."),
    (r"\bλόφος\b", "λόφ."),
]

_abbrev_compiled = [
    (re.compile(pat, flags=re.IGNORECASE), repl)
    for pat, repl in ABBREV_RULES
]


def abbreviate_gr_name(s):
    if not isinstance(s, str):
        return s
    out = s.strip()
    if not out:
        return out
    for rx, repl in _abbrev_compiled:
        out = rx.sub(repl, out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def station_name(row):
    cg = row.get("citygr", None)
    if isinstance(cg, str) and cg.strip():
        return cg.strip(), True
    return str(row.get("webcode", "")).strip(), False


def fmt_mm_gr(x):
    try:
        return f"{float(x):.1f}".replace(".", ",")
    except Exception:
        return ""


def idw_optimized(x, y, z, xi, yi, power=2, k=8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if len(z) == 0:
        return np.full(xi.shape, np.nan, dtype=float)

    tree = cKDTree(np.c_[x, y])
    k_eff = min(k, len(z))

    dist, idx = tree.query(np.c_[xi.ravel(), yi.ravel()], k=k_eff)

    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / (dist ** power)
        weights[dist == 0] = 1e12
        zi = np.sum(weights * z[idx], axis=1) / np.sum(weights, axis=1)

    return zi.reshape(xi.shape)


def bbox_filter(df, lon_min, lon_max, lat_min, lat_max):
    return df[
        (df["longitude"].between(lon_min, lon_max))
        & (df["latitude"].between(lat_min, lat_max))
    ].copy()


def build_top_text(region_df, top_n):
    top_source = region_df.dropna(subset=["total_precipitation_missing"]).copy()
    if top_source.empty:
        return ""

    topn = top_source.nlargest(int(top_n), "total_precipitation_missing")
    lines = []

    for _, r in topn.iterrows():
        name, is_citygr = station_name(r)
        if is_citygr:
            name = abbreviate_gr_name(name)
        mm = fmt_mm_gr(r["total_precipitation_missing"])
        if name and mm:
            lines.append(f"{name} {mm} mm")

    return "\n".join(lines)


def upload_to_ftp(file_buffer, filename):
    try:
        ftps = FTP_TLS()
        ftps.connect(FTP_HOST, 21)
        ftps.login(user=FTP_USER, passwd=FTP_PASS)
        ftps.prot_p()
        file_buffer.seek(0)
        ftps.storbinary("STOR {}".format(filename), file_buffer)
        ftps.quit()
        print("📤 Uploaded: {}".format(filename))
    except Exception as e:
        print("⚠️ FTP upload failed for {}: {}".format(filename, e))


def load_greece_geometry(secret_value):
    """
    GEOJSON_PASS may contain either:
    - a URL
    - a filesystem path

    For GitHub Actions, URL is usually the realistic choice.
    """
    if secret_value.startswith("http://") or secret_value.startswith("https://"):
        response = requests.get(secret_value, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            gdf = gpd.read_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return gdf

    if not os.path.exists(secret_value):
        raise RuntimeError("GEOJSON_PASS does not point to an existing local file.")

    return gpd.read_file(secret_value)


# ======================
# LOAD DATA
# ======================
response = requests.get(TXT_URL, headers=HEADERS, timeout=TIMEOUT)
response.raise_for_status()
response.encoding = "utf-8"
data = pd.read_csv(StringIO(response.text), delimiter="\t")

athens_now = datetime.now(ZoneInfo("Europe/Athens"))
today_day = athens_now.day

# Normalize webcode for robust filtering
w = data["webcode"].astype("string").str.strip().str.casefold()

excluded_exact = {
    "agrivate_stavroupoli", "age_dasosxiromerou",
    "agrivate_rizia", "age_agiosilias", "wu_lefkaditi", "wu_lampeia",
    "wu_karkalou", "hnms3_megara", "wu_varnavas", "wu_sykamino",
    "wu_avlonas", "age_galatas", "ierapetra", "agrivate_messouni",
    "pws_proti2", "age_vrana", "potamoi"
}
excluded_prefixes = ("hcmr_", "uoi_")

if "daysinmonth" not in data.columns:
    raise ValueError("Column 'daysinmonth' is missing from monthly source data.")

data = data[
    (pd.to_numeric(data["daysinmonth"], errors="coerce") == today_day)
    & (~w.isin(excluded_exact))
    & (~w.str.startswith(excluded_prefixes, na=False))
].copy()

# Numeric columns
data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
data["total_precipitation"] = pd.to_numeric(data["total_precipitation"], errors="coerce")

if "total_precipitation_missing" in data.columns:
    data["total_precipitation_missing"] = pd.to_numeric(
        data["total_precipitation_missing"], errors="coerce"
    )
else:
    data["total_precipitation_missing"] = np.nan

# Remove missing coordinates
data = data.dropna(subset=["latitude", "longitude"]).copy()

if data.empty:
    raise ValueError("❌ Δεν υπάρχουν δεδομένα μετά το φιλτράρισμα.")


# ======================
# LOAD GREECE GEOMETRY ONCE
# ======================
greece = load_greece_geometry(GEOJSON_SECRET)

if greece.crs is None:
    greece = greece.set_crs("EPSG:4326")
else:
    try:
        greece = greece.to_crs("EPSG:4326")
    except Exception:
        pass

if hasattr(greece.geometry, "union_all"):
    greece_union = greece.geometry.union_all()
else:
    greece_union = greece.unary_union


# ======================
# COLORS / LEVELS
# ======================
cmap = ListedColormap([
    "#ffffff",  # 0–0.1
    "#e3f2fd",  # 0.1–5
    "#90caf9",  # 5–10
    "#64b5f6",  # 10–20
    "#42a5f5",  # 20–30
    "#1e88e5",  # 30–50
    "#6a1b9a",  # 50–75
    "#b71c1c",  # 75–100
    "#d32f2f",  # 100–150
    "#fb8c00",  # 150–200
    "#fdd835",  # 200–300
    "#ffe082"   # 300+
])

bounds = [0, 0.1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 1000]
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
contour_levels = [0.2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]


# ======================
# RUN ALL REGIONS
# ======================
timestamp = athens_now.strftime("%Y-%m-%d %H:%M %Z")
timestamp_text = "Δημιουργήθηκε για το e-kairos.gr\n" + timestamp

for reg in REGIONS:
    lon_min = reg["lon_min"]
    lon_max = reg["lon_max"]
    lat_min = reg["lat_min"]
    lat_max = reg["lat_max"]

    region_df = bbox_filter(data, lon_min, lon_max, lat_min, lat_max)

    map_data = region_df.dropna(
        subset=["total_precipitation", "latitude", "longitude"]
    ).copy()

    if map_data.empty:
        print("⚠️ No strict monthly data for region:", reg["key"], "-> skipping map")
        continue

    top_n = int(reg.get("top_n", TOP_N))
    top_text = build_top_text(region_df, top_n)

    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, GRID_N),
        np.linspace(lat_min, lat_max, GRID_N)
    )

    lats = map_data["latitude"].to_numpy(dtype=float)
    lons = map_data["longitude"].to_numpy(dtype=float)
    values = map_data["total_precipitation"].to_numpy(dtype=float)

    grid_intensity = idw_optimized(lons, lats, values, grid_x, grid_y)

    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()),
        crs="EPSG:4326"
    )
    mask_bool = grid_points.geometry.within(greece_union).to_numpy()
    mask_2d = mask_bool.reshape(grid_x.shape)

    masked_intensity = np.full(grid_x.shape, np.nan)
    masked_intensity[mask_2d] = grid_intensity[mask_2d]

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    img = ax.imshow(
        masked_intensity,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=0.7
    )

    greece_clip = greece.cx[lon_min:lon_max, lat_min:lat_max]
    if not greece_clip.empty:
        greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.5)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.contour(
        grid_x, grid_y, masked_intensity,
        levels=contour_levels, colors="black", linewidths=0.5
    )

    cbar = plt.colorbar(img, ax=ax, orientation="vertical", boundaries=bounds)
    cbar.set_ticks([0, 0.1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300])
    cbar.set_label("Σωρευτικός υετός μήνα (mm)", fontsize=12)

    ax.set_title(reg["title"], fontsize=16)
    ax.set_xlabel("Γεωγραφικό μήκος", fontsize=12)
    ax.set_ylabel("Γεωγραφικό πλάτος", fontsize=12)

    ts = ax.text(
        0.01, 0.01, timestamp_text, transform=ax.transAxes,
        fontsize=10, color="black", ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3")
    )
    ts.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    if top_text.strip():
        top_loc = reg.get("top_loc", "top_right")

        if top_loc == "bottom_right":
            x, y = 0.99, 0.01
            ha, va = "right", "bottom"
        else:
            x, y = 0.99, 0.99
            ha, va = "right", "top"

        top_block = f"Υψηλότερα ποσά (Top {top_n}):\n─────────────────────\n" + top_text

        txt = ax.text(
            x, y, top_block, transform=ax.transAxes,
            fontsize=TOP_FONTSIZE, color="black", ha=ha, va=va,
            bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.22")
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    buffer_main = io.BytesIO()
    fig.savefig(buffer_main, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    upload_to_ftp(buffer_main, reg["outfile"])
    print("✅ Region monthly map uploaded:", reg["outfile"])

print("✅ Done (current month precipitation, all regions processed).")
