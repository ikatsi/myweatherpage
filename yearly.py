#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import zipfile
import subprocess
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
from ftplib import FTP_TLS
import time
import socket

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter, MaxNLocator

import requests
import rasterio
from pyproj import Transformer


# ======================
# CONFIGURATION
# ======================
TXT_URL = os.environ.get("CURRENTYEARURL", "").strip()
GEOJSON_PATH = "greece.geojson"
DEM_PATH = "GRC_alt.vrt"

FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()

PLOT_WIDTH = 12
PLOT_HEIGHT = 8

GRID_N = 300
TOP_N = 30
TOP_FONTSIZE = 7.0

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 15

BASE_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
ALT_ENC = os.path.join(BASE_DIR, "altitude.zip.enc")
ALT_ZIP = os.path.join(BASE_DIR, "altitude.zip")

CRS_WGS84 = "EPSG:4326"
CRS_EGSA87 = "EPSG:2100"
WGS_TO_EGSA = Transformer.from_crs(CRS_WGS84, CRS_EGSA87, always_xy=True)
EGSA_TO_WGS = Transformer.from_crs(CRS_EGSA87, CRS_WGS84, always_xy=True)

# Regional interpolation settings in meters
REG_IDW_K = 8
REG_IDW_POWER = 2
REG_MAX_DISTANCE_M = 120_000
REG_MIN_NEIGHBORS = 3
REG_DISTANCE_MASK_M = 170_000
REG_STATION_BUFFER_M = 200_000

# Yearly mean temperature lapse-rate settings
LAPSE_DEFAULT = -0.0065
LAPSE_MIN = -0.0100
LAPSE_MAX = 0.0020
LAPSE_K = 20
LAPSE_RADIUS_M = 150_000
LAPSE_MIN_NBR = 6
LAPSE_ALT_RANGE_MIN_M = 150

if not TXT_URL:
    raise RuntimeError("Environment variable CURRENTYEARURL is not set.")

if not GEOJSON_PASS:
    raise RuntimeError("Environment variable GEOJSON_PASS is not set.")

if not FTP_HOST or not FTP_USER or not FTP_PASS:
    raise RuntimeError("FTP_HOST / FTP_USER / FTP_PASS environment variables are not all set.")


# ======================
# REGION DEFINITIONS
# ======================
REGIONS = [
    {
        "key": "greece",
        "title": "Υπολ. σωρευτικός υετός στην Ελλάδα (τρέχον έτος)",
        "outfile": "yearlyppn.png",
        "temp_title": "Μέση θερμοκρασία στην Ελλάδα (τρέχον έτος, προσαρμογή υψομέτρου)",
        "temp_outfile": "yearlytavg.png",
        "hydro_title": "Υπολ. σωρευτικός υετός στην Ελλάδα (τρέχον υδρολογικό έτος)",
        "hydro_outfile": "yearlyhydroppn.png",
        "lon_min": 19.0,  "lon_max": 30.0,
        "lat_min": 34.5,  "lat_max": 42.5,
        "temp_mode": "wgs84"
    },
    {
        "key": "attica",
        "title": "Υπολ. σωρευτικός υετός Αττικής (τρέχον έτος)",
        "outfile": "yearlyppn_attica.png",
        "temp_title": "Μέση θερμοκρασία Αττικής (τρέχον έτος, προσαρμογή υψομέτρου)",
        "temp_outfile": "yearlytavg_attica.png",
        "hydro_title": "Υπολ. σωρευτικός υετός Αττικής (τρέχον υδρολογικό έτος)",
        "hydro_outfile": "yearlyhydroppn_attica.png",
        "lon_min": 22.7,  "lon_max": 25.0,
        "lat_min": 37.5,  "lat_max": 38.7,
        "temp_mode": "egsa"
    },
    {
        "key": "negreece",
        "title": "Υπολ. σωρευτικός υετός ΒΑ Ελλάδας (τρέχον έτος)",
        "outfile": "yearlyppn_negreece.png",
        "temp_title": "Μέση θερμοκρασία ΒΑ Ελλάδας (τρέχον έτος, προσαρμογή υψομέτρου)",
        "temp_outfile": "yearlytavg_negreece.png",
        "hydro_title": "Υπολ. σωρευτικός υετός ΒΑ Ελλάδας (τρέχον υδρολογικό έτος)",
        "hydro_outfile": "yearlyhydroppn_negreece.png",
        "lon_min": 22.0,  "lon_max": 26.6,
        "lat_min": 39.7,  "lat_max": 41.8,
        "top_n": 15,
        "top_loc": "bottom_right",
        "temp_mode": "egsa"
    },
    {
        "key": "crete",
        "title": "Υπολ. σωρευτικός υετός Κρήτης (τρέχον έτος)",
        "outfile": "yearlyppn_crete.png",
        "temp_title": "Μέση θερμοκρασία Κρήτης (τρέχον έτος, προσαρμογή υψομέτρου)",
        "temp_outfile": "yearlytavg_crete.png",
        "hydro_title": "Υπολ. σωρευτικός υετός Κρήτης (τρέχον υδρολογικό έτος)",
        "hydro_outfile": "yearlyhydroppn_crete.png",
        "lon_min": 23.37, "lon_max": 26.4,
        "lat_min": 34.7,  "lat_max": 35.78,
        "top_n": 12,
        "temp_mode": "egsa"
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


def fmt_c_gr(x):
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
        denom = np.sum(weights, axis=1)
        num = np.sum(weights * z[idx], axis=1)
        zi = np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float), where=denom > 0)

    return zi.reshape(xi.shape)


def idw_fast(x, y, z, xi, yi, k=8, power=2, max_distance=1.0, min_neighbors=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(z)

    if n == 0:
        return np.full(xi.shape, np.nan, dtype=float)

    tree = cKDTree(np.c_[x, y])
    dist, idx = tree.query(
        np.c_[xi.ravel(), yi.ravel()],
        k=min(k, n),
        distance_upper_bound=max_distance
    )

    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    finite = np.isfinite(dist) & (idx < n)
    neigh_count = np.sum(finite, axis=1)

    zi = np.full(xi.size, np.nan, dtype=float)
    ok_pts = neigh_count >= min_neighbors
    if not np.any(ok_pts):
        return zi.reshape(xi.shape)

    idx_safe = np.where(finite, idx, 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(dist == 0, 1e12, 1.0 / (dist ** power))
        w = np.where(np.isfinite(w) & finite, w, 0.0)

    z_nei = z[idx_safe]
    num = np.sum(w * z_nei, axis=1)
    den = np.sum(w, axis=1)

    zi_ok = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
    zi[ok_pts] = zi_ok[ok_pts]
    return zi.reshape(xi.shape)


def bbox_filter(df, lon_min, lon_max, lat_min, lat_max):
    return df[
        (df["longitude"].between(lon_min, lon_max))
        & (df["latitude"].between(lat_min, lat_max))
    ].copy()


def build_top_text(region_df, top_n, value_col):
    top_source = region_df.dropna(subset=[value_col]).copy()
    if top_source.empty:
        return ""

    topn = top_source.nlargest(int(top_n), value_col)
    lines = []

    for _, r in topn.iterrows():
        name, is_citygr = station_name(r)
        if is_citygr:
            name = abbreviate_gr_name(name)
        mm = fmt_mm_gr(r[value_col])
        if name and mm:
            lines.append(f"{name} {mm} mm")

    return "\n".join(lines)


def build_temp_top_text(region_df, top_n):
    top_source = region_df.dropna(subset=["avg_tavg"]).copy()
    if top_source.empty:
        return ""

    hotn = top_source.nlargest(int(top_n), "avg_tavg")
    lines = []

    for _, r in hotn.iterrows():
        name, is_citygr = station_name(r)
        if is_citygr:
            name = abbreviate_gr_name(name)
        tv = fmt_c_gr(r["avg_tavg"])
        if name and tv:
            lines.append(f"{name} {tv}°C")

    return "\n".join(lines)


def ensure_altitude_bundle():
    if os.path.exists(DEM_PATH):
        return

    if not os.path.exists(ALT_ENC):
        raise RuntimeError(
            "GRC_alt.vrt not found and altitude.zip.enc is also missing. "
            "Yearly temperature maps need the DEM bundle."
        )

    if not GEOJSON_PASS:
        raise RuntimeError("GEOJSON_PASS not set, cannot decrypt altitude.zip.enc")

    try:
        subprocess.check_call([
            "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
            "-in", ALT_ENC, "-out", ALT_ZIP, "-pass", "pass:" + GEOJSON_PASS
        ])
    except FileNotFoundError:
        raise RuntimeError("OpenSSL not found. Install it or decrypt altitude.zip.enc in the workflow.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("OpenSSL decryption failed for altitude.zip.enc: {}".format(e))

    with zipfile.ZipFile(ALT_ZIP, "r") as zf:
        zf.extractall(BASE_DIR)

    if not os.path.exists(DEM_PATH):
        raise RuntimeError("Decrypted bundle did not contain GRC_alt.vrt")

    try:
        os.remove(ALT_ZIP)
    except Exception:
        pass


def sample_dem_lonlat(dem_path, lons, lats):
    if not os.path.exists(dem_path):
        raise FileNotFoundError("DEM not found at: {}".format(dem_path))

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        samples = list(src.sample(zip(lons.tolist(), lats.tolist())))
        elev = np.array([s[0] for s in samples], dtype=float)

        if nodata is not None:
            elev = np.where(elev == nodata, np.nan, elev)
        elev = np.where(elev < -100, np.nan, elev)
        elev = np.where(np.isfinite(elev), elev, 0.0)

    return elev


def build_geo_mask_wgs(grid_x, grid_y, greece_gdf_wgs):
    if hasattr(greece_gdf_wgs.geometry, "union_all"):
        boundary = greece_gdf_wgs.geometry.union_all()
    else:
        boundary = greece_gdf_wgs.unary_union

    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()),
        crs=greece_gdf_wgs.crs
    )
    geom = pts.geometry
    mask = (geom.within(boundary) | geom.touches(boundary)).values
    return mask.reshape(grid_x.shape)


def build_geo_mask_egsa(grid_x_m, grid_y_m, greece_gdf_egsa, x_min, x_max, y_min, y_max):
    greece_clip = greece_gdf_egsa.cx[x_min:x_max, y_min:y_max].copy()
    if greece_clip.empty:
        return np.zeros(grid_x_m.shape, dtype=bool), greece_clip

    if hasattr(greece_clip.geometry, "union_all"):
        boundary = greece_clip.geometry.union_all()
    else:
        boundary = greece_clip.unary_union

    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_EGSA87
    )
    geom = pts.geometry
    mask = (geom.within(boundary) | geom.touches(boundary)).values.reshape(grid_x_m.shape)
    return mask, greece_clip


def build_distance_mask(xgrid, ygrid, xs, ys, max_dist):
    tree = cKDTree(np.c_[xs, ys])
    d, _ = tree.query(np.c_[xgrid.ravel(), ygrid.ravel()])
    return (d.reshape(xgrid.shape) <= max_dist)


def projected_bbox_from_wgs_bbox(lon_min, lon_max, lat_min, lat_max, n=200):
    top_lons = np.linspace(lon_min, lon_max, n)
    top_lats = np.full(n, lat_max)

    bottom_lons = np.linspace(lon_min, lon_max, n)
    bottom_lats = np.full(n, lat_min)

    left_lons = np.full(n, lon_min)
    left_lats = np.linspace(lat_min, lat_max, n)

    right_lons = np.full(n, lon_max)
    right_lats = np.linspace(lat_min, lat_max, n)

    all_lons = np.concatenate([top_lons, bottom_lons, left_lons, right_lons])
    all_lats = np.concatenate([top_lats, bottom_lats, left_lats, right_lats])

    xs, ys = WGS_TO_EGSA.transform(all_lons.tolist(), all_lats.tolist())
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    return float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))


def estimate_local_lapse_rates_wgs(st_lons, st_lats, st_temp, st_elev,
                                   k=12, max_deg=1.2,
                                   default_lapse=LAPSE_DEFAULT,
                                   clip_min=LAPSE_MIN, clip_max=LAPSE_MAX):
    tree = cKDTree(np.c_[st_lons, st_lats])
    d, idx = tree.query(np.c_[st_lons, st_lats], k=min(k, len(st_temp)), distance_upper_bound=max_deg)

    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]

    lapses = np.full(len(st_temp), np.nan, dtype=float)

    for i in range(len(st_temp)):
        neigh = idx[i]
        dist = d[i]
        ok = np.isfinite(dist) & (neigh < len(st_temp))
        neigh = neigh[ok]

        if neigh.size < 4:
            lapses[i] = default_lapse
            continue

        elev_n = st_elev[neigh]
        t_n = st_temp[neigh]
        good = np.isfinite(elev_n) & np.isfinite(t_n)

        elev_n = elev_n[good]
        t_n = t_n[good]

        if elev_n.size < 4 or (float(np.nanmax(elev_n)) - float(np.nanmin(elev_n))) < 100:
            lapses[i] = default_lapse
            continue

        try:
            b, _a = np.polyfit(elev_n, t_n, 1)
            b = float(np.clip(b, clip_min, clip_max))
            lapses[i] = b
        except Exception:
            lapses[i] = default_lapse

    return lapses


def estimate_local_lapse_rates_egsa(st_x, st_y, st_temp, st_elev,
                                    k=LAPSE_K, radius_m=LAPSE_RADIUS_M,
                                    default_lapse=LAPSE_DEFAULT,
                                    clip_min=LAPSE_MIN, clip_max=LAPSE_MAX):
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)
    st_temp = np.asarray(st_temp, dtype=float)
    st_elev = np.asarray(st_elev, dtype=float)

    tree = cKDTree(np.c_[st_x, st_y])
    d, idx = tree.query(np.c_[st_x, st_y], k=min(k, len(st_temp)), distance_upper_bound=radius_m)

    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]

    lapses = np.full(len(st_temp), np.nan, dtype=float)

    for i in range(len(st_temp)):
        neigh = idx[i]
        dist = d[i]
        ok = np.isfinite(dist) & (neigh < len(st_temp))
        neigh = neigh[ok]

        if neigh.size < LAPSE_MIN_NBR:
            lapses[i] = default_lapse
            continue

        elev_n = st_elev[neigh]
        t_n = st_temp[neigh]
        good = np.isfinite(elev_n) & np.isfinite(t_n)
        elev_n = elev_n[good]
        t_n = t_n[good]

        if elev_n.size < LAPSE_MIN_NBR or (np.nanmax(elev_n) - np.nanmin(elev_n)) < LAPSE_ALT_RANGE_MIN_M:
            lapses[i] = default_lapse
            continue

        try:
            b, _a = np.polyfit(elev_n, t_n, 1)
            b = float(np.clip(b, clip_min, clip_max))
            lapses[i] = b
        except Exception:
            lapses[i] = default_lapse

    return lapses


def add_temp_contours(ax, X, Y, field):
    levels = np.arange(-30, 46, 3, dtype=float)
    thin_levels = [lv for lv in levels if abs(lv) > 1e-9]

    cs_thin = None
    cs_zero = None

    try:
        cs_thin = ax.contour(
            X, Y, field,
            levels=thin_levels,
            colors="black",
            linewidths=0.6,
            alpha=0.70
        )
    except Exception:
        cs_thin = None

    try:
        cs_zero = ax.contour(
            X, Y, field,
            levels=[0.0],
            colors="black",
            linewidths=1.3,
            alpha=0.95
        )
    except Exception:
        cs_zero = None

    if cs_thin is not None:
        try:
            texts = ax.clabel(
                cs_thin,
                levels=cs_thin.levels[:],
                inline=True,
                inline_spacing=2,
                fmt="%d",
                fontsize=7
            )
            for t in texts:
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        except Exception:
            pass

    if cs_zero is not None:
        try:
            texts0 = ax.clabel(
                cs_zero,
                inline=True,
                inline_spacing=2,
                fmt="0",
                fontsize=7
            )
            for t in texts0:
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        except Exception:
            pass


# ======================
# SHARED TEMP PALETTE
# ======================
TEMP_VMIN = -25.0
TEMP_VMAX = 45.0


def build_shared_temp_cmap_norm():
    anchors = [
        (-25.0, "#0b1d5c"),
        (-18.0, "#123b8a"),
        (-12.0, "#1f63c6"),
        (-6.0,  "#2f8fe6"),
        (-2.0,  "#44b6ff"),
        (0.0,   "#2b7bff"),
        (3.0,   "#2fb8d6"),
        (7.0,   "#2fc4a0"),
        (12.0,  "#34c759"),
        (18.0,  "#b7dd2a"),
        (24.0,  "#ffe11a"),
        (30.0,  "#ff9a1a"),
        (35.0,  "#ff4d1a"),
        (40.0,  "#d1166f"),
        (45.0,  "#6a00a8"),
    ]
    vals = np.array([v for v, _ in anchors], dtype=float)
    cols = [c for _, c in anchors]
    t = (vals - TEMP_VMIN) / (TEMP_VMAX - TEMP_VMIN)
    t = np.clip(t, 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("t_shared", list(zip(t, cols)), N=256)
    norm = Normalize(vmin=TEMP_VMIN, vmax=TEMP_VMAX, clip=True)
    return cmap, norm


TEMP_CMAP, TEMP_NORM = build_shared_temp_cmap_norm()


# ======================
# FTP SESSION HELPERS
# ======================
def ftp_connect(max_attempts=4, sleep_seconds=15):
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            print("🔌 FTPS connect attempt {}/{} to host: {}".format(attempt, max_attempts, FTP_HOST))
            ftps = FTP_TLS()
            ftps.connect(FTP_HOST, 21, timeout=60)
            ftps.login(user=FTP_USER, passwd=FTP_PASS)
            ftps.prot_p()
            print("✅ FTPS connection established.")
            return ftps

        except socket.gaierror as e:
            last_exc = e
            print("⚠️ DNS resolution failed for FTP host on attempt {}: {}".format(attempt, e))

        except Exception as e:
            last_exc = e
            print("⚠️ FTPS connection/login failed on attempt {}: {}".format(attempt, e))

        if attempt < max_attempts:
            print("⏳ Waiting {}s before retry...".format(sleep_seconds))
            time.sleep(sleep_seconds)

    raise RuntimeError("FTPS connection failed after {} attempts: {}".format(max_attempts, last_exc))


def upload_via_session(ftps, file_buffer, filename):
    try:
        file_buffer.seek(0)
        ftps.storbinary("STOR {}".format(filename), file_buffer)
        print("📤 Uploaded: {}".format(filename))
        return True
    except Exception as e:
        print("⚠️ FTP upload failed for {}: {}".format(filename, e))
        return False


# ======================
# LOAD RAW DATA ONCE
# ======================
response = requests.get(TXT_URL, headers=HEADERS, timeout=TIMEOUT)
response.raise_for_status()
response.encoding = "utf-8"

text = response.text.replace("\r\n", "\n").replace("\r", "\n")

print("=== FIRST 20 RESPONSE LINES START ===")
for i, line in enumerate(text.splitlines()[:20], start=1):
    print(f"{i:02d}: {line}")
print("=== FIRST 20 RESPONSE LINES END ===")

raw_data = pd.read_csv(
    StringIO(text),
    sep="\t",
    engine="python"
)

athens_now = datetime.now(ZoneInfo("Europe/Athens"))
today_yday = athens_now.timetuple().tm_yday

if "webcode" not in raw_data.columns:
    raise RuntimeError("Input data has no 'webcode' column.")

raw_data["webcode"] = raw_data["webcode"].astype("string")
w = raw_data["webcode"].str.strip().str.casefold()

# Numeric columns
for col in [
    "latitude", "longitude",
    "daysinyear",
    "total_precipitation", "total_precipitation_missing",
    "avg_tavg",
    "hydro_year_precipitation", "hydro_year_precipitation_missing"
]:
    if col in raw_data.columns:
        raw_data[col] = pd.to_numeric(raw_data[col], errors="coerce")

# -------- Existing yearly/calendar filter logic --------
excluded_exact_year = {
    "agrivate_stavroupoli", "age_dasosxiromerou",
    "agrivate_rizia", "age_agiosilias",
    "hnms3_megara", "wu_varnavas", "wu_sykamino", "wu_avlonas",
    "age_galatas", "agrivate_messouni", "voutsaras", "pws_proti2", "age_vrana", "potamoi", "age_leptokarya"
}
excluded_prefixes_year = ("hcmr", "uoi_")

if "daysinyear" in raw_data.columns:
    data_year = raw_data[
        (raw_data["daysinyear"] == today_yday)
        & (~w.isin(excluded_exact_year))
        & (~w.str.startswith(excluded_prefixes_year, na=False))
    ].copy()
else:
    data_year = raw_data[
        (~w.isin(excluded_exact_year))
        & (~w.str.startswith(excluded_prefixes_year, na=False))
    ].copy()

data_year = data_year.dropna(subset=["latitude", "longitude"]).copy()

# -------- Hydro filter logic --------
excluded_exact_hydro = {
    "agrivate_stavroupoli", "age_dasosxiromerou",
    "agrivate_rizia", "age_agiosilias", "metaxochori",
    "hnms3_megara", "wu_varnavas", "wu_sykamino", "wu_avlonas",
    "age_galatas", "agrivate_messouni", "voutsaras",
    "ierapetra", "age_vrana", "potamoi"
}
excluded_prefixes_hydro = ("hcmr", "uoi_")

if "daysinyear" in raw_data.columns:
    data_hydro = raw_data[
        (raw_data["daysinyear"] == today_yday)
        & (~w.isin(excluded_exact_hydro))
        & (~w.str.startswith(excluded_prefixes_hydro, na=False))
    ].copy()
else:
    data_hydro = raw_data[
        (~w.isin(excluded_exact_hydro))
        & (~w.str.startswith(excluded_prefixes_hydro, na=False))
    ].copy()

data_hydro = data_hydro.dropna(subset=["latitude", "longitude"]).copy()

if data_year.empty and data_hydro.empty:
    raise ValueError("❌ Δεν υπάρχουν δεδομένα μετά το φιλτράρισμα.")


# ======================
# LOAD GREECE GEOMETRY ONCE
# ======================
if not os.path.exists(GEOJSON_PATH):
    raise RuntimeError("greece.geojson not found. It should be decrypted by the workflow before yearly.py runs.")

greece = gpd.read_file(GEOJSON_PATH)

if greece.crs is None:
    greece = greece.set_crs("EPSG:4326")
else:
    try:
        greece = greece.to_crs("EPSG:4326")
    except Exception:
        pass

greece_egsa = greece.to_crs(CRS_EGSA87)

if hasattr(greece.geometry, "union_all"):
    greece_union = greece.geometry.union_all()
else:
    greece_union = greece.unary_union


# ======================
# PRECIPITATION COLORS / LEVELS
# ======================
PPN_CMAP = ListedColormap([
    "#ffffff",
    "#e3f2fd",
    "#90caf9",
    "#64b5f6",
    "#42a5f5",
    "#1e88e5",
    "#6a1b9a",
    "#b71c1c",
    "#d32f2f",
    "#fb8c00",
    "#fdd835",
    "#ffe082"
])

PPN_BOUNDS = [0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1500, 10000]
PPN_NORM = BoundaryNorm(boundaries=PPN_BOUNDS, ncolors=PPN_CMAP.N)
PPN_CONTOUR_LEVELS = [120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1500]


# ======================
# MAP BUILDERS
# ======================
def build_precip_map_buffer_wgs(
    region_df,
    reg,
    timestamp_text,
    strict_value_col,
    top_value_col,
    title_text,
    cbar_label
):
    lon_min = reg["lon_min"]
    lon_max = reg["lon_max"]
    lat_min = reg["lat_min"]
    lat_max = reg["lat_max"]

    map_data = region_df.dropna(
        subset=[strict_value_col, "latitude", "longitude"]
    ).copy()

    if map_data.empty:
        print("⚠️ No precipitation data for region:", reg["key"], "column:", strict_value_col, "-> skipping map")
        return None

    top_n = int(reg.get("top_n", TOP_N))
    top_text = build_top_text(region_df, top_n, top_value_col)

    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, GRID_N),
        np.linspace(lat_min, lat_max, GRID_N)
    )

    lats = map_data["latitude"].to_numpy(dtype=float)
    lons = map_data["longitude"].to_numpy(dtype=float)
    values = map_data[strict_value_col].to_numpy(dtype=float)

    grid_intensity = idw_optimized(lons, lats, values, grid_x, grid_y)

    mask_2d = build_geo_mask_wgs(grid_x, grid_y, greece)
    masked_intensity = np.full(grid_x.shape, np.nan)
    masked_intensity[mask_2d] = grid_intensity[mask_2d]

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    img = ax.imshow(
        masked_intensity,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="lower",
        cmap=PPN_CMAP,
        norm=PPN_NORM,
        alpha=0.7
    )

    greece_clip = greece.cx[lon_min:lon_max, lat_min:lat_max]
    if not greece_clip.empty:
        greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.5)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.contour(
        grid_x, grid_y, masked_intensity,
        levels=PPN_CONTOUR_LEVELS, colors="black", linewidths=0.5
    )

    cbar = plt.colorbar(img, ax=ax, orientation="vertical", boundaries=PPN_BOUNDS)
    cbar.set_ticks([0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1500])
    cbar.set_label(cbar_label, fontsize=12)

    ax.set_title(title_text, fontsize=16)
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
    buffer_main.seek(0)
    return buffer_main


def build_precip_region_egsa_buffer(
    region_df,
    reg,
    timestamp_text,
    strict_value_col,
    top_value_col,
    title_text,
    cbar_label
):
    lon_min = reg["lon_min"]
    lon_max = reg["lon_max"]
    lat_min = reg["lat_min"]
    lat_max = reg["lat_max"]

    map_data = region_df.dropna(
        subset=[strict_value_col, "latitude", "longitude"]
    ).copy()

    if map_data.empty:
        print("⚠️ No precipitation data for region:", reg["key"], "column:", strict_value_col, "-> skipping map")
        return None

    top_n = int(reg.get("top_n", TOP_N))
    top_text = build_top_text(region_df, top_n, top_value_col)

    x_min, x_max, y_min, y_max = projected_bbox_from_wgs_bbox(
        lon_min, lon_max, lat_min, lat_max, n=200
    )

    st_lon = map_data["longitude"].to_numpy(dtype=float)
    st_lat = map_data["latitude"].to_numpy(dtype=float)
    st_val = map_data[strict_value_col].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    near = (
        (st_x >= (x_min - REG_STATION_BUFFER_M)) & (st_x <= (x_max + REG_STATION_BUFFER_M))
        & (st_y >= (y_min - REG_STATION_BUFFER_M)) & (st_y <= (y_max + REG_STATION_BUFFER_M))
    )

    st_x = st_x[near]
    st_y = st_y[near]
    st_val = st_val[near]

    if len(st_val) < REG_MIN_NEIGHBORS:
        print("⚠️ Too few nearby precipitation stations for region:", reg["key"])
        return None

    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, GRID_N),
        np.linspace(y_min, y_max, GRID_N)
    )

    grid_val = idw_fast(
        st_x, st_y, st_val,
        grid_x_m, grid_y_m,
        k=REG_IDW_K,
        power=REG_IDW_POWER,
        max_distance=REG_MAX_DISTANCE_M,
        min_neighbors=REG_MIN_NEIGHBORS
    )

    geo_mask, greece_clip = build_geo_mask_egsa(
        grid_x_m, grid_y_m, greece_egsa, x_min, x_max, y_min, y_max
    )
    if greece_clip.empty:
        print("⚠️ Empty clipped geometry for region:", reg["key"])
        return None

    dist_mask = build_distance_mask(
        grid_x_m, grid_y_m, st_x, st_y, max_dist=REG_DISTANCE_MASK_M
    )

    final_mask = geo_mask & dist_mask

    out = np.full(grid_x_m.shape, np.nan, dtype=float)
    out[final_mask] = grid_val[final_mask]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        np.ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=PPN_CMAP,
        norm=PPN_NORM,
        alpha=0.85
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)

    try:
        ax.contour(
            grid_x_m, grid_y_m, out,
            levels=PPN_CONTOUR_LEVELS,
            colors="black",
            linewidths=0.5
        )
    except Exception:
        pass

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = EGSA_TO_WGS.transform(x, y_ref_for_lon)
        return f"{lon:.2f}"

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return f"{lat:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)
    ax.set_title(title_text, fontsize=16, pad=10)

    cbar = fig.colorbar(img, ax=ax, orientation="vertical", boundaries=PPN_BOUNDS, fraction=0.035, pad=0.02)
    cbar.set_ticks([0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1500])
    cbar.set_label(cbar_label, fontsize=12)

    ts = ax.text(
        0.01, 0.01, timestamp_text,
        transform=ax.transAxes, fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3")
    )
    ts.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

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
    buffer_main.seek(0)
    return buffer_main


def build_yearly_tavg_greece_buffer(region_df, reg, greece_gdf_wgs, athens_now, timestamp_text):
    if "avg_tavg" not in region_df.columns:
        print("⚠️ avg_tavg missing -> skipping:", reg["key"])
        return None

    tt = region_df.dropna(subset=["avg_tavg", "latitude", "longitude"]).copy()
    if tt.empty:
        print("⚠️ No yearly avg_tavg data for region:", reg["key"])
        return None

    lon_min = reg["lon_min"]
    lon_max = reg["lon_max"]
    lat_min = reg["lat_min"]
    lat_max = reg["lat_max"]

    grid_x, grid_y = np.meshgrid(
        np.linspace(lon_min, lon_max, GRID_N),
        np.linspace(lat_min, lat_max, GRID_N)
    )

    st_lons = tt["longitude"].to_numpy(dtype=float)
    st_lats = tt["latitude"].to_numpy(dtype=float)
    st_t = tt["avg_tavg"].to_numpy(dtype=float)

    st_elev = sample_dem_lonlat(DEM_PATH, st_lons, st_lats)

    ok = np.isfinite(st_lons) & np.isfinite(st_lats) & np.isfinite(st_t) & np.isfinite(st_elev)
    st_lons = st_lons[ok]
    st_lats = st_lats[ok]
    st_t = st_t[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 5:
        print("⚠️ Too few stations for yearly avg_tavg in region:", reg["key"])
        return None

    st_lapse = estimate_local_lapse_rates_wgs(st_lons, st_lats, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    t0_grid = idw_fast(
        st_lons, st_lats, st_t0, grid_x, grid_y,
        k=8, power=2, max_distance=1.2, min_neighbors=3
    )
    lapse_grid = idw_fast(
        st_lons, st_lats, st_lapse, grid_x, grid_y,
        k=8, power=2, max_distance=1.2, min_neighbors=3
    )

    grid_elev = sample_dem_lonlat(DEM_PATH, grid_x.ravel(), grid_y.ravel()).reshape(grid_x.shape)
    t_grid = t0_grid + (lapse_grid * grid_elev)

    geo_mask = build_geo_mask_wgs(grid_x, grid_y, greece_gdf_wgs)
    dist_mask = build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_dist=0.8)
    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    img = ax.imshow(
        np.ma.masked_invalid(out),
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_clip = greece_gdf_wgs.cx[lon_min:lon_max, lat_min:lat_max]
    if not greece_clip.empty:
        greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.5)

    add_temp_contours(ax, grid_x, grid_y, out)

    cbar = plt.colorbar(img, ax=ax, orientation="vertical", extend="both")
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Μέση θερμοκρασία έτους (°C)", fontsize=12)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(reg["temp_title"], fontsize=16)
    ax.set_xlabel("Γεωγραφικό μήκος", fontsize=12)
    ax.set_ylabel("Γεωγραφικό πλάτος", fontsize=12)

    ts = ax.text(
        0.01, 0.01, timestamp_text, transform=ax.transAxes,
        fontsize=10, color="black", ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3")
    )
    ts.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    if np.isfinite(out).any():
        interp_min = float(np.nanmin(out))
        interp_max = float(np.nanmax(out))
        mm_text = "Εύρος παρεμβολής (ξηρά):\n{0:.1f} έως {1:.1f}°C".format(interp_min, interp_max)
        tx = ax.text(
            0.01, 0.985, mm_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            color="black",
            bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.2")
        )
        tx.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    top_n = int(reg.get("top_n", TOP_N))
    top_text = build_temp_top_text(region_df, top_n)

    if top_text.strip():
        top_loc = reg.get("top_loc", "top_right")
        if top_loc == "bottom_right":
            x, y = 0.99, 0.01
            ha, va = "right", "bottom"
        else:
            x, y = 0.99, 0.99
            ha, va = "right", "top"

        top_block = f"Υψηλότερες μέσες Τ (Top {top_n}):\n─────────────────────\n" + top_text

        txt = ax.text(
            x, y, top_block, transform=ax.transAxes,
            fontsize=TOP_FONTSIZE, color="black", ha=ha, va=va,
            bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.22")
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    buffer_main = io.BytesIO()
    fig.savefig(buffer_main, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buffer_main.seek(0)
    return buffer_main


def build_yearly_tavg_region_egsa_buffer(region_df, reg, greece_gdf_wgs, athens_now, timestamp_text):
    if "avg_tavg" not in region_df.columns:
        print("⚠️ avg_tavg missing -> skipping:", reg["key"])
        return None

    tt = region_df.dropna(subset=["avg_tavg", "latitude", "longitude"]).copy()
    if tt.empty:
        print("⚠️ No yearly avg_tavg data for region:", reg["key"])
        return None

    lon_min = reg["lon_min"]
    lon_max = reg["lon_max"]
    lat_min = reg["lat_min"]
    lat_max = reg["lat_max"]

    x_min, x_max, y_min, y_max = projected_bbox_from_wgs_bbox(
        lon_min, lon_max, lat_min, lat_max, n=200
    )

    st_lon = tt["longitude"].to_numpy(dtype=float)
    st_lat = tt["latitude"].to_numpy(dtype=float)
    st_t = tt["avg_tavg"].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    near = (
        (st_x >= (x_min - REG_STATION_BUFFER_M)) & (st_x <= (x_max + REG_STATION_BUFFER_M))
        & (st_y >= (y_min - REG_STATION_BUFFER_M)) & (st_y <= (y_max + REG_STATION_BUFFER_M))
    )

    st_lon = st_lon[near]
    st_lat = st_lat[near]
    st_t = st_t[near]
    st_x = st_x[near]
    st_y = st_y[near]

    if len(st_t) < 8:
        print("⚠️ Too few nearby stations for yearly avg_tavg in region:", reg["key"])
        return None

    st_elev = sample_dem_lonlat(DEM_PATH, st_lon, st_lat)

    ok = np.isfinite(st_t) & np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_elev)
    st_t = st_t[ok]
    st_x = st_x[ok]
    st_y = st_y[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 8:
        print("⚠️ Too few valid stations after DEM for region:", reg["key"])
        return None

    st_lapse = estimate_local_lapse_rates_egsa(st_x, st_y, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, GRID_N),
        np.linspace(y_min, y_max, GRID_N)
    )

    t0_grid = idw_fast(
        st_x, st_y, st_t0, grid_x_m, grid_y_m,
        k=REG_IDW_K, power=REG_IDW_POWER,
        max_distance=REG_MAX_DISTANCE_M, min_neighbors=REG_MIN_NEIGHBORS
    )

    lapse_grid = idw_fast(
        st_x, st_y, st_lapse, grid_x_m, grid_y_m,
        k=REG_IDW_K, power=REG_IDW_POWER,
        max_distance=REG_MAX_DISTANCE_M, min_neighbors=REG_MIN_NEIGHBORS
    )

    glon, glat = EGSA_TO_WGS.transform(grid_x_m.ravel().tolist(), grid_y_m.ravel().tolist())
    grid_elev = sample_dem_lonlat(
        DEM_PATH,
        np.array(glon, dtype=float),
        np.array(glat, dtype=float)
    ).reshape(grid_x_m.shape)

    t_grid = t0_grid + (lapse_grid * grid_elev)

    geo_mask, greece_clip = build_geo_mask_egsa(
        grid_x_m, grid_y_m, greece_egsa, x_min, x_max, y_min, y_max
    )
    if greece_clip.empty:
        print("⚠️ Empty clipped geometry for region:", reg["key"])
        return None

    dist_mask = build_distance_mask(
        grid_x_m, grid_y_m, st_x, st_y, max_dist=REG_DISTANCE_MASK_M
    )

    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x_m.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        np.ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = EGSA_TO_WGS.transform(x, y_ref_for_lon)
        return f"{lon:.2f}"

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return f"{lat:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_temp_contours(ax, grid_x_m, grid_y_m, out)

    cbar = fig.colorbar(img, ax=ax, orientation="vertical", extend="both", fraction=0.035, pad=0.02)
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Μέση θερμοκρασία έτους (°C)", fontsize=12)

    ax.set_title(reg["temp_title"], fontsize=16, pad=10)

    ts = ax.text(
        0.01, 0.01, timestamp_text,
        transform=ax.transAxes, fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3")
    )
    ts.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

    if np.isfinite(out).any():
        interp_min = float(np.nanmin(out))
        interp_max = float(np.nanmax(out))
        mm_text = "Εύρος παρεμβολής (ξηρά):\n{0:.1f} έως {1:.1f}°C".format(interp_min, interp_max)
        tx = ax.text(
            0.01, 0.985, mm_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            color="black",
            bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.2")
        )
        tx.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    top_n = int(reg.get("top_n", TOP_N))
    top_text = build_temp_top_text(region_df, top_n)

    if top_text.strip():
        top_loc = reg.get("top_loc", "top_right")
        if top_loc == "bottom_right":
            x, y = 0.99, 0.01
            ha, va = "right", "bottom"
        else:
            x, y = 0.99, 0.99
            ha, va = "right", "top"

        top_block = f"Υψηλότερες μέσες Τ (Top {top_n}):\n─────────────────────\n" + top_text

        txt = ax.text(
            x, y, top_block, transform=ax.transAxes,
            fontsize=TOP_FONTSIZE, color="black", ha=ha, va=va,
            bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.22")
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    buffer_main = io.BytesIO()
    fig.savefig(buffer_main, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buffer_main.seek(0)
    return buffer_main


# ======================
# MAIN
# ======================
def main():
    ensure_altitude_bundle()

    timestamp = athens_now.strftime("%Y-%m-%d %H:%M %Z")
    timestamp_text = "Δημιουργήθηκε για το e-kairos.gr\n" + timestamp

    outputs = []

    # -------------------------------------------------
    # 1) National calendar-year precipitation (WGS84)
    # -------------------------------------------------
    reg = REGIONS[0]
    region_df_year = bbox_filter(data_year, reg["lon_min"], reg["lon_max"], reg["lat_min"], reg["lat_max"])
    buf = build_precip_map_buffer_wgs(
        region_df=region_df_year,
        reg=reg,
        timestamp_text=timestamp_text,
        strict_value_col="total_precipitation",
        top_value_col="total_precipitation_missing",
        title_text=reg["title"],
        cbar_label="Υπολογ. σωρευτικός υετός έτους (mm)"
    )
    if buf is not None:
        outputs.append(("precip", reg["outfile"], buf))

    # -------------------------------------------------
    # 2) National yearly mean temperature (WGS84)
    # -------------------------------------------------
    buf = build_yearly_tavg_greece_buffer(region_df_year, reg, greece, athens_now, timestamp_text)
    if buf is not None:
        outputs.append(("temp", reg["temp_outfile"], buf))

    # -------------------------------------------------
    # 3) Regional calendar-year precipitation (EGSA87)
    # -------------------------------------------------
    for reg in REGIONS[1:]:
        region_df_year = bbox_filter(data_year, reg["lon_min"], reg["lon_max"], reg["lat_min"], reg["lat_max"])
        buf = build_precip_region_egsa_buffer(
            region_df=region_df_year,
            reg=reg,
            timestamp_text=timestamp_text,
            strict_value_col="total_precipitation",
            top_value_col="total_precipitation_missing",
            title_text=reg["title"],
            cbar_label="Υπολογ. σωρευτικός υετός έτους (mm)"
        )
        if buf is not None:
            outputs.append(("precip", reg["outfile"], buf))

    # -------------------------------------------------
    # 4) Regional yearly mean temperature (EGSA87)
    # -------------------------------------------------
    for reg in REGIONS[1:]:
        region_df_year = bbox_filter(data_year, reg["lon_min"], reg["lon_max"], reg["lat_min"], reg["lat_max"])
        buf = build_yearly_tavg_region_egsa_buffer(region_df_year, reg, greece, athens_now, timestamp_text)
        if buf is not None:
            outputs.append(("temp", reg["temp_outfile"], buf))

    # -------------------------------------------------
    # 5) National hydrological-year precipitation (WGS84)
    # -------------------------------------------------
    reg = REGIONS[0]
    region_df_hydro = bbox_filter(data_hydro, reg["lon_min"], reg["lon_max"], reg["lat_min"], reg["lat_max"])
    buf = build_precip_map_buffer_wgs(
        region_df=region_df_hydro,
        reg=reg,
        timestamp_text=timestamp_text,
        strict_value_col="hydro_year_precipitation",
        top_value_col="hydro_year_precipitation_missing",
        title_text=reg["hydro_title"],
        cbar_label="Υπολογ. σωρευτικός υετός υδρολογικού έτους (mm)"
    )
    if buf is not None:
        outputs.append(("hydro_precip", reg["hydro_outfile"], buf))

    # -------------------------------------------------
    # 6) Regional hydrological-year precipitation (EGSA87)
    # -------------------------------------------------
    for reg in REGIONS[1:]:
        region_df_hydro = bbox_filter(data_hydro, reg["lon_min"], reg["lon_max"], reg["lat_min"], reg["lat_max"])
        buf = build_precip_region_egsa_buffer(
            region_df=region_df_hydro,
            reg=reg,
            timestamp_text=timestamp_text,
            strict_value_col="hydro_year_precipitation",
            top_value_col="hydro_year_precipitation_missing",
            title_text=reg["hydro_title"],
            cbar_label="Υπολογ. σωρευτικός υετός υδρολογικού έτους (mm)"
        )
        if buf is not None:
            outputs.append(("hydro_precip", reg["hydro_outfile"], buf))

    if not outputs:
        raise RuntimeError("No output maps were generated.")

    ftps = None
    try:
        ftps = ftp_connect()
        print("✅ FTPS session opened once for all uploads.")

        for kind, filename, file_buffer in outputs:
            ok = upload_via_session(ftps, file_buffer, filename)
            if ok:
                if kind == "precip":
                    print("✅ Region yearly precipitation map uploaded:", filename)
                elif kind == "hydro_precip":
                    print("✅ Region hydrological-year precipitation map uploaded:", filename)
                else:
                    print("✅ Region yearly tavg map uploaded:", filename)
            else:
                if kind == "precip":
                    print("❌ Region yearly precipitation map NOT uploaded:", filename)
                elif kind == "hydro_precip":
                    print("❌ Region hydrological-year precipitation map NOT uploaded:", filename)
                else:
                    print("❌ Region yearly tavg map NOT uploaded:", filename)

    finally:
        if ftps is not None:
            try:
                ftps.quit()
            except Exception:
                try:
                    ftps.close()
                except Exception:
                    pass

    print("✅ Done (calendar-year precipitation + yearly average temperature + hydrological-year precipitation, all regions processed).")


if __name__ == "__main__":
    main()
