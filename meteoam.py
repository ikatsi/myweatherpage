#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
meteoam.py

Greece-wide precipitation-type maps using:
- H-SAF H60B precipitation rate (external source)
- station-based wet-bulb temperature from CURRENTWEATHER_URL + DEM

Outputs are written locally to:
    ./ptype_hsaf_greece/

Remote FTP upload:
- uploads PNG basenames only
- no remote subfolders are created or used by this script

Expected existing secrets/env:
    CURRENTWEATHER_URL
    GEOJSON_PASS
    BRAND_NAME
    FTP_HOST
    FTP_USER
    FTP_PASS
    HSAF_HOST
    HSAF_USER
    HSAF_PASS
    HSAF_REMOTE_DIR   -> should now be h60/h60_cur_mon_data
"""

import os
import re
import io
import gzip
import time
import math
import shutil
import random
import socket
import zipfile
import tempfile
import subprocess
from io import StringIO
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from ftplib import FTP, FTP_TLS

import numpy as np
import pandas as pd
import geopandas as gpd
import numpy.ma as ma

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial import cKDTree
from scipy.ndimage import zoom
from scipy.interpolate import LinearNDInterpolator

import rasterio
from rasterio.warp import transform as rio_transform
from pyproj import Transformer, CRS
import requests

from netCDF4 import Dataset


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATHENS_TZ = ZoneInfo("Europe/Athens")
UTC = ZoneInfo("UTC")

CURRENTWEATHER_URL = os.environ.get("CURRENTWEATHER_URL", "").strip()
if not CURRENTWEATHER_URL:
    raise SystemExit("❌ CURRENTWEATHER_URL secret/env not set.")

BRAND_NAME = os.environ.get("BRAND_NAME", "").strip() or "e-kairos.gr"

# Existing encrypted assets
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()
GREECE_GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
GREECE_GEOJSON_ENC  = os.path.join(BASE_DIR, "greece.geojson.enc")
ALT_VRT_PATH        = os.path.join(BASE_DIR, "GRC_alt.vrt")
ALT_ENC             = os.path.join(BASE_DIR, "altitude.zip.enc")
ALT_ZIP             = os.path.join(BASE_DIR, "altitude.zip")

# Existing optional FTP upload
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()

# H-SAF
HSAF_HOST = os.environ.get("HSAF_HOST", "").strip() or "ftphsaf.meteoam.it"
HSAF_USER = os.environ.get("HSAF_USER", "").strip()
HSAF_PASS = os.environ.get("HSAF_PASS", "").strip()
HSAF_REMOTE_DIR = os.environ.get("HSAF_REMOTE_DIR", "").strip() or "h60/h60_cur_mon_data"

if not HSAF_USER or not HSAF_PASS:
    raise SystemExit("❌ HSAF_USER / HSAF_PASS not set.")

# Same EXACT Greece bbox as your existing Greece script
GRID_N = 300
GRID_LON_MIN, GRID_LON_MAX = 19.0, 30.0
GRID_LAT_MIN, GRID_LAT_MAX = 34.5, 42.5

# Greece outline is only for reference, not clipping
SHOW_GREECE_OUTLINE = True

# Wet-bulb thresholds chosen by you
TW_SNOW_MAX = 0.5
TW_MIXED_MAX = 1.5

# Time windows
WEATHER_TIME_WINDOW_MIN = 60
HSAF_MAX_AGE_MIN = 120  # fail only if very stale

# Tw lapse regression controls
LAPSE_DEFAULT = -0.0055
LAPSE_MIN = -0.0100
LAPSE_MAX = -0.0010

TEMP_COARSE_N = 120
K_LOCAL = 25
R_LOCAL_M = 150_000
ALT_RANGE_MIN_M = 400
MIN_NBR = 8
USE_DISTANCE_WEIGHTS = True

# H60B filename pattern from PUM
H60_FILE_RE = re.compile(r"^h60_\d{8}_\d{4}_fdk\.nc\.gz$", re.IGNORECASE)

# Quality filtering
MIN_QIND = 1.0
MIN_RR_TO_PLOT = 0.1

# Local output/cache dirs
OUTPUT_DIR = os.path.join(BASE_DIR, "ptype_hsaf_greece")
CACHE_DIR = os.path.join(BASE_DIR, "hsaf_cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Rain-rate colormap
PRECIP_CMAP = ListedColormap([
    "#f7fbff",
    "#deebf7",
    "#9ecae1",
    "#4292c6",
    "#2171b5",
    "#084594",
    "#6a51a3",
    "#ce1256",
])
PRECIP_CMAP.set_under("#ffffff")
PRECIP_CMAP.set_bad("#ffffff")
PRECIP_BOUNDS = [0.1, 0.5, 1, 2, 5, 10, 20, 40, 100]
PRECIP_NORM = BoundaryNorm(PRECIP_BOUNDS, PRECIP_CMAP.N)

# Phase colormap
# 0 rain, 1 mixed, 2 snow
PHASE_CMAP = ListedColormap([
    "#3182bd",
    "#9e9ac8",
    "#f16913",
])
PHASE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], PHASE_CMAP.N)


# =============================================================================
# SMALL HELPERS
# =============================================================================
def ftp_enabled():
    return bool(FTP_HOST and FTP_USER and FTP_PASS)

def transparent_bbox(pad=0.3, rounded=True):
    boxstyle = ("round,pad=" + str(pad)) if rounded else ("square,pad=" + str(pad))
    return dict(
        facecolor=(1, 1, 1, 0.0),
        edgecolor=(0, 0, 0, 0.0),
        boxstyle=boxstyle,
    )

def athens_abbrev(dt: datetime) -> str:
    try:
        dt_ath = dt.astimezone(ATHENS_TZ)
        is_dst = bool(dt_ath.dst()) and dt_ath.dst() != timedelta(0)
        return "EEST" if is_dst else "EET"
    except Exception:
        return "EET"

def fmt_data_until(dtval) -> str:
    if dtval is None or (isinstance(dtval, float) and np.isnan(dtval)):
        return "—"
    try:
        ts = pd.to_datetime(dtval)
        if getattr(ts, "tzinfo", None) is None and getattr(ts, "tz", None) is None:
            ts = ts.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT")
        else:
            ts = ts.tz_convert("Europe/Athens")
        if pd.isna(ts):
            return "—"
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        try:
            return dtval.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "—"

def parse_h60_time_from_name(fname: str):
    m = re.match(r"^h60_(\d{8})_(\d{4})_fdk\.nc\.gz$", os.path.basename(fname), re.IGNORECASE)
    if not m:
        return None
    d, hm = m.groups()
    return datetime.strptime(d + hm, "%Y%m%d%H%M").replace(tzinfo=UTC)


# =============================================================================
# ENCRYPTED ASSET HELPERS
# =============================================================================
def _openssl_decrypt(enc_path: str, out_path: str, password: str):
    try:
        subprocess.check_call([
            "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
            "-in", enc_path, "-out", out_path,
            "-pass", "pass:" + password
        ])
    except FileNotFoundError:
        raise SystemExit("❌ OpenSSL not found.")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"❌ OpenSSL decryption failed for {os.path.basename(enc_path)}: {e}")

def ensure_geojson_and_altitude_bundle():
    if not os.path.exists(GREECE_GEOJSON_PATH) and os.path.exists(GREECE_GEOJSON_ENC):
        if not GEOJSON_PASS:
            raise SystemExit("❌ greece.geojson missing and GEOJSON_PASS not set.")
        _openssl_decrypt(GREECE_GEOJSON_ENC, GREECE_GEOJSON_PATH, GEOJSON_PASS)

    if os.path.exists(ALT_VRT_PATH):
        return

    if not os.path.exists(ALT_ENC):
        raise SystemExit("❌ GRC_alt.vrt missing and altitude.zip.enc not found.")

    if not GEOJSON_PASS:
        raise SystemExit("❌ DEM bundle missing and GEOJSON_PASS not set.")

    _openssl_decrypt(ALT_ENC, ALT_ZIP, GEOJSON_PASS)

    try:
        with zipfile.ZipFile(ALT_ZIP, "r") as zf:
            zf.extractall(BASE_DIR)
    finally:
        try:
            os.remove(ALT_ZIP)
        except Exception:
            pass

    if not os.path.exists(ALT_VRT_PATH):
        raise SystemExit("❌ Decrypted DEM bundle did not produce GRC_alt.vrt.")


# =============================================================================
# WEATHER FEED FETCH / CLEAN
# =============================================================================
def robust_fetch_text(url: str, cache_txt: str, timeout: int = 60, tries: int = 6):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/plain,text/*;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }

    if url.startswith("file://"):
        path = url[7:]
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(), "localfile"

    if os.path.exists(url):
        with open(url, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "localfile"

    session = requests.Session()
    last_err = None

    for attempt in range(1, tries + 1):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            text = r.text or ""
            if not text.strip():
                raise RuntimeError("Empty response body.")
            first_line = text.splitlines()[0].strip()
            looks_like_tsv = ("\t" in first_line) and (len(first_line) >= 10)
            if ("Datetime" not in text) and (not looks_like_tsv):
                raise RuntimeError("Response does not look like expected TSV.")
            return text, "network"
        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 20) + random.random()
            print(f"[weathernow] attempt {attempt}/{tries} failed: {e}. Retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)

    if cache_txt and os.path.exists(cache_txt):
        print(f"[weathernow] using cache: {cache_txt}")
        with open(cache_txt, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "cache"

    raise RuntimeError("Failed to fetch weather feed") from last_err

def load_and_clean_weather_feed(text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text), delimiter="\t", engine="python")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    needed = ["Datetime", "TNow", "RHNow", "Latitude", "Longitude", "Country", "webcode"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError("Missing columns in weather feed: " + ", ".join(missing))

    for c in ["TNow", "RHNow", "Latitude", "Longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    dt = df["Datetime"]
    try:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT")
        else:
            dt = dt.dt.tz_convert("Europe/Athens")
    except Exception:
        pass
    df["Datetime"] = dt

    df = df.dropna(subset=["Datetime", "TNow", "RHNow", "Latitude", "Longitude"]).copy()
    df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()

    c = df["Country"].astype(str).str.strip().str.upper()
    df = df[c.isin(["GR", "GREECE"])].copy()

    # loose geographic guard
    df = df[
        df["Longitude"].between(GRID_LON_MIN - 2.0, GRID_LON_MAX + 2.0) &
        df["Latitude"].between(GRID_LAT_MIN - 2.0, GRID_LAT_MAX + 2.0)
    ].copy()

    athens_now = datetime.now(ATHENS_TZ)
    threshold = athens_now - timedelta(minutes=WEATHER_TIME_WINDOW_MIN)
    df = df[df["Datetime"] >= threshold].copy()

    if df.empty:
        raise RuntimeError("No valid recent Greek weather observations available.")

    return df


# =============================================================================
# WET BULB
# =============================================================================
def wet_bulb_stull(t_c, rh_pct):
    t = np.asarray(t_c, dtype=float)
    rh = np.asarray(rh_pct, dtype=float)
    rh = np.clip(rh, 1.0, 100.0)

    tw = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw


# =============================================================================
# DEM / ALTITUDE
# =============================================================================
def sample_altitude_vrt_m(vrt_path: str, lons, lats) -> np.ndarray:
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    with rasterio.open(vrt_path) as src:
        if src.crs is None:
            raise RuntimeError("Altitude VRT has no CRS defined.")
        if src.crs.to_string() != "EPSG:4326":
            xs, ys = rio_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        else:
            xs, ys = lons.tolist(), lats.tolist()

        samples = list(src.sample(zip(xs, ys)))
        arr = np.array(samples, dtype=float).reshape(-1)

        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        arr = np.where(np.isfinite(arr), arr, np.nan)

    return arr


# =============================================================================
# LOCAL REGRESSION FOR WET BULB FIELD
# =============================================================================
def fit_global_lapse_rate(var_c: np.ndarray, alt_m: np.ndarray) -> float:
    mask = np.isfinite(var_c) & np.isfinite(alt_m)
    if mask.sum() < 8:
        return LAPSE_DEFAULT
    x = alt_m[mask]
    y = var_c[mask]
    b, _a = np.polyfit(x, y, 1)
    if not (LAPSE_MIN <= b <= LAPSE_MAX):
        return LAPSE_DEFAULT
    return float(b)

def build_variable_grid_local_lr_wgs(
    lon_min, lon_max, lat_min, lat_max,
    grid_lon, grid_lat,
    station_lons, station_lats,
    station_var, station_alt_m,
    alt_vrt_path,
    grid_n,
):
    to_egsa = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)
    st_x, st_y = to_egsa.transform(station_lons.tolist(), station_lats.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    st_v = np.asarray(station_var, dtype=float)
    st_alt = np.asarray(station_alt_m, dtype=float)

    ok = np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_v) & np.isfinite(st_alt)
    st_x, st_y, st_v, st_alt = st_x[ok], st_y[ok], st_v[ok], st_alt[ok]

    if len(st_v) < 10:
        return np.full(grid_lon.shape, np.nan, dtype=float), LAPSE_DEFAULT

    b_global = fit_global_lapse_rate(st_v, st_alt)
    v0_global = np.nanmedian(st_v - b_global * st_alt)

    c_lon = np.linspace(lon_min, lon_max, TEMP_COARSE_N)
    c_lat = np.linspace(lat_min, lat_max, TEMP_COARSE_N)
    c_grid_lon, c_grid_lat = np.meshgrid(c_lon, c_lat)

    c_alt = sample_altitude_vrt_m(alt_vrt_path, c_grid_lon.ravel(), c_grid_lat.ravel()).reshape(c_grid_lon.shape)

    c_x, c_y = to_egsa.transform(c_grid_lon.ravel().tolist(), c_grid_lat.ravel().tolist())
    c_x = np.asarray(c_x, dtype=float)
    c_y = np.asarray(c_y, dtype=float)

    tree = cKDTree(np.c_[st_x, st_y])
    dists, idx = tree.query(
        np.c_[c_x, c_y],
        k=min(K_LOCAL, len(st_v)),
        distance_upper_bound=R_LOCAL_M
    )

    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    pred = np.full(c_x.shape[0], np.nan, dtype=float)

    for i in range(c_x.shape[0]):
        di = dists[i]
        ii = idx[i]
        m = np.isfinite(di) & (ii < len(st_v))
        if not np.any(m):
            continue
        di = di[m]
        ii = ii[m]
        v = st_v[ii]
        a = st_alt[ii]

        m2 = np.isfinite(v) & np.isfinite(a)
        if m2.sum() < MIN_NBR:
            continue
        v = v[m2]
        a = a[m2]
        di = di[m2]

        if (np.nanmax(a) - np.nanmin(a)) < ALT_RANGE_MIN_M:
            continue

        try:
            if USE_DISTANCE_WEIGHTS:
                w = 1.0 / (di + 2000.0)
                b, intercept = np.polyfit(a, v, 1, w=w)
            else:
                b, intercept = np.polyfit(a, v, 1)
        except Exception:
            continue

        if not (LAPSE_MIN <= b <= LAPSE_MAX):
            continue

        alt_here = c_alt.ravel()[i]
        if not np.isfinite(alt_here):
            continue

        pred[i] = intercept + b * alt_here

    c_alt_flat = c_alt.ravel()
    fallback_var = v0_global + b_global * c_alt_flat
    pred = np.where(np.isfinite(pred), pred, fallback_var)

    pred_coarse = pred.reshape(c_grid_lon.shape)

    zoom_factor = grid_n / float(TEMP_COARSE_N)
    var_fine = zoom(pred_coarse, zoom=(zoom_factor, zoom_factor), order=1)

    if var_fine.shape != grid_lon.shape:
        out = np.full(grid_lon.shape, np.nan, dtype=float)
        h = min(out.shape[0], var_fine.shape[0])
        w = min(out.shape[1], var_fine.shape[1])
        out[:h, :w] = var_fine[:h, :w]
        var_fine = out

    return var_fine, b_global


# =============================================================================
# H-SAF FTP / NETCDF
# =============================================================================
def ftp_connect_hsaf(host, user, passwd, attempts=5, timeout=90):
    last_err = None
    for i in range(attempts):
        try:
            ftp = FTP()
            ftp.connect(host, 21, timeout=timeout)
            ftp.login(user=user, passwd=passwd)
            ftp.set_pasv(True)
            return ftp
        except Exception as e:
            last_err = e
            sleep_s = min(3 * (2 ** i), 20)
            print(f"[hsaf] FTP connect failed ({e}), retry in {sleep_s}s...")
            time.sleep(sleep_s)
    raise last_err


def hsaf_list_latest_h60_file():
    ftp = ftp_connect_hsaf(HSAF_HOST, HSAF_USER, HSAF_PASS)
    try:
        ftp.cwd(HSAF_REMOTE_DIR)
        names = ftp.nlst()
        basenames = [os.path.basename(n) for n in names if n]
        files = [n for n in basenames if H60_FILE_RE.match(n)]
        if not files:
            raise RuntimeError(f"No H60 files found in {HSAF_REMOTE_DIR}")
        files.sort()
        return files[-1]
    finally:
        try:
            ftp.quit()
        except Exception:
            pass


def hsaf_download_file(remote_dir: str, remote_name: str, local_path: str):
    ftp = ftp_connect_hsaf(HSAF_HOST, HSAF_USER, HSAF_PASS)
    try:
        ftp.cwd(remote_dir)
        with open(local_path, "wb") as f:
            ftp.retrbinary("RETR " + remote_name, f.write)
        print(f"[hsaf] downloaded {remote_name}")
    finally:
        try:
            ftp.quit()
        except Exception:
            pass


def _find_var_name(ds, candidates):
    vars_lower = {name.lower(): name for name in ds.variables.keys()}
    for cand in candidates:
        if cand.lower() in vars_lower:
            return vars_lower[cand.lower()]
    return None


def _find_geos_var(ds):
    for name, var in ds.variables.items():
        try:
            gmn = getattr(var, "grid_mapping_name", None)
            if gmn and str(gmn).lower() == "geostationary":
                return name
        except Exception:
            pass

    for name, var in ds.variables.items():
        attrs = getattr(var, "ncattrs", lambda: [])()
        if "perspective_point_height" in attrs or "longitude_of_projection_origin" in attrs:
            return name

    return None


def _build_lonlat_from_geos(ds):
    """
    Derive lon/lat from geostationary projection metadata in the H60 file.

    Strategy:
    1) Use explicit x/y coordinate variables if they exist.
    2) Otherwise reconstruct x/y from MSG navigation coefficients
       (CFAC/LFAC/COFF/LOFF style attributes), using the rr array shape.
    """
    geos_name = _find_geos_var(ds)
    if geos_name is None:
        raise RuntimeError("Could not find geostationary projection metadata variable in H60 NetCDF.")

    gvar = ds.variables[geos_name]

    lon_0 = getattr(gvar, "longitude_of_projection_origin", None)
    h = getattr(gvar, "perspective_point_height", None)
    sweep = getattr(gvar, "sweep_angle_axis", "y")
    a = getattr(gvar, "semi_major_axis", None)
    b = getattr(gvar, "semi_minor_axis", None)

    if lon_0 is None or h is None:
        raise RuntimeError("Missing longitude_of_projection_origin and/or perspective_point_height in H60 NetCDF.")

    x_name = _find_var_name(ds, ["x", "xc", "nx"])
    y_name = _find_var_name(ds, ["y", "yc", "ny"])

    # ------------------------------------------------------------------
    # Case 1: explicit x/y variables exist
    # ------------------------------------------------------------------
    if x_name is not None and y_name is not None:
        x = np.array(ds.variables[x_name][:], dtype=float)
        y = np.array(ds.variables[y_name][:], dtype=float)

        x_units = str(getattr(ds.variables[x_name], "units", "")).lower()
        y_units = str(getattr(ds.variables[y_name], "units", "")).lower()

        x_is_angle = ("rad" in x_units) or (np.nanmax(np.abs(x)) < 1.0)
        y_is_angle = ("rad" in y_units) or (np.nanmax(np.abs(y)) < 1.0)

        if x_is_angle:
            x = x * float(h)
        if y_is_angle:
            y = y * float(h)

        xx, yy = np.meshgrid(x, y)

    # ------------------------------------------------------------------
    # Case 2: no x/y variables, reconstruct from CFAC/LFAC/COFF/LOFF
    # ------------------------------------------------------------------
    else:
        rr_var = ds.variables["rr"]
        if rr_var.ndim < 2:
            raise RuntimeError("rr variable is not 2D, cannot reconstruct H60 geostationary grid.")

        nlines, ncols = rr_var.shape[-2], rr_var.shape[-1]

        def _get_attr_any(obj, names, default=None):
            for nm in names:
                if hasattr(obj, nm):
                    return getattr(obj, nm)
            return default

        # Try projection-variable attrs first, then global attrs
        cfac = _get_attr_any(gvar, ["cfac", "CFAC"], None)
        lfac = _get_attr_any(gvar, ["lfac", "LFAC"], None)
        coff = _get_attr_any(gvar, ["coff", "COFF"], None)
        loff = _get_attr_any(gvar, ["loff", "LOFF"], None)

        if cfac is None:
            cfac = getattr(ds, "cfac", getattr(ds, "CFAC", None))
        if lfac is None:
            lfac = getattr(ds, "lfac", getattr(ds, "LFAC", None))
        if coff is None:
            coff = getattr(ds, "coff", getattr(ds, "COFF", None))
        if loff is None:
            loff = getattr(ds, "loff", getattr(ds, "LOFF", None))

        if None in (cfac, lfac, coff, loff):
            raise RuntimeError(
                "Could not find x/y variables or CFAC/LFAC/COFF/LOFF navigation metadata in H60 NetCDF."
            )
    
        cfac = float(cfac)
        lfac = float(lfac)
        coff = float(coff)
        loff = float(loff)

        cols = np.arange(ncols, dtype=float)
        lines = np.arange(nlines, dtype=float)

        # MSG navigation coefficients produce scan angles in radians
        x_ang = (cols - coff) * (2.0 ** 16) / cfac
        y_ang = (lines - loff) * (2.0 ** 16) / lfac

        # convert scan angles to projected metres for pyproj geos
        x = x_ang * float(h)
        y = y_ang * float(h)

        xx, yy = np.meshgrid(x, y)

    crs_geos = CRS.from_proj4(
        f"+proj=geos +lon_0={float(lon_0)} +h={float(h)} "
        f"+a={float(a) if a is not None else 6378169.0} "
        f"+b={float(b) if b is not None else 6356583.8} "
        f"+sweep={sweep} +units=m +no_defs"
    )
    transformer = Transformer.from_crs(crs_geos, "EPSG:4326", always_xy=True)
    lon2d, lat2d = transformer.transform(xx, yy)

    lon2d = np.asarray(lon2d, dtype=float)
    lat2d = np.asarray(lat2d, dtype=float)

    lon2d[~np.isfinite(lon2d)] = np.nan
    lat2d[~np.isfinite(lat2d)] = np.nan

    return lat2d, lon2d


def open_h60_netcdf_from_gz(gz_path: str):
    """
    Returns:
        rr, qind, lat2d, lon2d
    """
    tmp_nc = tempfile.NamedTemporaryFile(prefix="h60_", suffix=".nc", delete=False)
    tmp_nc_path = tmp_nc.name
    tmp_nc.close()

    ds = None
    try:
        with gzip.open(gz_path, "rb") as f_in, open(tmp_nc_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        ds = Dataset(tmp_nc_path, "r")

        if "rr" not in ds.variables or "qind" not in ds.variables:
            raise RuntimeError("H60 NetCDF missing rr and/or qind variables.")

        v_rr = ds.variables["rr"]
        v_qi = ds.variables["qind"]

        rr_raw = np.array(v_rr[:], dtype=float)
        qind_raw = np.array(v_qi[:], dtype=float)

        rr_missing = getattr(v_rr, "missing_value", -99)
        qi_missing = getattr(v_qi, "missing_value", -99)
        rr_scale = getattr(v_rr, "scale_factor", 1.0)
        rr_offset = getattr(v_rr, "add_offset", 0.0)

        rr = np.where(rr_raw == rr_missing, np.nan, rr_raw)
        rr = rr * float(rr_scale) + float(rr_offset)

        qind = np.where(qind_raw == qi_missing, np.nan, qind_raw)

        # Prefer explicit lat/lon if present
        lat_name = _find_var_name(ds, ["lat", "latitude"])
        lon_name = _find_var_name(ds, ["lon", "longitude"])

        if lat_name is not None and lon_name is not None:
            latv = np.array(ds.variables[lat_name][:], dtype=float)
            lonv = np.array(ds.variables[lon_name][:], dtype=float)

            if latv.ndim == 1 and lonv.ndim == 1:
                lon2d, lat2d = np.meshgrid(lonv, latv)
            elif latv.ndim == 2 and lonv.ndim == 2:
                lat2d, lon2d = latv, lonv
            else:
                raise RuntimeError("Unsupported lat/lon dimensions in H60 NetCDF.")
        else:
            lat2d, lon2d = _build_lonlat_from_geos(ds)

        return rr, qind, lat2d, lon2d

    finally:
        try:
            if ds is not None:
                ds.close()
        except Exception:
            pass
        try:
            os.remove(tmp_nc_path)
        except Exception:
            pass

# =============================================================================
# OPTIONAL UPLOAD HELPERS
# =============================================================================
def ftps_connect_with_retries(host, user, passwd, attempts=6, base_sleep=5, timeout=60):
    last_err = None
    for i in range(attempts):
        try:
            ftps = FTP_TLS()
            ftps.connect(host, 21, timeout=timeout)
            ftps.login(user=user, passwd=passwd)
            ftps.prot_p()
            ftps.set_pasv(True)
            return ftps
        except (socket.gaierror, OSError) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** i)
            print(f"⚠️ FTPS connect failed ({type(e).__name__}: {e}). Retry in {sleep_s}s...")
            time.sleep(sleep_s)
    raise last_err

def ftp_upload_file(local_file: str, timeout: int = 60):
    if not ftp_enabled():
        print("ℹ️ FTP disabled. Skipping upload.")
        return

    remote_filename = os.path.basename(local_file)
    ftps = ftps_connect_with_retries(FTP_HOST, FTP_USER, FTP_PASS, attempts=6, base_sleep=5, timeout=timeout)

    try:
        with open(local_file, "rb") as f:
            # Upload to current remote directory only, no subfolders
            ftps.storbinary("STOR " + remote_filename, f)
        print(f"📤 Uploaded: {remote_filename}")
    finally:
        try:
            ftps.quit()
        except Exception:
            pass


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def draw_greece_outline(ax):
    if not SHOW_GREECE_OUTLINE:
        return
    if not os.path.exists(GREECE_GEOJSON_PATH):
        return
    try:
        greece = gpd.read_file(GREECE_GEOJSON_PATH)
        if greece.crs is None:
            greece = greece.set_crs("EPSG:4326")
        if greece.crs.to_string() != "EPSG:4326":
            greece = greece.to_crs("EPSG:4326")
        greece.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.8)
    except Exception as e:
        print(f"⚠️ Could not draw Greece outline: {e}")

def common_footer(ax, created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse):
    timestamp_text = created_dt_ath.strftime("%Y-%m-%d %H:%M") + f" {athens_abbrev(created_dt_ath)}"
    hsaf_time = h60_dt_utc.strftime("%Y-%m-%d %H:%M UTC") if h60_dt_utc else "—"

    left_text = (
        f"Δημιουργήθηκε για το {BRAND_NAME}\n"
        f"{timestamp_text}\n"
        f"weathernow έως: {weather_until}\n"
        f"Copyright {created_dt_ath.year} EUMETSAT"
    )
    right_text = (
        f"H-SAF H60B\n"
        f"{hsaf_time}\n"
        f"Tw lapse: {tw_global_lapse*1000:.2f} °C/km"
    )

    ax.text(
        0.01, 0.01, left_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left", va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )
    ax.text(
        0.99, 0.01, right_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="right", va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )

def save_phase_map(
    out_path,
    lon2d, lat2d, phase_idx, rr2d,
    created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse
):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    rr_for_contour = np.where(np.isfinite(rr2d) & (rr2d >= MIN_RR_TO_PLOT), rr2d, np.nan)

    pcm = ax.pcolormesh(
        lon2d, lat2d, phase_idx,
        cmap=PHASE_CMAP,
        norm=PHASE_NORM,
        shading="auto",
        alpha=0.78
    )

    try:
        ax.contour(
            lon2d, lat2d, rr_for_contour,
            levels=[0.1, 0.5, 1, 2, 5, 10, 20],
            colors="black",
            linewidths=0.6
        )
    except Exception:
        pass

    draw_greece_outline(ax)

    handles = [
        Patch(facecolor="#3182bd", edgecolor="black", label="Βροχή πιθανή"),
        Patch(facecolor="#9e9ac8", edgecolor="black", label="Μικτός / sleet πιθανός"),
        Patch(facecolor="#f16913", edgecolor="black", label="Χιόνι πιθανό"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)

    ax.set_xlim(GRID_LON_MIN, GRID_LON_MAX)
    ax.set_ylim(GRID_LAT_MIN, GRID_LAT_MAX)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)
    ax.set_title("Πιθανή φάση υετού με βάση H-SAF H60B + Tw", fontsize=14, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    common_footer(ax, created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")

def save_rr_map(
    out_path,
    title,
    lon2d, lat2d, rr2d,
    created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse
):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    arr = ma.masked_invalid(rr2d)

    img = ax.pcolormesh(
        lon2d, lat2d, arr,
        cmap=PRECIP_CMAP,
        norm=PRECIP_NORM,
        shading="auto"
    )

    try:
        ax.contour(
            lon2d, lat2d, np.where(np.isfinite(rr2d) & (rr2d >= MIN_RR_TO_PLOT), rr2d, np.nan),
            levels=[0.1, 0.5, 1, 2, 5, 10, 20],
            colors="black",
            linewidths=0.6
        )
    except Exception:
        pass

    draw_greece_outline(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax, boundaries=PRECIP_BOUNDS, extend="max")
    cbar.set_label("H-SAF H60B precipitation rate (mm/h)", fontsize=11)

    ax.set_xlim(GRID_LON_MIN, GRID_LON_MAX)
    ax.set_ylim(GRID_LAT_MIN, GRID_LAT_MAX)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    common_footer(ax, created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_geojson_and_altitude_bundle()

    # -------------------------------------------------------------------------
    # 1) Latest H60B file
    # -------------------------------------------------------------------------
    latest_remote = hsaf_list_latest_h60_file()
    h60_dt_utc = parse_h60_time_from_name(latest_remote)

    if h60_dt_utc is not None:
        age_min = (datetime.now(UTC) - h60_dt_utc).total_seconds() / 60.0
        if age_min > HSAF_MAX_AGE_MIN:
            print(f"⚠️ Latest H60B file is older than expected: {latest_remote} ({age_min:.0f} min old)")

    local_gz = os.path.join(CACHE_DIR, latest_remote)
    hsaf_download_file(HSAF_REMOTE_DIR, latest_remote, local_gz)

    rr_full, qind_full, lat_full, lon_full = open_h60_netcdf_from_gz(local_gz)

    if rr_full.shape != lat_full.shape or rr_full.shape != lon_full.shape:
        if rr_full.T.shape == lat_full.shape:
            rr_full = rr_full.T
            qind_full = qind_full.T
        else:
            raise RuntimeError(
                f"Shape mismatch after H60 read: rr={rr_full.shape}, lat={lat_full.shape}, lon={lon_full.shape}"
            )

    # -------------------------------------------------------------------------
    # 2) Weather feed and Tw field
    # -------------------------------------------------------------------------
    cache_txt = os.path.join(CACHE_DIR, "weathernow_cached.txt")
    weather_text, source = robust_fetch_text(CURRENTWEATHER_URL, cache_txt=cache_txt, timeout=60, tries=6)
    with open(cache_txt, "w", encoding="utf-8") as f:
        f.write(weather_text)

    wdf = load_and_clean_weather_feed(weather_text)
    weather_until = fmt_data_until(wdf["Datetime"].max())

    wdf["Tw"] = wet_bulb_stull(wdf["TNow"].values, wdf["RHNow"].values)

    station_lons = wdf["Longitude"].values.astype(float)
    station_lats = wdf["Latitude"].values.astype(float)
    station_tw = wdf["Tw"].values.astype(float)
    station_alt = sample_altitude_vrt_m(ALT_VRT_PATH, station_lons, station_lats)

    # Regular Greece grid for Tw interpolation
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_N),
        np.linspace(GRID_LAT_MIN, GRID_LAT_MAX, GRID_N)
    )

    tw_grid, tw_global_lapse = build_variable_grid_local_lr_wgs(
        GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX,
        grid_lon, grid_lat,
        station_lons, station_lats,
        station_tw, station_alt,
        ALT_VRT_PATH,
        grid_n=GRID_N
    )

    # -------------------------------------------------------------------------
    # 3) Subset H60B over Greece bbox, no clipping to land
    # -------------------------------------------------------------------------
    bbox_mask = (
        np.isfinite(lat_full) &
        np.isfinite(lon_full) &
        lon_full >= GRID_LON_MIN & lon_full <= GRID_LON_MAX &
        lat_full >= GRID_LAT_MIN & lat_full <= GRID_LAT_MAX
    )

    if not np.any(bbox_mask):
        raise RuntimeError("No H60B pixels found inside Greece bbox.")

    rows, cols = np.where(bbox_mask)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    lat_sub = lat_full[r0:r1, c0:c1]
    lon_sub = lon_full[r0:r1, c0:c1]
    rr_sub = rr_full[r0:r1, c0:c1]
    qi_sub = qind_full[r0:r1, c0:c1]

    # quality filtering
    rr_sub = np.where(np.isfinite(rr_sub), rr_sub, np.nan)
    qi_sub = np.where(np.isfinite(qi_sub), qi_sub, np.nan)

    valid_precip_mask = (
        np.isfinite(lat_sub) &
        np.isfinite(lon_sub) &
        np.isfinite(rr_sub) &
        np.isfinite(qi_sub) &
        (qi_sub >= MIN_QIND) &
        (rr_sub >= MIN_RR_TO_PLOT)
    )

    # -------------------------------------------------------------------------
    # 4) Interpolate Tw onto H60B pixels
    # -------------------------------------------------------------------------
    tw_interp = LinearNDInterpolator(
        np.column_stack([grid_lon.ravel(), grid_lat.ravel()]),
        tw_grid.ravel(),
        fill_value=np.nan
    )

    tw_sub = tw_interp(lon_sub, lat_sub)

    # -------------------------------------------------------------------------
    # 5) Phase classification
    # -------------------------------------------------------------------------
    phase_idx = np.full(rr_sub.shape, np.nan, dtype=float)

    rain_mask = valid_precip_mask & np.isfinite(tw_sub) & (tw_sub > TW_MIXED_MAX)
    mixed_mask = valid_precip_mask & np.isfinite(tw_sub) & (tw_sub > TW_SNOW_MAX) & (tw_sub <= TW_MIXED_MAX)
    snow_mask = valid_precip_mask & np.isfinite(tw_sub) & (tw_sub <= TW_SNOW_MAX)

    phase_idx[rain_mask] = 0
    phase_idx[mixed_mask] = 1
    phase_idx[snow_mask] = 2

    rr_rain = np.where(rain_mask, rr_sub, np.nan)
    rr_mixed = np.where(mixed_mask, rr_sub, np.nan)
    rr_snow = np.where(snow_mask, rr_sub, np.nan)

    # -------------------------------------------------------------------------
    # 6) Outputs
    # -------------------------------------------------------------------------
    created_dt_ath = datetime.now(ATHENS_TZ)
    ts = created_dt_ath.strftime("%Y-%m-%d-%H-%M")

    out_combined = os.path.join(OUTPUT_DIR, f"ptype_h60_combined_{ts}.png")
    out_rain = os.path.join(OUTPUT_DIR, f"ptype_h60_rain_{ts}.png")
    out_mixed = os.path.join(OUTPUT_DIR, f"ptype_h60_mixed_{ts}.png")
    out_snow = os.path.join(OUTPUT_DIR, f"ptype_h60_snow_{ts}.png")

    latest_combined = os.path.join(OUTPUT_DIR, "ptype_h60_combined_latest.png")
    latest_rain = os.path.join(OUTPUT_DIR, "ptype_h60_rain_latest.png")
    latest_mixed = os.path.join(OUTPUT_DIR, "ptype_h60_mixed_latest.png")
    latest_snow = os.path.join(OUTPUT_DIR, "ptype_h60_snow_latest.png")

    save_phase_map(
        out_combined,
        lon_sub, lat_sub, phase_idx, rr_sub,
        created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse
    )

    save_rr_map(
        out_rain,
        "H-SAF H60B precipitation rate όπου βροχή πιθανή (Tw > 1.5°C)",
        lon_sub, lat_sub, rr_rain,
        created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse
    )

    save_rr_map(
        out_mixed,
        "H-SAF H60B precipitation rate όπου μικτός / sleet πιθανός (0.5°C < Tw ≤ 1.5°C)",
        lon_sub, lat_sub, rr_mixed,
        created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse
    )

    save_rr_map(
        out_snow,
        "H-SAF H60B precipitation rate όπου χιόνι πιθανό (Tw ≤ 0.5°C)",
        lon_sub, lat_sub, rr_snow,
        created_dt_ath, weather_until, h60_dt_utc, tw_global_lapse
    )

    shutil.copy(out_combined, latest_combined)
    shutil.copy(out_rain, latest_rain)
    shutil.copy(out_mixed, latest_mixed)
    shutil.copy(out_snow, latest_snow)

    print(f"✅ Saved latest: {latest_combined}")
    print(f"✅ Saved latest: {latest_rain}")
    print(f"✅ Saved latest: {latest_mixed}")
    print(f"✅ Saved latest: {latest_snow}")

    # -------------------------------------------------------------------------
    # 7) Upload PNGs to current remote FTP folder only, no subfolders
    # -------------------------------------------------------------------------
    try:
        for fp in [
            out_combined, out_rain, out_mixed, out_snow,
            latest_combined, latest_rain, latest_mixed, latest_snow
        ]:
            ftp_upload_file(fp)
    except Exception as e:
        print(f"⚠️ FTP upload failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
