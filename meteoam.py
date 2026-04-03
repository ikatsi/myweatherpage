#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ptype_hsaf_h68_greece.py

New independent Greece-only product.

What it does:
- Downloads latest H-SAF H68 file from FTP
- Reads external quantitative precipitation rate (rr, mm/h)
- Reads your existing weathernow.txt feed from CURRENTWEATHER_URL
- Computes station wet-bulb temperature from TNow + RHNow
- Builds a Greece-wide wet-bulb grid using DEM-aware local lapse regression
- Classifies precipitation phase using:
    Tw <= 0.5°C          -> snow likely
    0.5 < Tw <= 1.5°C    -> mixed / sleet-favoured
    Tw > 1.5°C           -> rain likely
- Produces Greece maps on the EXACT same bbox as your existing Greece script:
    lon 19.0 to 30.0
    lat 34.5 to 42.5
- Uses EPSG:4326 like your existing Greece workflow
- DOES NOT clip to Greece polygon features
- Can optionally draw the Greece outline for reference
- Can optionally upload outputs by FTPS using your existing FTP_* secrets
"""

import os
import re
import io
import sys
import gzip
import time
import math
import json
import shutil
import random
import socket
import zipfile
import tempfile
import subprocess
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from ftplib import FTP, FTP_TLS, error_perm

import numpy as np
import pandas as pd
import geopandas as gpd
import numpy.ma as ma

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial import cKDTree
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator

import rasterio
from rasterio.warp import transform as rio_transform
from pyproj import Transformer
import requests

# Try xarray first, then netCDF4 fallback
XR_OK = False
NC4_OK = False
try:
    import xarray as xr
    XR_OK = True
except Exception:
    xr = None

try:
    from netCDF4 import Dataset
    NC4_OK = True
except Exception:
    Dataset = None


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

# New H-SAF secrets
HSAF_HOST = os.environ.get("HSAF_HOST", "").strip() or "ftphsaf.meteoam.it"
HSAF_USER = os.environ.get("HSAF_USER", "").strip()
HSAF_PASS = os.environ.get("HSAF_PASS", "").strip()
HSAF_REMOTE_DIR = os.environ.get("HSAF_REMOTE_DIR", "").strip() or "h68/h68_cur_mon_data"

if not HSAF_USER or not HSAF_PASS:
    raise SystemExit("❌ HSAF_USER / HSAF_PASS not set.")

# Same EXACT Greece bbox as your existing Greece runner
GRID_N = 300
GRID_LON_MIN, GRID_LON_MAX = 19.0, 30.0
GRID_LAT_MIN, GRID_LAT_MAX = 34.5, 42.5

# Tw thresholds chosen by you
TW_SNOW_MAX = 0.5
TW_MIXED_MAX = 1.5

# Time windows
WEATHER_TIME_WINDOW_MIN = 60
HSAF_MAX_AGE_HOURS = 8

# Wet-bulb local regression controls
LAPSE_DEFAULT = -0.0055   # conservative default for Tw lapse (degC/m)
LAPSE_MIN = -0.0100
LAPSE_MAX = -0.0010

TEMP_COARSE_N = 120
K_LOCAL = 25
R_LOCAL_M = 150_000
ALT_RANGE_MIN_M = 400
MIN_NBR = 8
USE_DISTANCE_WEIGHTS = True

# H-SAF file regex based on product manual + observed folder contents
H68_FILE_RE = re.compile(r"^h68_\d{8}_\d{6}_\d{6}_hea\.nc\.gz$", re.IGNORECASE)

# Use H-SAF support filtering to avoid plotting unsupported pixels
MIN_TOTALCOUNT = 1
MIN_QIND = 0.0

# External precipitation visibility threshold
MIN_RR_TO_PLOT = 0.05  # mm/h

# Plotting
SHOW_GREECE_OUTLINE = True

PRECIP_CMAP = ListedColormap([
    "#f7fbff",
    "#deebf7",
    "#9ecae1",
    "#4292c6",
    "#2171b5",
    "#084594",
])
PRECIP_CMAP.set_under("#ffffff")
PRECIP_CMAP.set_bad("#ffffff")
PRECIP_BOUNDS = [0.1, 0.5, 1, 2, 5, 10, 25]
PRECIP_NORM = BoundaryNorm(PRECIP_BOUNDS, PRECIP_CMAP.N)

PHASE_CMAP = ListedColormap([
    "#3182bd",  # rain likely
    "#9e9ac8",  # mixed / sleet-favoured
    "#f16913",  # snow likely
])
PHASE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], PHASE_CMAP.N)

OUTPUT_DIR = os.path.join(BASE_DIR, "ptype_hsaf_greece")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

def parse_h68_times_from_name(fname: str):
    """
    h68_20260403_080000_082959_hea.nc.gz
    """
    m = re.match(r"^h68_(\d{8})_(\d{6})_(\d{6})_hea\.nc\.gz$", os.path.basename(fname), re.IGNORECASE)
    if not m:
        return None, None
    d, s, f = m.groups()
    start = datetime.strptime(d + s, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    end = datetime.strptime(d + f, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return start, end


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
        raise SystemExit("❌ OpenSSL not found. Install it or decrypt in CI before running.")
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

    # Greece only
    c = df["Country"].astype(str).str.strip().str.upper()
    df = df[c.isin(["GR", "GREECE"])].copy()

    # Same bbox guard for Greece-only product
    df = df[
        df["Longitude"].between(GRID_LON_MIN - 2.0, GRID_LON_MAX + 2.0) &
        df["Latitude"].between(GRID_LAT_MIN - 2.0, GRID_LAT_MAX + 2.0)
    ].copy()

    # Recent observations only
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
    """
    Stull (2011) approximation, valid for typical near-surface conditions.
    Inputs:
      t_c: air temperature in °C
      rh_pct: RH in %
    """
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
# H-SAF H68 DOWNLOAD / READ
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

def hsaf_list_latest_file():
    ftp = ftp_connect_hsaf(HSAF_HOST, HSAF_USER, HSAF_PASS)
    try:
        ftp.cwd(HSAF_REMOTE_DIR)
        names = ftp.nlst()
        basenames = [os.path.basename(n) for n in names if n]
        files = [n for n in basenames if H68_FILE_RE.match(n)]
        if not files:
            raise RuntimeError(f"No H68 files found in {HSAF_REMOTE_DIR}")
        files.sort()
        latest = files[-1]
        return latest
    finally:
        try:
            ftp.quit()
        except Exception:
            pass

def hsaf_download_file(remote_name: str, local_path: str):
    ftp = ftp_connect_hsaf(HSAF_HOST, HSAF_USER, HSAF_PASS)
    try:
        ftp.cwd(HSAF_REMOTE_DIR)
        with open(local_path, "wb") as f:
            ftp.retrbinary("RETR " + remote_name, f.write)
        print(f"[hsaf] downloaded {remote_name}")
    finally:
        try:
            ftp.quit()
        except Exception:
            pass

def open_h68_dataset_from_gz(gz_path: str):
    """
    Returns dict with lon, lat, rr, qind, TotalCount.
    """
    tmp_nc = tempfile.NamedTemporaryFile(prefix="h68_", suffix=".nc", delete=False)
    tmp_nc_path = tmp_nc.name
    tmp_nc.close()

    try:
        with gzip.open(gz_path, "rb") as f_in, open(tmp_nc_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        if XR_OK:
            ds = xr.open_dataset(tmp_nc_path)
            out = {}
            for var in ["lon", "lat", "rr", "qind", "TotalCount", "phase"]:
                if var in ds.variables:
                    out[var] = ds[var].values
                else:
                    out[var] = None
            ds.close()
            return out

        if NC4_OK:
            ds = Dataset(tmp_nc_path, "r")
            out = {}
            for var in ["lon", "lat", "rr", "qind", "TotalCount", "phase"]:
                if var in ds.variables:
                    out[var] = ds.variables[var][:]
                else:
                    out[var] = None
            ds.close()
            return out

        raise RuntimeError("Neither xarray nor netCDF4 is available to read H68 NetCDF.")

    finally:
        try:
            os.remove(tmp_nc_path)
        except Exception:
            pass

def prepare_h68_grid(raw: dict):
    lon = np.asarray(raw["lon"], dtype=float).reshape(-1)
    lat = np.asarray(raw["lat"], dtype=float).reshape(-1)
    rr = np.asarray(raw["rr"], dtype=float)
    qind = None if raw["qind"] is None else np.asarray(raw["qind"], dtype=float)
    tcount = None if raw["TotalCount"] is None else np.asarray(raw["TotalCount"], dtype=float)

    # Handle shape ambiguity from manual / file structure
    if rr.shape == (len(lat), len(lon)):
        pass
    elif rr.shape == (len(lon), len(lat)):
        rr = rr.T
        if qind is not None and qind.shape == (len(lon), len(lat)):
            qind = qind.T
        if tcount is not None and tcount.shape == (len(lon), len(lat)):
            tcount = tcount.T
    else:
        raise RuntimeError(f"Unexpected rr shape {rr.shape} for lon={len(lon)}, lat={len(lat)}")

    if qind is None:
        qind = np.full_like(rr, np.nan, dtype=float)
    else:
        if qind.shape == (len(lon), len(lat)):
            qind = qind.T
        elif qind.shape != rr.shape:
            qind = np.full_like(rr, np.nan, dtype=float)

    if tcount is None:
        tcount = np.full_like(rr, np.nan, dtype=float)
    else:
        if tcount.shape == (len(lon), len(lat)):
            tcount = tcount.T
        elif tcount.shape != rr.shape:
            tcount = np.full_like(rr, np.nan, dtype=float)

    # Ensure ascending axes for RegularGridInterpolator
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        rr = rr[::-1, :]
        qind = qind[::-1, :]
        tcount = tcount[::-1, :]

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        rr = rr[:, ::-1]
        qind = qind[:, ::-1]
        tcount = tcount[:, ::-1]

    return lon, lat, rr, qind, tcount


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
        return
    remote_filename = os.path.basename(local_file)
    ftps = ftps_connect_with_retries(FTP_HOST, FTP_USER, FTP_PASS, attempts=6, base_sleep=5, timeout=timeout)
    try:
        with open(local_file, "rb") as f:
            ftps.storbinary("STOR " + remote_filename, f)
        print(f"📤 Uploaded: {remote_filename}")
    finally:
        try:
            ftps.quit()
        except Exception:
            pass


# =============================================================================
# PLOTTING
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

def common_footer(ax, created_dt_ath, weather_until, h68_start_utc, h68_end_utc, tw_global_lapse):
    timestamp_text = created_dt_ath.strftime("%Y-%m-%d %H:%M") + f" {athens_abbrev(created_dt_ath)}"
    hsaf_window = "—"
    if h68_start_utc and h68_end_utc:
        hsaf_window = f"{h68_start_utc.strftime('%Y-%m-%d %H:%M')} to {h68_end_utc.strftime('%H:%M')} UTC"

    left_text = (
        f"Δημιουργήθηκε για το {BRAND_NAME}\n"
        f"{timestamp_text}\n"
        f"weathernow έως: {weather_until}"
    )
    right_text = (
        f"H-SAF H68\n"
        f"{hsaf_window}\n"
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

def save_combined_phase_map(
    out_path,
    grid_lon, grid_lat,
    phase_idx,
    rr_grid,
    created_dt_ath,
    weather_until,
    h68_start_utc, h68_end_utc,
    tw_global_lapse
):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    phase_arr = ma.masked_invalid(phase_idx)
    ax.imshow(
        phase_arr,
        extent=(GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX),
        origin="lower",
        cmap=PHASE_CMAP,
        norm=PHASE_NORM,
        alpha=0.75
    )

    rr_for_contour = np.where(np.isfinite(rr_grid) & (rr_grid >= 0.1), rr_grid, np.nan)
    try:
        ax.contour(
            grid_lon, grid_lat, rr_for_contour,
            levels=[0.1, 0.5, 1, 2, 5, 10],
            colors="black",
            linewidths=0.7
        )
    except Exception:
        pass

    draw_greece_outline(ax)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#3182bd", edgecolor="black", label="Rain likely"),
        Patch(facecolor="#9e9ac8", edgecolor="black", label="Mixed / sleet-favoured"),
        Patch(facecolor="#f16913", edgecolor="black", label="Snow likely"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)

    ax.set_xlim(GRID_LON_MIN, GRID_LON_MAX)
    ax.set_ylim(GRID_LAT_MIN, GRID_LAT_MAX)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)
    ax.set_title("Πιθανή φάση υετού με βάση H-SAF H68 + Tw", fontsize=14, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    common_footer(ax, created_dt_ath, weather_until, h68_start_utc, h68_end_utc, tw_global_lapse)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")

def save_rate_map(
    out_path,
    title,
    rr_masked,
    grid_lon, grid_lat,
    created_dt_ath,
    weather_until,
    h68_start_utc, h68_end_utc,
    tw_global_lapse
):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    arr = ma.masked_invalid(rr_masked)
    img = ax.imshow(
        arr,
        extent=(GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX),
        origin="lower",
        cmap=PRECIP_CMAP,
        norm=PRECIP_NORM,
        alpha=0.85
    )

    try:
        ax.contour(
            grid_lon, grid_lat, np.where(np.isfinite(rr_masked) & (rr_masked >= 0.1), rr_masked, np.nan),
            levels=[0.1, 0.5, 1, 2, 5, 10],
            colors="black",
            linewidths=0.7
        )
    except Exception:
        pass

    draw_greece_outline(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax, boundaries=PRECIP_BOUNDS, extend="max")
    cbar.set_label("H-SAF H68 precipitation rate (mm/h)", fontsize=11)

    ax.set_xlim(GRID_LON_MIN, GRID_LON_MAX)
    ax.set_ylim(GRID_LAT_MIN, GRID_LAT_MAX)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    common_footer(ax, created_dt_ath, weather_until, h68_start_utc, h68_end_utc, tw_global_lapse)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_geojson_and_altitude_bundle()

    # -------------------------------------------------------------------------
    # 1) Download latest H68 file
    # -------------------------------------------------------------------------
    latest_remote = hsaf_list_latest_file()
    h68_start_utc, h68_end_utc = parse_h68_times_from_name(latest_remote)

    if h68_end_utc is not None:
        age_h = (datetime.now(UTC) - h68_end_utc).total_seconds() / 3600.0
        if age_h > HSAF_MAX_AGE_HOURS:
            print(f"⚠️ Latest H68 file is old ({age_h:.1f} h): {latest_remote}")

    local_gz = os.path.join(OUTPUT_DIR, latest_remote)
    hsaf_download_file(latest_remote, local_gz)

    raw = open_h68_dataset_from_gz(local_gz)
    h68_lon, h68_lat, h68_rr, h68_qind, h68_tcount = prepare_h68_grid(raw)

    # -------------------------------------------------------------------------
    # 2) Weather feed
    # -------------------------------------------------------------------------
    cache_txt = os.path.join(BASE_DIR, "weathernow_cached.txt")
    weather_text, source = robust_fetch_text(CURRENTWEATHER_URL, cache_txt=cache_txt, timeout=60, tries=6)
    with open(cache_txt, "w", encoding="utf-8") as f:
        f.write(weather_text)

    wdf = load_and_clean_weather_feed(weather_text)
    weather_until = fmt_data_until(wdf["Datetime"].max())

    # Station wet-bulb
    wdf["Tw"] = wet_bulb_stull(wdf["TNow"].values, wdf["RHNow"].values)

    station_lons = wdf["Longitude"].values.astype(float)
    station_lats = wdf["Latitude"].values.astype(float)
    station_tw = wdf["Tw"].values.astype(float)
    station_alt = sample_altitude_vrt_m(ALT_VRT_PATH, station_lons, station_lats)

    # -------------------------------------------------------------------------
    # 3) Target Greece grid (same bbox, EPSG:4326)
    # -------------------------------------------------------------------------
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_N),
        np.linspace(GRID_LAT_MIN, GRID_LAT_MAX, GRID_N)
    )

    # Wet-bulb field
    tw_grid, tw_global_lapse = build_variable_grid_local_lr_wgs(
        GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX,
        grid_lon, grid_lat,
        station_lons, station_lats,
        station_tw, station_alt,
        ALT_VRT_PATH,
        grid_n=GRID_N
    )

    # -------------------------------------------------------------------------
    # 4) Interpolate H68 to target grid
    # -------------------------------------------------------------------------
    rr_interp = RegularGridInterpolator(
        (h68_lat, h68_lon), h68_rr,
        bounds_error=False, fill_value=np.nan
    )
    qind_interp = RegularGridInterpolator(
        (h68_lat, h68_lon), h68_qind,
        bounds_error=False, fill_value=np.nan
    )
    tcount_interp = RegularGridInterpolator(
        (h68_lat, h68_lon), h68_tcount,
        bounds_error=False, fill_value=np.nan
    )

    pts = np.c_[grid_lat.ravel(), grid_lon.ravel()]
    rr_grid = rr_interp(pts).reshape(grid_lon.shape)
    qind_grid = qind_interp(pts).reshape(grid_lon.shape)
    tcount_grid = tcount_interp(pts).reshape(grid_lon.shape)

    # H-SAF support filtering
    support_mask = (
        np.isfinite(rr_grid) &
        np.isfinite(qind_grid) &
        np.isfinite(tcount_grid) &
        (tcount_grid >= MIN_TOTALCOUNT) &
        (qind_grid >= MIN_QIND)
    )

    rr_grid = np.where(support_mask, rr_grid, np.nan)

    # -------------------------------------------------------------------------
    # 5) Phase classification based on your Tw thresholds
    # -------------------------------------------------------------------------
    precip_mask = np.isfinite(rr_grid) & (rr_grid >= MIN_RR_TO_PLOT) & np.isfinite(tw_grid)

    phase_idx = np.full(grid_lon.shape, np.nan, dtype=float)
    # 0 rain, 1 mixed, 2 snow
    phase_idx[precip_mask & (tw_grid > TW_MIXED_MAX)] = 0
    phase_idx[precip_mask & (tw_grid > TW_SNOW_MAX) & (tw_grid <= TW_MIXED_MAX)] = 1
    phase_idx[precip_mask & (tw_grid <= TW_SNOW_MAX)] = 2

    rr_rain = np.where(phase_idx == 0, rr_grid, np.nan)
    rr_mixed = np.where(phase_idx == 1, rr_grid, np.nan)
    rr_snow = np.where(phase_idx == 2, rr_grid, np.nan)

    # -------------------------------------------------------------------------
    # 6) Outputs
    # -------------------------------------------------------------------------
    created_dt_ath = datetime.now(ATHENS_TZ)
    ts = created_dt_ath.strftime("%Y-%m-%d-%H-%M")

    out_combined = os.path.join(OUTPUT_DIR, f"ptype_h68_combined_{ts}.png")
    out_rain = os.path.join(OUTPUT_DIR, f"ptype_h68_rain_{ts}.png")
    out_mixed = os.path.join(OUTPUT_DIR, f"ptype_h68_mixed_{ts}.png")
    out_snow = os.path.join(OUTPUT_DIR, f"ptype_h68_snow_{ts}.png")

    latest_combined = os.path.join(OUTPUT_DIR, "ptype_h68_combined_latest.png")
    latest_rain = os.path.join(OUTPUT_DIR, "ptype_h68_rain_latest.png")
    latest_mixed = os.path.join(OUTPUT_DIR, "ptype_h68_mixed_latest.png")
    latest_snow = os.path.join(OUTPUT_DIR, "ptype_h68_snow_latest.png")

    save_combined_phase_map(
        out_combined,
        grid_lon, grid_lat,
        phase_idx,
        rr_grid,
        created_dt_ath,
        weather_until,
        h68_start_utc, h68_end_utc,
        tw_global_lapse
    )

    save_rate_map(
        out_rain,
        "H-SAF H68 precipitation rate where rain is likely (Tw > 1.5°C)",
        rr_rain,
        grid_lon, grid_lat,
        created_dt_ath,
        weather_until,
        h68_start_utc, h68_end_utc,
        tw_global_lapse
    )

    save_rate_map(
        out_mixed,
        "H-SAF H68 precipitation rate where mixed / sleet-favoured (0.5°C < Tw ≤ 1.5°C)",
        rr_mixed,
        grid_lon, grid_lat,
        created_dt_ath,
        weather_until,
        h68_start_utc, h68_end_utc,
        tw_global_lapse
    )

    save_rate_map(
        out_snow,
        "H-SAF H68 precipitation rate where snow is likely (Tw ≤ 0.5°C)",
        rr_snow,
        grid_lon, grid_lat,
        created_dt_ath,
        weather_until,
        h68_start_utc, h68_end_utc,
        tw_global_lapse
    )

    shutil.copy(out_combined, latest_combined)
    shutil.copy(out_rain, latest_rain)
    shutil.copy(out_mixed, latest_mixed)
    shutil.copy(out_snow, latest_snow)

    print(f"✅ Saved latest: {latest_combined}")
    print(f"✅ Saved latest: {latest_rain}")
    print(f"✅ Saved latest: {latest_mixed}")
    print(f"✅ Saved latest: {latest_snow}")

    # Optional upload using your existing FTP credentials
    try:
        for fp in [out_combined, out_rain, out_mixed, out_snow,
                   latest_combined, latest_rain, latest_mixed, latest_snow]:
            ftp_upload_file(fp)
    except Exception as e:
        print(f"⚠️ FTP upload failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
