#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rainintensityall.py

Single entrypoint for all rain-intensity maps:
  - Greece (WGS84 degrees) + snowflakes + snowline altitude labels (local lapse regression)
  - Attica / Crete / NE Greece (EGSA87 EPSG:2100 meters) + snowflakes + snowline altitude labels
  - Cyprus (UTM 36N EPSG:32636 meters) + snowflakes + snowline altitude labels

Outputs:
  Greece + EGSA regions: ./rainintensitymaps/
  Cyprus:               ./cyprusrainintensitymaps/

FTP:
  Optional everywhere. If FTP_HOST/FTP_USER/FTP_PASS missing -> upload/prune skipped.

Encrypted assets (Greece + EGSA regions only):
  - greece.geojson.enc -> greece.geojson (password: GEOJSON_PASS)
  - altitude.zip.enc   -> extracts GRC_alt.vrt (+ .grd/.gri) next to script (password: GEOJSON_PASS)

Feed URL:
  CURRENTWEATHER_URL must be set.
"""

import os
import re
import sys
import time
import math
import json
import shutil
import random
import socket
import zipfile
import argparse
import subprocess
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import StringIO

import numpy as np
import pandas as pd
import geopandas as gpd
import numpy.ma as ma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial import cKDTree
from scipy.ndimage import zoom

import rasterio
from rasterio.warp import transform as rio_transform
from pyproj import Transformer

from ftplib import FTP_TLS, error_perm
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

import requests


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATHENS_TZ = ZoneInfo("Europe/Athens")

CURRENTWEATHER_URL = os.environ.get("CURRENTWEATHER_URL", "").strip()
if not CURRENTWEATHER_URL:
    raise SystemExit("❌ CURRENTWEATHER_URL secret/env not set.")

# Optional FTP
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()

def ftp_enabled():
    return bool(FTP_HOST and FTP_USER and FTP_PASS)

def ftps_connect_with_retries(host, user, passwd, attempts=6, base_sleep=5, timeout=60):
    """
    Retries FTPS connect/login. Fixes transient runner DNS failures like:
      [Errno -3] Temporary failure in name resolution
    """
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


# Encrypted assets (Greece + EGSA)
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()
GREECE_GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
GREECE_GEOJSON_ENC  = os.path.join(BASE_DIR, "greece.geojson.enc")
ALT_VRT_PATH        = os.path.join(BASE_DIR, "GRC_alt.vrt")
ALT_ENC             = os.path.join(BASE_DIR, "altitude.zip.enc")
ALT_ZIP             = os.path.join(BASE_DIR, "altitude.zip")

# Cyprus assets
CYPRUS_GEOJSON_PATH = os.path.join(BASE_DIR, "cyprus.geojson")
CYPRUS_ALT_TIF_PATH = os.path.join(BASE_DIR, "cyprus_dsm_90m.tif")

# Common thresholds
SNOW_T_C = 2.0
RAIN_THRESH = 0.0

# Lapse bounds (degC per meter)
LAPSE_DEFAULT = -0.0065
LAPSE_MIN = -0.0120
LAPSE_MAX = -0.0010

# Local regression controls (Greece + EGSA)
TEMP_COARSE_N = 120
K_LOCAL = 25
R_LOCAL_M = 150_000
ALT_RANGE_MIN_M = 400
MIN_NBR = 8
USE_DISTANCE_WEIGHTS = True

# Snowflake controls (common defaults, region overrides allowed)
DEFAULT_SNOW_FONTSIZE = 6
DEFAULT_SNOW_STROKE_W = 1.2
DEFAULT_MAX_SNOWFLAKES = 5000
DEFAULT_SNOW_SEED = 123

# Label controls (common defaults, region overrides allowed)
DEFAULT_LABEL_EVERY_N = 10
DEFAULT_LABEL_FONTSIZE = 6
DEFAULT_LABEL_STROKE_W = 1.2
ISO_ALT_ROUND_M = 50
ISO_ALT_MIN_M = 0
ISO_ALT_MAX_M = 5000

# Colormap (Greece + EGSA) to match your Greece/Attica/Crete/NE scripts
CMAP_GREECE_EGSA = ListedColormap(["#deebf7", "#9ecae1", "#4292c6", "#08519c"])
CMAP_GREECE_EGSA.set_under("#ffffff")
CMAP_GREECE_EGSA.set_over("#08306b")
CMAP_GREECE_EGSA.set_bad("#ffffff")
BOUNDS_GREECE_EGSA = [0.1, 2, 6, 36, 100]
NORM_GREECE_EGSA = BoundaryNorm(boundaries=BOUNDS_GREECE_EGSA, ncolors=CMAP_GREECE_EGSA.N)

# Cyprus colormap (from your Cyprus script)
CMAP_CYPRUS = ListedColormap(["#ffffff", "#deebf7", "#9ecae1", "#4292c6", "#08519c"])
BOUNDS_CYPRUS = [0.0, 0.2, 2, 6, 40, 100]
NORM_CYPRUS = BoundaryNorm(boundaries=BOUNDS_CYPRUS, ncolors=CMAP_CYPRUS.N)


# =============================================================================
# ENCRYPTED ASSET HELPERS (Greece + EGSA only)
# =============================================================================
def ensure_geojson_and_altitude_bundle():
    """
    Ensures:
      - greece.geojson exists (decrypt from greece.geojson.enc if needed)
      - GRC_alt.vrt exists (decrypt+unzip altitude.zip.enc if needed)
    Uses GEOJSON_PASS.
    """
    # GeoJSON
    if not os.path.exists(GREECE_GEOJSON_PATH) and os.path.exists(GREECE_GEOJSON_ENC):
        if not GEOJSON_PASS:
            raise SystemExit("❌ greece.geojson missing and GEOJSON_PASS not set to decrypt greece.geojson.enc")
        _openssl_decrypt(GREECE_GEOJSON_ENC, GREECE_GEOJSON_PATH, GEOJSON_PASS)

    # Altitude bundle
    if os.path.exists(ALT_VRT_PATH):
        return

    if not os.path.exists(ALT_ENC):
        return

    if not GEOJSON_PASS:
        raise SystemExit("❌ DEM bundle missing and GEOJSON_PASS not set to decrypt altitude.zip.enc")

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
        raise SystemExit("❌ Decrypted altitude bundle didn’t contain GRC_alt.vrt next to the script.")


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


# =============================================================================
# FETCH HELPERS
# =============================================================================
def robust_fetch_text(url: str, cache_txt: str, timeout: int = 60, tries: int = 6):
    """
    Returns (text, source) where source is one of:
      "network", "curl", "cache", "localfile"

    Supports:
      - url starting with file://
      - url being a local path
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/plain,text/*;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }

    # local file fast path
    if url.startswith("file://"):
        path = url[7:]
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(), "localfile"

    if os.path.exists(url):
        with open(url, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "localfile"

    last_err = None
    session = requests.Session()

    for attempt in range(1, tries + 1):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            text = r.text or ""
            if not text.strip():
                raise RuntimeError("Empty response body.")
            # light sanity: must look like TSV-ish
            first_line = text.splitlines()[0].strip()
            looks_like_tsv = ("\t" in first_line) and (len(first_line) >= 10)
            if ("Datetime" not in text) and (not looks_like_tsv):
                raise RuntimeError("Response does not look like expected TSV.")
            return text, "network"
        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 20) + random.random()
            print(f"[fetch] attempt {attempt}/{tries} failed: {e}. Retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)

    # curl fallback
    try:
        print("[fetch] falling back to curl...")
        cp = subprocess.run(
            ["curl", "-L", "-A", headers["User-Agent"], "--max-time", str(timeout), url],
            check=True,
            capture_output=True,
            text=True,
        )
        text = cp.stdout or ""
        if text.strip():
            first_line = text.splitlines()[0].strip()
            looks_like_tsv = ("\t" in first_line) and (len(first_line) >= 10)
            if ("Datetime" in text) or looks_like_tsv:
                return text, "curl"
        raise RuntimeError("curl returned empty/non-TSV output.")
    except Exception as e:
        print(f"[fetch] curl fallback failed: {e}")

    # cached fallback
    if cache_txt and os.path.exists(cache_txt):
        print(f"[fetch] using cached file: {cache_txt}")
        with open(cache_txt, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "cache"

    raise RuntimeError("Failed to fetch feed") from last_err


def print_latest_rows(df: pd.DataFrame, n: int = 8, title: str = ""):
    if title:
        print(title)
    if df is None or df.empty or "Datetime" not in df.columns:
        print("ℹ️ No rows to preview.")
        return

    cols = [c for c in ["Datetime", "webcode", "Latitude", "Longitude", "TNow", "RainIntensity"] if c in df.columns]
    tmp = df.dropna(subset=["Datetime"]).copy()
    tmp = tmp.sort_values("Datetime", ascending=False).head(n)

    print(f"🕒 Latest {min(n, len(tmp))} rows (Datetime desc):")
    for _, r in tmp.iterrows():
        parts = []
        for c in cols:
            v = r.get(c, "")
            if c == "Datetime":
                try:
                    vv = pd.to_datetime(v)
                    parts.append(f"{c}={vv.strftime('%Y-%m-%d %H:%M:%S%z') if vv.tzinfo else vv.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception:
                    parts.append(f"{c}={v}")
            else:
                parts.append(f"{c}={v}")
        print("  - " + " | ".join(parts))


# =============================================================================
# FTP HELPERS (optional)
# =============================================================================
def ftp_upload_file(local_file: str, timeout: int = 60):
    if not ftp_enabled():
        print("ℹ️ FTP disabled (missing env). Skipping upload.")
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


def ftp_prune_timestamped(prefix: str, latest_name: str, keep: int):
    """
    Delete old timestamped PNGs matching:
      {prefix}YYYY-MM-DD-HH-MM.png   (or prefix + digits with same pattern)

    Keeps:
      - latest_name
      - newest keep timestamped
    """
    if not ftp_enabled():
        print("ℹ️ FTP disabled (missing env). Skipping remote prune.")
        return

    pat = re.compile(rf"^{re.escape(prefix)}\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{2}}\.png$")

    ftps = ftps_connect_with_retries(FTP_HOST, FTP_USER, FTP_PASS, attempts=6, base_sleep=5, timeout=60)


    try:
        try:
            names = ftps.nlst()
        except Exception as e:
            print("⚠️ Could not list remote directory:", e)
            return

        basenames = [os.path.basename(n) for n in names if n]
        timestamped = [n for n in basenames if pat.match(n) and n != latest_name]

        if not timestamped:
            print("ℹ️ No timestamped PNGs to prune remotely.")
            return

        timestamped.sort()
        if len(timestamped) <= keep:
            print(f"ℹ️ {len(timestamped)} timestamped files ≤ keep={keep}. Nothing to delete.")
            return

        to_delete = timestamped[:-keep]
        for fname in to_delete:
            try:
                ftps.delete(fname)
                print("🧹 Deleted old remote file:", fname)
            except Exception as e:
                print(f"⚠️ Failed to delete {fname}: {e}")

    finally:
        try:
            ftps.quit()
        except Exception:
            pass


# =============================================================================
# NUMERICS: ALTITUDE SAMPLING
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


def sample_altitude_raster_m(raster_path: str, xs, ys, input_crs: str) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise RuntimeError("Altitude raster has no CRS defined.")

        if str(src.crs) != str(input_crs):
            x2, y2 = rio_transform(str(input_crs), src.crs, xs.tolist(), ys.tolist())
        else:
            x2, y2 = xs.tolist(), ys.tolist()

        samples = list(src.sample(zip(x2, y2)))
        arr = np.array(samples, dtype=float).reshape(-1)

        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        arr = np.where(np.isfinite(arr), arr, np.nan)

    return arr


# =============================================================================
# NUMERICS: LAPSE
# =============================================================================
def fit_global_lapse_rate(tnow_c: np.ndarray, alt_m: np.ndarray) -> float:
    mask = np.isfinite(tnow_c) & np.isfinite(alt_m)
    if mask.sum() < 8:
        return LAPSE_DEFAULT
    x = alt_m[mask]
    y = tnow_c[mask]
    b, _a = np.polyfit(x, y, 1)
    if not (LAPSE_MIN <= b <= LAPSE_MAX):
        return LAPSE_DEFAULT
    return float(b)


# =============================================================================
# NUMERICS: IDW (ANY-NEIGHBOR VALIDITY FIX)
# =============================================================================
def idw_any_neighbor(x, y, z, xi, yi, power=2, max_distance=1.0, k=8) -> np.ndarray:
    """
    Generic IDW:
      - x,y,z are station coordinates (same units)
      - xi,yi are grids
      - max_distance in same units
      - Valid if ANY neighbor within max_distance exists (not "all K finite")
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]

    if len(z) < 3:
        return np.full_like(xi, np.nan, dtype=float)

    tree = cKDTree(np.c_[x, y])
    dist, idx = tree.query(
        np.c_[xi.ravel(), yi.ravel()],
        k=min(k, len(z)),
        distance_upper_bound=max_distance
    )

    if dist.ndim == 1:
        dist = dist.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    finite = np.isfinite(dist) & (idx < len(z))
    has_any = np.any(finite, axis=1)

    zi = np.full(dist.shape[0], np.nan, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.zeros_like(dist, dtype=float)
        w[finite] = 1.0 / np.power(dist[finite], power)
        w = np.where((dist == 0) & finite, 1e12, w)

        zv = np.zeros_like(dist, dtype=float)
        zv[finite] = z[idx[finite]]

        num = np.sum(w * zv, axis=1)
        den = np.sum(w, axis=1)

        good = has_any & (den > 0)
        zi[good] = num[good] / den[good]

    return zi.reshape(xi.shape)


# =============================================================================
# NUMERICS: THINNING (MIN-DISTANCE)
# =============================================================================
def downsample_mask_points(grid_x, grid_y, mask, max_points=5000, min_sep=0.04, seed=123):
    """
    Works in whatever units grid_x/grid_y are in.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([]), np.array([])

    pts = np.column_stack([grid_x[ys, xs], grid_y[ys, xs]])
    rng = np.random.default_rng(seed)
    pts = pts[rng.permutation(len(pts))]

    kept = []
    kept_tree = None

    for p in pts:
        if not kept:
            kept.append(p)
            kept_tree = cKDTree(np.array(kept))
        else:
            d, _ = kept_tree.query(p, k=1)
            if d >= min_sep:
                kept.append(p)
                if len(kept) % 150 == 0:
                    kept_tree = cKDTree(np.array(kept))
        if len(kept) >= max_points:
            break

    kept = np.array(kept)
    return kept[:, 0], kept[:, 1]


# =============================================================================
# TEMPERATURE GRID: LOCAL REGRESSION (Greece WGS84)
# =============================================================================
def build_temperature_grid_local_lr_wgs(
    lon_min, lon_max, lat_min, lat_max,
    grid_lon, grid_lat,
    station_lons, station_lats,
    station_tnow, station_alt_m,
    alt_vrt_path,
    grid_n,
):
    to_egsa = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)
    st_x, st_y = to_egsa.transform(station_lons.tolist(), station_lats.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    st_t = np.asarray(station_tnow, dtype=float)
    st_alt = np.asarray(station_alt_m, dtype=float)

    ok = np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_t) & np.isfinite(st_alt)
    st_x, st_y, st_t, st_alt = st_x[ok], st_y[ok], st_t[ok], st_alt[ok]

    if len(st_t) < 10:
        shape = grid_lon.shape
        return (np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan), LAPSE_DEFAULT)

    b_global = fit_global_lapse_rate(st_t, st_alt)
    t0_global = np.nanmedian(st_t - b_global * st_alt)

    # Coarse lon/lat grid
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
        k=min(K_LOCAL, len(st_t)),
        distance_upper_bound=R_LOCAL_M
    )

    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    pred = np.full(c_x.shape[0], np.nan, dtype=float)
    b_loc = np.full(c_x.shape[0], np.nan, dtype=float)
    a_loc = np.full(c_x.shape[0], np.nan, dtype=float)

    for i in range(c_x.shape[0]):
        di = dists[i]
        ii = idx[i]
        m = np.isfinite(di) & (ii < len(st_t))
        if not np.any(m):
            continue
        di = di[m]
        ii = ii[m]
        t = st_t[ii]
        a = st_alt[ii]

        m2 = np.isfinite(t) & np.isfinite(a)
        if m2.sum() < MIN_NBR:
            continue
        t = t[m2]
        a = a[m2]
        di = di[m2]

        if (np.nanmax(a) - np.nanmin(a)) < ALT_RANGE_MIN_M:
            continue

        try:
            if USE_DISTANCE_WEIGHTS:
                w = 1.0 / (di + 2000.0)
                b, intercept = np.polyfit(a, t, 1, w=w)
            else:
                b, intercept = np.polyfit(a, t, 1)
        except Exception:
            continue

        if not (LAPSE_MIN <= b <= LAPSE_MAX):
            continue

        alt_here = c_alt.ravel()[i]
        if not np.isfinite(alt_here):
            continue

        pred[i] = intercept + b * alt_here
        b_loc[i] = b
        a_loc[i] = intercept

    c_alt_flat = c_alt.ravel()
    fallback_temp = t0_global + b_global * c_alt_flat
    pred = np.where(np.isfinite(pred), pred, fallback_temp)

    missing = ~np.isfinite(b_loc) | ~np.isfinite(a_loc)
    b_loc = np.where(missing, b_global, b_loc)
    a_loc = np.where(missing, t0_global, a_loc)

    pred_coarse = pred.reshape(c_grid_lon.shape)
    b_coarse = b_loc.reshape(c_grid_lon.shape)
    a_coarse = a_loc.reshape(c_grid_lon.shape)

    zoom_factor = grid_n / float(TEMP_COARSE_N)
    temp_fine = zoom(pred_coarse, zoom=(zoom_factor, zoom_factor), order=1)
    b_fine = zoom(b_coarse, zoom=(zoom_factor, zoom_factor), order=1)
    a_fine = zoom(a_coarse, zoom=(zoom_factor, zoom_factor), order=1)

    def force_shape(arr, shape):
        if arr.shape == shape:
            return arr
        out = np.full(shape, np.nan, dtype=float)
        h = min(shape[0], arr.shape[0])
        w = min(shape[1], arr.shape[1])
        out[:h, :w] = arr[:h, :w]
        return out

    shape = grid_lon.shape
    temp_fine = force_shape(temp_fine, shape)
    b_fine = force_shape(b_fine, shape)
    a_fine = force_shape(a_fine, shape)

    return temp_fine, b_fine, a_fine, b_global


# =============================================================================
# TEMPERATURE GRID: LOCAL REGRESSION (EGSA regions)
# =============================================================================
def build_temperature_grid_local_lr_projected(
    grid_x_m, grid_y_m,
    station_x_m, station_y_m,
    station_tnow, station_alt_m,
    transformer_proj_to_wgs,
    alt_vrt_path,
    x_min, x_max, y_min, y_max,
    grid_n,
):
    st_x = np.asarray(station_x_m, dtype=float)
    st_y = np.asarray(station_y_m, dtype=float)
    st_t = np.asarray(station_tnow, dtype=float)
    st_alt = np.asarray(station_alt_m, dtype=float)

    ok = np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_t) & np.isfinite(st_alt)
    st_x, st_y, st_t, st_alt = st_x[ok], st_y[ok], st_t[ok], st_alt[ok]

    if len(st_t) < 10:
        shape = grid_x_m.shape
        return (np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan), LAPSE_DEFAULT)

    b_global = fit_global_lapse_rate(st_t, st_alt)
    t0_global = np.nanmedian(st_t - b_global * st_alt)

    c_x = np.linspace(x_min, x_max, TEMP_COARSE_N)
    c_y = np.linspace(y_min, y_max, TEMP_COARSE_N)
    c_grid_x, c_grid_y = np.meshgrid(c_x, c_y)

    c_lon, c_lat = transformer_proj_to_wgs.transform(c_grid_x.ravel().tolist(), c_grid_y.ravel().tolist())
    c_alt = sample_altitude_vrt_m(alt_vrt_path, np.array(c_lon), np.array(c_lat)).reshape(c_grid_x.shape)

    tree = cKDTree(np.c_[st_x, st_y])
    dists, idx = tree.query(
        np.c_[c_grid_x.ravel(), c_grid_y.ravel()],
        k=min(K_LOCAL, len(st_t)),
        distance_upper_bound=R_LOCAL_M
    )

    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    pred = np.full(c_grid_x.size, np.nan, dtype=float)
    b_loc = np.full(c_grid_x.size, np.nan, dtype=float)
    a_loc = np.full(c_grid_x.size, np.nan, dtype=float)

    for i in range(c_grid_x.size):
        di = dists[i]
        ii = idx[i]
        m = np.isfinite(di) & (ii < len(st_t))
        if not np.any(m):
            continue
        di = di[m]
        ii = ii[m]
        t = st_t[ii]
        a = st_alt[ii]

        m2 = np.isfinite(t) & np.isfinite(a)
        if m2.sum() < MIN_NBR:
            continue
        t = t[m2]
        a = a[m2]
        di = di[m2]

        if (np.nanmax(a) - np.nanmin(a)) < ALT_RANGE_MIN_M:
            continue

        try:
            if USE_DISTANCE_WEIGHTS:
                w = 1.0 / (di + 2000.0)
                b, intercept = np.polyfit(a, t, 1, w=w)
            else:
                b, intercept = np.polyfit(a, t, 1)
        except Exception:
            continue

        if not (LAPSE_MIN <= b <= LAPSE_MAX):
            continue

        alt_here = c_alt.ravel()[i]
        if not np.isfinite(alt_here):
            continue

        pred[i] = intercept + b * alt_here
        b_loc[i] = b
        a_loc[i] = intercept

    c_alt_flat = c_alt.ravel()
    fallback_temp = t0_global + b_global * c_alt_flat
    pred = np.where(np.isfinite(pred), pred, fallback_temp)

    missing = ~np.isfinite(b_loc) | ~np.isfinite(a_loc)
    b_loc = np.where(missing, b_global, b_loc)
    a_loc = np.where(missing, t0_global, a_loc)

    pred_coarse = pred.reshape(c_grid_x.shape)
    b_coarse = b_loc.reshape(c_grid_x.shape)
    a_coarse = a_loc.reshape(c_grid_x.shape)

    zoom_factor = grid_n / float(TEMP_COARSE_N)
    temp_fine = zoom(pred_coarse, zoom=(zoom_factor, zoom_factor), order=1)
    b_fine = zoom(b_coarse, zoom=(zoom_factor, zoom_factor), order=1)
    a_fine = zoom(a_coarse, zoom=(zoom_factor, zoom_factor), order=1)

    def force_shape(arr, shape):
        if arr.shape == shape:
            return arr
        out = np.full(shape, np.nan, dtype=float)
        h = min(shape[0], arr.shape[0])
        w = min(shape[1], arr.shape[1])
        out[:h, :w] = arr[:h, :w]
        return out

    shape = grid_x_m.shape
    temp_fine = force_shape(temp_fine, shape)
    b_fine = force_shape(b_fine, shape)
    a_fine = force_shape(a_fine, shape)

    return temp_fine, b_fine, a_fine, b_global


# =============================================================================
# COMMON FEED CLEANING
# =============================================================================
def load_and_clean_feed(text: str, cache_txt: str = "", lon_min=None, lon_max=None) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text), delimiter="\t", engine="python")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    needed = ["Latitude", "Longitude", "RainIntensity", "Datetime"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError("Missing columns in feed: " + ", ".join(missing))

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["RainIntensity"] = pd.to_numeric(df["RainIntensity"], errors="coerce")

    if "TNow" in df.columns:
        df["TNow"] = pd.to_numeric(df["TNow"], errors="coerce")
    else:
        df["TNow"] = np.nan

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    dt = df["Datetime"]
    try:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT")
        else:
            dt = dt.dt.tz_convert("Europe/Athens")
    except Exception as e:
        print(f"⚠️ Datetime tz handling failed ({e}); leaving as-is (may be naive).")
    df["Datetime"] = dt

    before = len(df)
    df = df.dropna(subset=["Latitude", "Longitude", "RainIntensity", "Datetime"]).copy()
    print(f"🧹 dropna essentials: {before} -> {len(df)} (removed {before - len(df)})")

    before = len(df)
    df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()
    print(f"🧹 drop zero lat/lon: {before} -> {len(df)} (removed {before - len(df)})")

    if "webcode" in df.columns:
        bad = ["agrivate_rizia", "metaxochori"]
        before = len(df)
        df = df[~df["webcode"].astype(str).str.lower().isin(bad)].copy()
        print(f"🧹 remove bad webcodes {bad}: {before} -> {len(df)} (removed {before - len(df)})")

    # Optional lon/lat guard (region-specific)
    if lon_min is not None:
        before = len(df)
        df = df[df["Longitude"] >= float(lon_min)].copy()
        print(f"🧹 lon >= {lon_min}: {before} -> {len(df)} (removed {before - len(df)})")

    if lon_max is not None:
        before = len(df)
        df = df[df["Longitude"] <= float(lon_max)].copy()
        print(f"🧹 lon <= {lon_max}: {before} -> {len(df)} (removed {before - len(df)})")

    before = len(df)
    df = df.dropna(subset=["Datetime"]).copy()
    print(f"🧹 drop NaT after tz: {before} -> {len(df)} (removed {before - len(df)})")

    return df


# =============================================================================
# REGION RUNNERS: GREECE (WGS84)
# =============================================================================
def run_greece():
    print("\n====================")
    print("RUN: Greece (WGS84)")
    print("====================")

    ensure_geojson_and_altitude_bundle()

    if not os.path.exists(GREECE_GEOJSON_PATH):
        print(f"❌ Missing: {GREECE_GEOJSON_PATH}")
        return
    if not os.path.exists(ALT_VRT_PATH):
        print(f"❌ Missing: {ALT_VRT_PATH}")
        return

    # Greece region parameters (from your script)
    GRID_N = 300
    GRID_LON_MIN, GRID_LON_MAX = 19.0, 30.0
    GRID_LAT_MIN, GRID_LAT_MAX = 34.5, 42.5

    IDW_POWER = 2
    IDW_K = 8
    MAX_DISTANCE_DEG = 1.0
    DISTANCE_MASK_DEG = 1.5

    TIME_WINDOW_MIN = 45

    # snowflake thinning in degrees
    MIN_SEP_DEG = 0.04
    MAX_SNOWFLAKES = DEFAULT_MAX_SNOWFLAKES
    SNOW_SEED = DEFAULT_SNOW_SEED
    SNOW_FONTSIZE = DEFAULT_SNOW_FONTSIZE
    SNOW_STROKE_W = DEFAULT_SNOW_STROKE_W

    # label density
    LABEL_EVERY_N = 25
    LABEL_FONTSIZE = 6
    LABEL_Y_OFFSET_DEG = 0.03

    output_dir = os.path.join(BASE_DIR, "rainintensitymaps")
    os.makedirs(output_dir, exist_ok=True)

    cache_txt = os.path.join(BASE_DIR, "weathernow_cached.txt")

    # Fetch
    try:
        text, source = robust_fetch_text(CURRENTWEATHER_URL, cache_txt=cache_txt, timeout=60, tries=6)
        print("source:", source)
        with open(cache_txt, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print("❌ Failed to fetch data from feed.")
        return

    df = load_and_clean_feed(text, lon_max=30)
    print(f"📥 Raw(cleaned) rows: {len(df)}")
    print_latest_rows(df, n=8, title="(Cleaned feed preview)")

    # Time filter
    athens_now = datetime.now(ATHENS_TZ)

    time_threshold = athens_now - timedelta(minutes=TIME_WINDOW_MIN)
    print("athens_now:", athens_now)
    print("max file datetime:", df["Datetime"].max())
    print("min file datetime:", df["Datetime"].min())
    print("time_threshold:", time_threshold)

    before = len(df)
    df = df[df["Datetime"] >= time_threshold].copy()
    print(f"⏱️ time filter (last {TIME_WINDOW_MIN} min): {before} -> {len(df)} (removed {before - len(df)})")
    if df.empty:
        print("⚠️ No data points available after filters. Exiting.")
        return

    # Optional Greece-only filter
    if "Country" in df.columns:
        before = len(df)
        c = df["Country"].astype(str).str.strip().str.upper()
        df = df[c.isin(["GR", "GREECE"])].copy()
        print(f"🇬🇷 Country filter: {before} -> {len(df)} (removed {before - len(df)})")

    print_latest_rows(df, n=8, title="(After time/country preview)")

    lats = df["Latitude"].values.astype(float)
    lons = df["Longitude"].values.astype(float)
    intens = df["RainIntensity"].values.astype(float)
    tnow = df["TNow"].values.astype(float)

    # grid in degrees
    grid_x, grid_y = np.meshgrid(
        np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_N),
        np.linspace(GRID_LAT_MIN, GRID_LAT_MAX, GRID_N)
    )

    # geometry mask
    greece = gpd.read_file(GREECE_GEOJSON_PATH)
    if greece.crs is None:
        greece = greece.set_crs("EPSG:4326")
    if greece.crs.to_string() != "EPSG:4326":
        greece = greece.to_crs("EPSG:4326")

    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()),
        crs="EPSG:4326"
    )
    try:
        boundary = greece.geometry.union_all()
    except AttributeError:
        boundary = greece.geometry.unary_union
    geo_mask = grid_points.geometry.within(boundary).values.reshape(grid_x.shape)

    # distance mask (degrees)
    st_tree = cKDTree(np.c_[lons, lats])
    dists, _ = st_tree.query(np.c_[grid_x.ravel(), grid_y.ravel()])
    distance_mask = dists.reshape(grid_x.shape) <= DISTANCE_MASK_DEG

    final_mask = geo_mask & distance_mask

    # interpolate intensity
    grid_intensity = idw_any_neighbor(
        lons, lats, intens, grid_x, grid_y,
        power=IDW_POWER,
        max_distance=MAX_DISTANCE_DEG,
        k=IDW_K
    )

    masked_intensity = np.full(grid_x.shape, np.nan, dtype=float)
    masked_intensity[final_mask] = grid_intensity[final_mask]

    # local regression temperature grid
    station_alt = sample_altitude_vrt_m(ALT_VRT_PATH, lons, lats)
    grid_tnow, grid_b, grid_a, b_global = build_temperature_grid_local_lr_wgs(
        GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX,
        grid_x, grid_y,
        lons, lats,
        tnow, station_alt,
        ALT_VRT_PATH,
        grid_n=GRID_N
    )

    snow_mask = (
        final_mask &
        np.isfinite(masked_intensity) &
        np.isfinite(grid_tnow) &
        (masked_intensity > RAIN_THRESH) &
        (grid_tnow <= SNOW_T_C)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        iso_alt_m = (SNOW_T_C - grid_a) / grid_b
    iso_alt_m = np.where(np.isfinite(iso_alt_m), iso_alt_m, np.nan)
    iso_alt_m = np.clip(iso_alt_m, ISO_ALT_MIN_M, ISO_ALT_MAX_M)

    # outputs
    ts = athens_now.strftime("%Y-%m-%d-%H-%M")
    prefix = "rain_intensity_"
    latest_name = "latest.png"

    out_png = os.path.join(output_dir, f"{prefix}{ts}.png")
    out_latest = os.path.join(output_dir, latest_name)

    # plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    arr = ma.masked_invalid(masked_intensity)

    img = ax.imshow(
        arr,
        extent=(GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX),
        origin="lower",
        cmap=CMAP_GREECE_EGSA,
        norm=NORM_GREECE_EGSA,
        alpha=0.7
    )

    greece.boundary.plot(ax=ax, color="black", linewidth=0.5)

    contour_levels = [0.1, 2, 6, 36, 100]
    ax.contour(grid_x, grid_y, masked_intensity, levels=contour_levels, colors="black", linewidths=1)

    sx, sy = downsample_mask_points(
        grid_x, grid_y, snow_mask,
        max_points=MAX_SNOWFLAKES,
        min_sep=MIN_SEP_DEG,
        seed=SNOW_SEED
    )

    for x, y in zip(sx, sy):
        ax.text(
            x, y, "❄",
            ha="center", va="center",
            fontsize=SNOW_FONTSIZE,
            color="black",
            path_effects=[pe.withStroke(linewidth=SNOW_STROKE_W, foreground="white")]
        )

    if len(sx) > 0:
        print(f"❄ Snow flags placed at {len(sx)} point(s) (min_sep={MIN_SEP_DEG}).")
    else:
        print("❄ No snow grid points found with current thresholds.")

    # labels
    if len(sx) > 0:
        x_axis = grid_x[0, :]
        y_axis = grid_y[:, 0]
        for k, (x, y) in enumerate(zip(sx, sy)):
            if LABEL_EVERY_N > 1 and (k % LABEL_EVERY_N) != 0:
                continue
            j = int(np.argmin(np.abs(x_axis - x)))
            i = int(np.argmin(np.abs(y_axis - y)))
            z = iso_alt_m[i, j]
            if not np.isfinite(z):
                continue
            z_round = int(round(z / ISO_ALT_ROUND_M) * ISO_ALT_ROUND_M)
            ax.text(
                x, y - LABEL_Y_OFFSET_DEG,
                f"{z_round}m+",
                ha="center", va="top",
                fontsize=LABEL_FONTSIZE,
                color="black",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]
            )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax, boundaries=BOUNDS_GREECE_EGSA, extend="both")
    cbar.set_ticks([2, 6, 36, 100])
    cbar.set_ticklabels(["2", "6", "36", "100"])
    cbar.set_label("Ραγδαιότητα υετού (mm/h)", fontsize=12)

    ax.set_title("Υπολογ. τελευταία διαθέσιμη ραγδαιότητα υετού", fontsize=14, pad=10, loc="center")
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    timestamp_text = athens_now.strftime("%Y-%m-%d %H:%M %Z")
    left_text = f"Δημιουργήθηκε για το e-kairos.gr\n{timestamp_text}"
    right_text = f"v4.0-local\nΧιονόπτωση ≤ {SNOW_T_C:.1f}°C\nΣυνολ. βαροβαθμιδα: {b_global*1000:.2f} °C/km"

    ax.text(
        0.01, 0.01, left_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )
    ax.text(
        0.99, 0.01, right_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.92)

    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    shutil.copy(out_png, out_latest)
    plt.close(fig)

    print(f"✅ Saved: {out_png}")
    print(f"✅ Saved: {out_latest}")

    # FTP
    try:
        ftp_upload_file(out_png)
        ftp_upload_file(out_latest)
        ftp_prune_timestamped(prefix=prefix, latest_name=latest_name, keep=200)
    except Exception as e:
        print(f"⚠️ FTP upload/prune failed: {e}")


# =============================================================================
# REGION RUNNERS: EGSA GENERIC (Attica/Crete/NE Greece)
# =============================================================================
def run_egsa_region(cfg: dict):
    """
    cfg keys:
      name, bbox (lon_min, lon_max, lat_min, lat_max),
      prefix, latest_name,
      time_window_min,
      grid_n,
      idw: power, k, max_distance_m, distance_mask_m,
      snow: min_sep_m, max_snowflakes, fontsize, stroke_w, seed,
      labels: every_n, min_sep_m, y_offset_m, fontsize,
      footer_version
    """
    print("\n====================")
    print(f"RUN: {cfg['name']} (EGSA87 EPSG:2100)")
    print("====================")

    ensure_geojson_and_altitude_bundle()
    if not os.path.exists(GREECE_GEOJSON_PATH):
        print(f"❌ Missing: {GREECE_GEOJSON_PATH}")
        return
    if not os.path.exists(ALT_VRT_PATH):
        print(f"❌ Missing: {ALT_VRT_PATH}")
        return

    CRS_WGS84 = "EPSG:4326"
    CRS_EGSA87 = "EPSG:2100"

    lon_min, lon_max, lat_min, lat_max = cfg["bbox"]
    GRID_N = int(cfg.get("grid_n", 300))

    # transforms
    wgs_to_egsa = Transformer.from_crs(CRS_WGS84, CRS_EGSA87, always_xy=True)
    egsa_to_wgs = Transformer.from_crs(CRS_EGSA87, CRS_WGS84, always_xy=True)

    # bbox corners to EGSA
    corners_lon = [lon_min, lon_min, lon_max, lon_max]
    corners_lat = [lat_min, lat_max, lat_min, lat_max]
    corners_x, corners_y = wgs_to_egsa.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(corners_x)), float(np.max(corners_x))
    y_min, y_max = float(np.min(corners_y)), float(np.max(corners_y))

    output_dir = os.path.join(BASE_DIR, "rainintensitymaps")
    os.makedirs(output_dir, exist_ok=True)
    cache_txt = os.path.join(BASE_DIR, "weathernow_cached.txt")

    # fetch
    try:
        text, source = robust_fetch_text(CURRENTWEATHER_URL, cache_txt=cache_txt, timeout=60, tries=6)
        print("source:", source)
        with open(cache_txt, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return

    df = load_and_clean_feed(text, lon_min=19, lon_max=30)

    print(f"📥 Raw(cleaned) rows: {len(df)}")

    # time filter
    athens_now = datetime.now(ATHENS_TZ)
    tw = int(cfg.get("time_window_min", 45))
    time_threshold = athens_now - timedelta(minutes=tw)

    print("athens_now:", athens_now)
    print("max file datetime:", df["Datetime"].max())
    print("min file datetime:", df["Datetime"].min())
    print("time_threshold:", time_threshold)

    before = len(df)
    df = df[df["Datetime"] >= time_threshold].copy()
    print(f"⏱️ time filter (last {tw} min): {before} -> {len(df)} (removed {before - len(df)})")

    # Greece only if column exists
    if "Country" in df.columns:
        c = df["Country"].astype(str).str.strip().str.upper()
        df = df[c.isin(["GR", "GREECE"])].copy()

    if df.empty:
        print("⚠️ No data points available within the time window. Exiting.")
        return

    # station arrays
    lats = df["Latitude"].values.astype(float)
    lons = df["Longitude"].values.astype(float)
    intens = df["RainIntensity"].values.astype(float)
    tnow = df["TNow"].values.astype(float)

    # project stations to EGSA
    st_x, st_y = wgs_to_egsa.transform(lons.tolist(), lats.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    # grid in EGSA meters
    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, GRID_N),
        np.linspace(y_min, y_max, GRID_N)
    )

    # load greece geometry in EGSA, clip to bbox
    greece = gpd.read_file(GREECE_GEOJSON_PATH)
    if greece.crs is None:
        raise RuntimeError("Greece GeoJSON has no CRS.")
    if greece.crs.to_string() != CRS_EGSA87:
        greece = greece.to_crs(CRS_EGSA87)

    bbox_poly = box(x_min, y_min, x_max, y_max)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs=CRS_EGSA87)

    try:
        greece_clip = gpd.clip(greece, bbox_gdf)
    except Exception as e:
        print(f"⚠️ gpd.clip failed ({e}); plotting full Greece but forcing view limits.")
        greece_clip = greece

    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_EGSA87
    )
    try:
        boundary = greece_clip.geometry.union_all()
    except AttributeError:
        boundary = greece_clip.geometry.unary_union
    geo_mask = grid_points.geometry.within(boundary).values.reshape(grid_x_m.shape)

    # distance mask in meters
    dist_mask_m = float(cfg["idw"]["distance_mask_m"])
    st_tree = cKDTree(np.c_[st_x, st_y])
    dists, _ = st_tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    distance_mask = dists.reshape(grid_x_m.shape) <= dist_mask_m

    final_mask = geo_mask & distance_mask

    # interpolate intensity (IDW in meters)
    grid_intensity = idw_any_neighbor(
        st_x, st_y, intens,
        grid_x_m, grid_y_m,
        power=float(cfg["idw"]["power"]),
        max_distance=float(cfg["idw"]["max_distance_m"]),
        k=int(cfg["idw"]["k"])
    )

    masked_intensity = np.full(grid_x_m.shape, np.nan, dtype=float)
    masked_intensity[final_mask] = grid_intensity[final_mask]

    # temperature local regression
    station_alt = sample_altitude_vrt_m(ALT_VRT_PATH, lons, lats)
    grid_tnow, grid_b, grid_a, b_global = build_temperature_grid_local_lr_projected(
        grid_x_m, grid_y_m,
        st_x, st_y,
        tnow, station_alt,
        egsa_to_wgs,
        ALT_VRT_PATH,
        x_min, x_max, y_min, y_max,
        grid_n=GRID_N
    )

    snow_mask = (
        final_mask &
        np.isfinite(masked_intensity) &
        np.isfinite(grid_tnow) &
        (masked_intensity > RAIN_THRESH) &
        (grid_tnow <= SNOW_T_C)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        iso_alt_m = (SNOW_T_C - grid_a) / grid_b
    iso_alt_m = np.where(np.isfinite(iso_alt_m), iso_alt_m, np.nan)
    iso_alt_m = np.clip(iso_alt_m, ISO_ALT_MIN_M, ISO_ALT_MAX_M)

    # output
    ts = athens_now.strftime("%Y-%m-%d-%H-%M")
    prefix = cfg["prefix"]
    latest_name = cfg["latest_name"]
    out_png = os.path.join(output_dir, f"{prefix}{ts}.png")
    out_latest = os.path.join(output_dir, latest_name)

    # plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    arr = ma.masked_invalid(masked_intensity)

    img = ax.imshow(
        arr,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=CMAP_GREECE_EGSA,
        norm=NORM_GREECE_EGSA,
        alpha=0.7
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    # show lon/lat degrees on axes (labels only)
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = egsa_to_wgs.transform(x, y_ref_for_lon)
        return f"{lon:.2f}"

    def fmt_lat(y, pos):
        _lon, lat = egsa_to_wgs.transform(x_ref_for_lat, y)
        return f"{lat:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    contour_levels = [0.1, 2, 6, 36, 100]
    ax.contour(grid_x_m, grid_y_m, masked_intensity, levels=contour_levels, colors="black", linewidths=1)

    # snowflakes
    snow_cfg = cfg["snow"]
    sx, sy = downsample_mask_points(
        grid_x_m, grid_y_m, snow_mask,
        max_points=int(snow_cfg.get("max_snowflakes", DEFAULT_MAX_SNOWFLAKES)),
        min_sep=float(snow_cfg.get("min_sep_m", 4500.0)),
        seed=int(snow_cfg.get("seed", DEFAULT_SNOW_SEED))
    )

    for x, y in zip(sx, sy):
        ax.text(
            x, y, "❄",
            ha="center", va="center",
            fontsize=float(snow_cfg.get("fontsize", DEFAULT_SNOW_FONTSIZE)),
            color="black",
            path_effects=[pe.withStroke(linewidth=float(snow_cfg.get("stroke_w", DEFAULT_SNOW_STROKE_W)), foreground="white")]
        )

    if len(sx) > 0:
        print(f"❄ Snow flags placed at {len(sx)} point(s).")
    else:
        print("❄ No snow grid points found with current thresholds.")

    # labels with crowd-control
    lab_cfg = cfg["labels"]
    if len(sx) > 0:
        x_axis = grid_x_m[0, :]
        y_axis = grid_y_m[:, 0]

        every_n = int(lab_cfg.get("every_n", DEFAULT_LABEL_EVERY_N))
        min_sep = float(lab_cfg.get("min_sep_m", 12000.0))
        y_off = float(lab_cfg.get("y_offset_m", 3000.0))
        fz = float(lab_cfg.get("fontsize", DEFAULT_LABEL_FONTSIZE))

        label_pts = []
        label_tree = None

        for k, (x, y) in enumerate(zip(sx, sy)):
            if every_n > 1 and (k % every_n) != 0:
                continue

            if min_sep and label_tree is not None and len(label_pts) > 0:
                d, _ = label_tree.query([x, y], k=1)
                if d < min_sep:
                    continue

            j = int(np.argmin(np.abs(x_axis - x)))
            i = int(np.argmin(np.abs(y_axis - y)))

            z = iso_alt_m[i, j]
            if not np.isfinite(z):
                continue

            z_round = int(round(z / ISO_ALT_ROUND_M) * ISO_ALT_ROUND_M)
            ax.text(
                x, y - y_off,
                f"{z_round}m+",
                ha="center", va="top",
                fontsize=fz,
                color="black",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]
            )

            label_pts.append([x, y])
            if len(label_pts) == 1 or (len(label_pts) % 50 == 0):
                label_tree = cKDTree(np.array(label_pts))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax, boundaries=BOUNDS_GREECE_EGSA, extend="both")
    cbar.set_ticks([2, 6, 36, 100])
    cbar.set_ticklabels(["2", "6", "36", "100"])
    cbar.set_label("Ραγδαιότητα υετού (mm/h)", fontsize=12)

    ax.set_title("Υπολογ. τελευταία διαθέσιμη ραγδαιότητα υετού", fontsize=14, pad=10, loc="center")
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    timestamp_text = athens_now.strftime("%Y-%m-%d %H:%M %Z")
    left_text = f"Δημιουργήθηκε για το e-kairos.gr\n{timestamp_text}"
    right_text = f"{cfg.get('footer_version','v4.x-egsa2100')}\nΧιονόπτωση ≤ {SNOW_T_C:.1f}°C\nΣυνολ. βαροβαθμιδα: {b_global*1000:.2f} °C/km"

    ax.text(
        0.01, 0.01, left_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )
    ax.text(
        0.99, 0.01, right_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.92)

    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    shutil.copy(out_png, out_latest)
    plt.close(fig)

    print(f"✅ Saved: {out_png}")
    print(f"✅ Saved: {out_latest}")

    # FTP
    try:
        ftp_upload_file(out_png)
        ftp_upload_file(out_latest)
        ftp_prune_timestamped(prefix=prefix, latest_name=latest_name, keep=int(cfg.get("remote_keep", 200)))
    except Exception as e:
        print(f"⚠️ FTP upload/prune failed: {e}")


# =============================================================================
# REGION RUNNER: CYPRUS (UTM 36N)
# =============================================================================
def bounds_reasonable(geom, lon_min=31.0, lon_max=36.0, lat_min=34.0, lat_max=36.5):
    try:
        minx, miny, maxx, maxy = geom.bounds
        return (lon_min <= minx <= lon_max) and (lon_min <= maxx <= lon_max) and \
               (lat_min <= miny <= lat_max) and (lat_min <= maxy <= lat_max)
    except Exception:
        return False

def swap_geom(geom):
    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        return Polygon(np.column_stack([y, x]))
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([swap_geom(g) for g in geom.geoms])
    return geom

def plot_boundary_proj(ax, geom, linewidth=0.5, color="black"):
    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.plot(x, y, linewidth=linewidth, color=color, zorder=3)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=linewidth, color=color, zorder=3)

def run_cyprus():
    print("\n====================")
    print("RUN: Cyprus (UTM 36N)")
    print("====================")

    if not os.path.exists(CYPRUS_GEOJSON_PATH):
        print("❌ cyprus.geojson not found (decrypt cyprus.geojson.enc before running).")
        return
    if not os.path.exists(CYPRUS_ALT_TIF_PATH):
        print(f"❌ Missing altitude GeoTIFF: {CYPRUS_ALT_TIF_PATH}")
        return

    # Cyprus config (from your script)
    CRS_WGS84 = "EPSG:4326"
    CRS_UTM = "EPSG:32636"
    GRID_N = 300

    # bbox for station filtering
    LON_MIN, LON_MAX = 32.0, 34.9
    LAT_MIN, LAT_MAX = 34.4, 35.9

    TIME_WINDOW_MIN = 30

    IDW_POWER = 2
    IDW_K = 8
    MAX_DISTANCE_M = 40000.0
    DISTANCE_MASK_M = 40000.0

    MIN_SEP_M = 1500.0
    MAX_SNOWFLAKES = DEFAULT_MAX_SNOWFLAKES
    SNOW_SEED = DEFAULT_SNOW_SEED
    SNOW_FONTSIZE = DEFAULT_SNOW_FONTSIZE
    SNOW_STROKE_W = DEFAULT_SNOW_STROKE_W

    LABEL_EVERY_N = 10
    LABEL_MIN_SEP_M = 12000.0
    LABEL_Y_OFFSET_M = 2500.0
    LABEL_FONTSIZE = 7
    LABEL_STROKE_W = 1.2

    output_dir = os.path.join(BASE_DIR, "cyprusrainintensitymaps")
    os.makedirs(output_dir, exist_ok=True)
    cache_txt = os.path.join(BASE_DIR, "weathernow_cyprus_cached.txt")

    # Load boundary
    cyprus = gpd.read_file(CYPRUS_GEOJSON_PATH)
    if cyprus.crs is None:
        cyprus = cyprus.set_crs(epsg=4326)
    cyprus = cyprus[~cyprus.geometry.is_empty]
    if not cyprus.geometry.is_valid.all():
        cyprus.geometry = cyprus.buffer(0)

    try:
        cyprus_boundary_ll = cyprus.union_all()
    except AttributeError:
        cyprus_boundary_ll = unary_union(cyprus.geometry)

    # Fix possible lat/lon swapped
    if not bounds_reasonable(cyprus_boundary_ll):
        cyprus.geometry = cyprus.geometry.apply(swap_geom)
        try:
            cyprus_boundary_ll = cyprus.union_all()
        except AttributeError:
            cyprus_boundary_ll = unary_union(cyprus.geometry)

    cyprus_utm = cyprus.to_crs(CRS_UTM)
    try:
        cyprus_boundary_utm = cyprus_utm.union_all()
    except AttributeError:
        cyprus_boundary_utm = unary_union(cyprus_utm.geometry)

    # fetch
    try:
        text, source = robust_fetch_text(CURRENTWEATHER_URL, cache_txt=cache_txt, timeout=60, tries=6)
        print("source:", source)
        with open(cache_txt, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"❌ Failed to fetch feed: {e}")
        return

    df = load_and_clean_feed(text)
    print(f"📥 Raw(cleaned) rows: {len(df)}")
    print_latest_rows(df, n=8, title="(Cleaned feed preview)")

    athens_now = datetime.now(ATHENS_TZ)
    time_threshold = athens_now - timedelta(minutes=TIME_WINDOW_MIN)
    print("athens_now:", athens_now)
    print("max file datetime:", df["Datetime"].max())
    print("min file datetime:", df["Datetime"].min())
    print("time_threshold:", time_threshold)

    before = len(df)
    df = df[df["Datetime"] >= time_threshold].copy()
    print(f"⏱️ time filter (last {TIME_WINDOW_MIN} min): {before} -> {len(df)} (removed {before - len(df)})")

    # bbox filter
    before = len(df)
    df = df[
        df["Longitude"].between(LON_MIN, LON_MAX) &
        df["Latitude"].between(LAT_MIN, LAT_MAX)
    ].copy()
    print(f"🗺️ bbox Cyprus lon[{LON_MIN},{LON_MAX}] lat[{LAT_MIN},{LAT_MAX}]: {before} -> {len(df)} (removed {before - len(df)})")

    if df.empty:
        print("⚠️ No data points after filters. Exiting.")
        return

    # stations in UTM
    to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)
    lons = df["Longitude"].values.astype(float)
    lats = df["Latitude"].values.astype(float)
    easts, norths = to_utm.transform(lons, lats)

    intens = df["RainIntensity"].values.astype(float)
    tnow = df["TNow"].values.astype(float)

    # grid over bbox corners
    corn_lon = np.array([LON_MIN, LON_MAX, LON_MIN, LON_MAX])
    corn_lat = np.array([LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX])
    corn_E, corn_N = to_utm.transform(corn_lon, corn_lat)
    E_MIN, E_MAX = float(np.min(corn_E)), float(np.max(corn_E))
    N_MIN, N_MAX = float(np.min(corn_N)), float(np.max(corn_N))

    grid_E, grid_N = np.meshgrid(
        np.linspace(E_MIN, E_MAX, GRID_N),
        np.linspace(N_MIN, N_MAX, GRID_N)
    )

    # interpolate intensity
    grid_intensity = idw_any_neighbor(
        easts, norths, intens,
        grid_E, grid_N,
        power=IDW_POWER,
        max_distance=MAX_DISTANCE_M,
        k=IDW_K
    )

    # mask inside boundary + within distance
    grid_points_utm = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_E.ravel(), grid_N.ravel()),
        crs=CRS_UTM
    )
    geo_mask = grid_points_utm.geometry.within(cyprus_boundary_utm).values.reshape(grid_E.shape)

    st_tree = cKDTree(np.c_[easts, norths])
    dists, _ = st_tree.query(np.c_[grid_E.ravel(), grid_N.ravel()])
    distance_mask = dists.reshape(grid_E.shape) <= DISTANCE_MASK_M
    final_mask = geo_mask & distance_mask

    masked_intensity = np.full(grid_E.shape, np.nan, dtype=float)
    masked_intensity[final_mask] = grid_intensity[final_mask]

    # altitude + lapse (Cyprus uses global lapse only, per your script)
    station_alt = sample_altitude_raster_m(CYPRUS_ALT_TIF_PATH, lons, lats, input_crs=CRS_WGS84)
    lapse = fit_global_lapse_rate(tnow, station_alt)

    ok_t = np.isfinite(tnow) & np.isfinite(station_alt)
    t0 = np.full_like(tnow, np.nan, dtype=float)
    t0[ok_t] = tnow[ok_t] - lapse * station_alt[ok_t]

    grid_t0 = idw_any_neighbor(
        easts, norths, t0,
        grid_E, grid_N,
        power=IDW_POWER,
        max_distance=MAX_DISTANCE_M,
        k=IDW_K
    )

    grid_alt = sample_altitude_raster_m(
        CYPRUS_ALT_TIF_PATH,
        grid_E.ravel(), grid_N.ravel(),
        input_crs=CRS_UTM
    ).reshape(grid_E.shape)

    grid_tnow_adj = np.full(grid_E.shape, np.nan, dtype=float)
    ok_grid = np.isfinite(grid_t0) & np.isfinite(grid_alt)
    grid_tnow_adj[ok_grid] = grid_t0[ok_grid] + lapse * grid_alt[ok_grid]

    snow_mask = (
        final_mask &
        np.isfinite(masked_intensity) &
        np.isfinite(grid_tnow_adj) &
        (masked_intensity > RAIN_THRESH) &
        (grid_tnow_adj <= SNOW_T_C)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        iso_alt_m = (SNOW_T_C - grid_t0) / lapse
    iso_alt_m = np.where(final_mask & np.isfinite(iso_alt_m), iso_alt_m, np.nan)
    iso_alt_m = np.clip(iso_alt_m, ISO_ALT_MIN_M, ISO_ALT_MAX_M)

    # output names
    ts = athens_now.strftime("%Y-%m-%d-%H-%M")
    prefix = "cyprusrainintensity"
    latest_name = "cyprusrainintensity_latest.png"
    out_png = os.path.join(output_dir, f"{prefix}{ts}.png")
    out_latest = os.path.join(output_dir, latest_name)

    # plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # degree-like tick labels while plotting in meters
    from matplotlib.ticker import FixedLocator
    lon0 = (LON_MIN + LON_MAX) / 2.0
    lat0 = (LAT_MIN + LAT_MAX) / 2.0
    lon_step = 0.5
    lat_step = 0.5

    lon_ticks = np.arange(np.floor(LON_MIN / lon_step) * lon_step, LON_MAX + 1e-9, lon_step)
    lat_ticks = np.arange(np.floor(LAT_MIN / lat_step) * lat_step, LAT_MAX + 1e-9, lat_step)

    x_ticks_m, _ = to_utm.transform(lon_ticks, np.full_like(lon_ticks, lat0))
    _, y_ticks_m = to_utm.transform(np.full_like(lat_ticks, lon0), lat_ticks)

    ax.xaxis.set_major_locator(FixedLocator(x_ticks_m))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks_m))
    ax.set_xticklabels([f"{lon:.2f}°E" for lon in lon_ticks])
    ax.set_yticklabels([f"{lat:.2f}°N" for lat in lat_ticks])
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    arr = ma.masked_invalid(masked_intensity)
    img = ax.imshow(
        arr,
        extent=(E_MIN, E_MAX, N_MIN, N_MAX),
        origin="lower",
        cmap=CMAP_CYPRUS,
        norm=NORM_CYPRUS,
        alpha=0.7
    )

    plot_boundary_proj(ax, cyprus_boundary_utm, linewidth=0.5, color="black")
    ax.set_aspect("equal")

    contour_levels = [0.2, 2, 6, 40]
    ax.contour(grid_E, grid_N, masked_intensity, levels=contour_levels, colors="black", linewidths=1)

    # snowflakes
    sx, sy = downsample_mask_points(
        grid_E, grid_N, snow_mask,
        max_points=MAX_SNOWFLAKES,
        min_sep=MIN_SEP_M,
        seed=SNOW_SEED
    )
    for x, y in zip(sx, sy):
        ax.text(
            x, y, "❄",
            ha="center", va="center",
            fontsize=SNOW_FONTSIZE,
            color="black",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=SNOW_STROKE_W, foreground="white")]
        )

    if len(sx) > 0:
        print(f"❄ Snow flags placed at {len(sx)} point(s).")
    else:
        print("❄ No snow grid points found with current thresholds.")

    # snowline altitude labels
    if len(sx) > 0:
        x_axis = grid_E[0, :]
        y_axis = grid_N[:, 0]

        label_pts = []
        label_tree = None

        for k, (x, y) in enumerate(zip(sx, sy)):
            if LABEL_EVERY_N > 1 and (k % LABEL_EVERY_N) != 0:
                continue

            if LABEL_MIN_SEP_M and label_tree is not None and len(label_pts) > 0:
                d, _ = label_tree.query([x, y], k=1)
                if d < LABEL_MIN_SEP_M:
                    continue

            j = int(np.argmin(np.abs(x_axis - x)))
            i = int(np.argmin(np.abs(y_axis - y)))

            z = iso_alt_m[i, j]
            if not np.isfinite(z):
                continue

            z_round = int(round(z / ISO_ALT_ROUND_M) * ISO_ALT_ROUND_M)
            z_round = max(ISO_ALT_MIN_M, min(ISO_ALT_MAX_M, z_round))

            ax.text(
                x, y - LABEL_Y_OFFSET_M,
                f"{z_round}m+",
                ha="center", va="top",
                fontsize=LABEL_FONTSIZE,
                color="black",
                zorder=6,
                path_effects=[pe.withStroke(linewidth=LABEL_STROKE_W, foreground="white")]
            )

            label_pts.append([x, y])
            if len(label_pts) == 1 or (len(label_pts) % 50 == 0):
                label_tree = cKDTree(np.array(label_pts))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label("Ραγδαιότητα υετού (mm/h)", fontsize=12)

    ax.set_title("Υπολογ. τελευταία διαθέσιμη ραγδαιότητα υετού", fontsize=14, pad=10, loc="center")
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    timestamp_text = athens_now.strftime("Δημιουργήθηκε: %Y-%m-%d %H:%M %Z για το e-kairos.gr")
    ax.text(
        0.01, 0.01, timestamp_text,
        transform=ax.transAxes,
        fontsize=10,
        color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    lapse_text = f"Lapse: {lapse*1000:.2f} °C/km | Snow labels: T≤{SNOW_T_C:.1f}°C"
    ax.text(
        0.99, 0.01, lapse_text,
        transform=ax.transAxes,
        fontsize=10,
        color="black",
        ha="right", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.92)

    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    shutil.copy(out_png, out_latest)
    plt.close(fig)

    print("✅ Saved:", out_png)
    print("✅ Saved:", out_latest)

    # FTP (optional). Cyprus prune pattern differs, so keep it separate:
    try:
        ftp_upload_file(out_png)
        ftp_upload_file(out_latest)
        # prune cyprus timestamped
        ftp_prune_timestamped(prefix=prefix, latest_name=latest_name, keep=144)
    except Exception as e:
        print(f"⚠️ FTP upload/prune failed: {e}")


# =============================================================================
# REGION CONFIGS
# =============================================================================
EGSA_REGIONS = {
    "attica": {
        "name": "Attica",
        "bbox": (22.7, 25.0, 37.5, 38.7),
        "prefix": "rain_intensity_attica_",
        "latest_name": "latestattica.png",
        "remote_keep": 200,
        "time_window_min": 45,
        "grid_n": 300,
        "idw": {
            "power": 2,
            "k": 8,
            "max_distance_m": 120_000,
            "distance_mask_m": 170_000,
        },
        "snow": {
            "min_sep_m": 4_500,
            "max_snowflakes": 5000,
            "fontsize": 6,
            "stroke_w": 1.2,
            "seed": 123,
        },
        "labels": {
            "every_n": 10,
            "min_sep_m": 12_000,
            "y_offset_m": 3_000,
            "fontsize": 6,
        },
        "footer_version": "v4.2-egsa2100",
    },
    "crete": {
        "name": "Crete",
        "bbox": (23.0, 26.5, 34.5, 36.0),
        "prefix": "rain_intensity_crete_",
        "latest_name": "latestcrete.png",
        "remote_keep": 200,
        "time_window_min": 45,
        "grid_n": 300,
        "idw": {
            "power": 2,
            "k": 8,
            "max_distance_m": 120_000,
            "distance_mask_m": 170_000,
        },
        "snow": {
            "min_sep_m": 4_500,
            "max_snowflakes": 5000,
            "fontsize": 6,
            "stroke_w": 1.2,
            "seed": 123,
        },
        "labels": {
            "every_n": 10,
            "min_sep_m": 12_000,
            "y_offset_m": 3_000,
            "fontsize": 6,
        },
        "footer_version": "v4.2-egsa2100",
    },
    "negreece": {
        "name": "NE Greece",
        "bbox": (22.0, 26.6, 39.8, 41.9),
        "prefix": "rain_intensity_negreece_",
        "latest_name": "latestnegreece.png",
        "remote_keep": 200,
        "time_window_min": 45,
        "grid_n": 300,
        "idw": {
            "power": 2,
            "k": 8,
            "max_distance_m": 120_000,
            "distance_mask_m": 170_000,
        },
        "snow": {
            "min_sep_m": 4_500,
            "max_snowflakes": 5000,
            "fontsize": 6,
            "stroke_w": 1.2,
            "seed": 123,
        },
        "labels": {
            "every_n": 10,
            "min_sep_m": 12_000,
            "y_offset_m": 3_000,
            "fontsize": 6,
        },
        "footer_version": "v4.2-egsa2100",
    },
}


# =============================================================================
# MAIN CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        default="all",
        choices=["all", "greece", "attica", "crete", "negreece", "cyprus"],
        help="Which region to run."
    )
    args = parser.parse_args()

    if args.region in ("all", "greece"):
        run_greece()

    if args.region in ("all", "attica"):
        run_egsa_region(EGSA_REGIONS["attica"])

    if args.region in ("all", "crete"):
        run_egsa_region(EGSA_REGIONS["crete"])

    if args.region in ("all", "negreece"):
        run_egsa_region(EGSA_REGIONS["negreece"])

    if args.region in ("all", "cyprus"):
        run_cyprus()


if __name__ == "__main__":
    main()
