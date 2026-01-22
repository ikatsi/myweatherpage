#!/usr/bin/env python3
# rain_intensity_greece_snow_altitude_labels.py
#
# Greece rain intensity map + snowflake overlay
# Adds: at snowflake locations, labels the altitude (m a.s.l.) above which T <= SNOW_T_C
# Uses: spatially varying local lapse (distance-weighted local regression on altitude), with global fallback
#
# Debug additions (same style as your Attica/Cyprus debugging):
# - robust_fetch_text() with retries, curl fallback, cached fallback + prints source
# - prints Athens now, max/min Datetime in feed, time threshold
# - prints row counts after each filter step
# - shows latest rows from the remote feed and after filtering
#
# IMPORTANT FIX:
# - IDW validity was too strict (np.all finite distances). Now uses "any finite neighbor" per grid cell,
#   so you don't lose big areas just because not all K neighbors are within max_distance.

import os
import re
import shutil
import time
import random
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

from scipy.spatial import cKDTree
from scipy.ndimage import zoom
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.patheffects as pe

import rasterio
from rasterio.warp import transform as rio_transform
from pyproj import Transformer
from ftplib import FTP_TLS

import requests


# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Files live at repo root (no 'vectors' folder)
GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
ALT_VRT_PATH = os.path.join(BASE_DIR, "GRC_alt.vrt")

# Encrypted bundles live at repo root
ALT_ENC   = os.path.join(BASE_DIR, "altitude.zip.enc")
ALT_ZIP   = os.path.join(BASE_DIR, "altitude.zip")
GEOJSON_ENC = os.path.join(BASE_DIR, "greece.geojson.enc")
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()  # password to decrypt


def ensure_altitude_bundle():
    # If the VRT is already present at repo root, nothing to do
    if os.path.exists(ALT_VRT_PATH):
        return

    # Try to decrypt and unzip if the encrypted bundle exists at repo root
    if not os.path.exists(ALT_ENC):
        return
    if not GEOJSON_PASS:
        raise SystemExit("DEM bundle missing and GEOJSON_PASS not set to decrypt altitude.zip.enc")

    # Decrypt altitude.zip.enc → altitude.zip at repo root
    try:
        subprocess.check_call([
            "openssl","enc","-d","-aes-256-cbc","-pbkdf2",
            "-in", ALT_ENC, "-out", ALT_ZIP, "-pass", "pass:" + GEOJSON_PASS
        ])
    except FileNotFoundError:
        raise SystemExit("OpenSSL not found. Install it or decrypt altitude.zip.enc in a prior CI step.")
    except subprocess.CalledProcessError as e:
        raise SystemExit("OpenSSL decryption failed for altitude.zip.enc: %s" % e)

    # Unzip into repo root so GRC_alt.vrt, .grd, .gri land next to the script
    import zipfile
    with zipfile.ZipFile(ALT_ZIP, "r") as zf:
        zf.extractall(BASE_DIR)

    # Verify VRT exists right after extraction
    if not os.path.exists(ALT_VRT_PATH):
        raise SystemExit("Decrypted bundle didn’t contain GRC_alt.vrt at repo root. Check ALT_VRT_PATH or the zip contents.")

    # Remove the plaintext zip
    try:
        os.remove(ALT_ZIP)
    except Exception:
        pass

def ensure_geojson():
    # If plain geojson already present, nothing to do
    if os.path.exists(GEOJSON_PATH):
        return
    # If no encrypted geojson, we cannot help here
    if not os.path.exists(GEOJSON_ENC):
        return
    if not GEOJSON_PASS:
        raise SystemExit("GeoJSON missing and GEOJSON_PASS not set to decrypt greece.geojson.enc")
    try:
        subprocess.check_call([
            "openssl","enc","-d","-aes-256-cbc","-pbkdf2",
            "-in", GEOJSON_ENC, "-out", GEOJSON_PATH, "-pass", "pass:" + GEOJSON_PASS
        ])
    except FileNotFoundError:
        raise SystemExit("OpenSSL not found. Install it or decrypt greece.geojson.enc in a prior CI step.")
    except subprocess.CalledProcessError as e:
        raise SystemExit("OpenSSL decryption failed for greece.geojson.enc: %s" % e)

# Feed URL ONLY from secret (no default). If missing, exit.
RAIN_URL = os.environ.get("CURRENTWEATHER_URL", "").strip()
if not RAIN_URL:
    raise SystemExit("CURRENTWEATHER_URL secret not set.")
CACHE_TXT = os.path.join(BASE_DIR, "weathernow_cached.txt")


# FTP settings ONLY from env; if any is missing, uploads are disabled
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()

def ftp_enabled():
    return bool(FTP_HOST and FTP_USER and FTP_PASS)


# Output naming
PREFIX = "rain_intensity_"
LATEST_NAME = "latest.png"

# Snow definition
SNOW_T_C = 2.0
RAIN_THRESH = 0.0

# Time filter window (minutes)
TIME_WINDOW_MIN = 45

# Grid and IDW settings (degrees)
GRID_N = 300
GRID_LON_MIN, GRID_LON_MAX = 19.0, 30.0
GRID_LAT_MIN, GRID_LAT_MAX = 34.5, 42.5
IDW_POWER = 2
IDW_K = 8
MAX_DISTANCE_DEG = 1.0
DISTANCE_MASK_DEG = 1.5

# Snowflake display controls
SNOW_FONTSIZE = 6
SNOW_STROKE_W = 1.2
MIN_SEP_DEG = 0.04
MAX_SNOWFLAKES = 5000
SNOW_SEED = 123

# Label altitude at snowflake locations
LABEL_EVERY_N_SNOWFLAKES = 25
LABEL_FONTSIZE = 6
LABEL_Y_OFFSET_DEG = 0.03
ISO_ALT_ROUND_M = 50
ISO_ALT_MIN_M = 0
ISO_ALT_MAX_M = 5000

# Lapse rate bounds (degC per meter)
LAPSE_DEFAULT = -0.0065
LAPSE_MIN = -0.0120
LAPSE_MAX = -0.0010

# Spatially varying lapse settings
TEMP_COARSE_N = 120
K_LOCAL = 25
R_LOCAL_M = 150_000
ALT_RANGE_MIN_M = 400
MIN_NBR = 8
USE_DISTANCE_WEIGHTS = True


# === DEBUG HELPERS ===
def print_latest_rows(df: pd.DataFrame, n: int = 8, title: str = "") -> None:
    if df is None or df.empty or "Datetime" not in df.columns:
        if title:
            print(title)
        print("ℹ️ No rows to preview.")
        return

    cols = []
    for c in ["Datetime", "webcode", "Latitude", "Longitude", "TNow", "RainIntensity"]:
        if c in df.columns:
            cols.append(c)

    tmp = df.dropna(subset=["Datetime"]).copy()
    tmp = tmp.sort_values("Datetime", ascending=False).head(n)

    if title:
        print(title)
    print(f"🕒 Latest {min(n, len(tmp))} rows (sorted by Datetime desc):")
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


# === FTP HELPERS ===
def upload_to_ftp(local_file: str) -> None:
    if not ftp_enabled():
        print("ℹ️ FTP disabled (missing env). Skipping upload.")
        return

    remote_filename = os.path.basename(local_file)

    ftps = FTP_TLS()
    ftps.connect(FTP_HOST, 21, timeout=30)
    ftps.login(user=FTP_USER, passwd=FTP_PASS)
    ftps.prot_p()

    try:
        with open(local_file, "rb") as f:
            ftps.storbinary("STOR " + remote_filename, f)
        print(f"📤 Uploaded: {remote_filename}")
    finally:
        try:
            ftps.quit()
        except Exception:
            pass



def prune_remote_pngs(keep: int = 40, prefix: str = PREFIX, latest_name: str = LATEST_NAME) -> None:
    """
    Remote prune:
      - Keep latest.png
      - Keep newest `keep` files matching rain_intensity_YYYY-MM-DD-HH-MM.png
      - Delete older ones
    """
    if not ftp_enabled():
        print("ℹ️ FTP disabled (missing env). Skipping remote prune.")
        return

    pat = re.compile(rf"^{re.escape(prefix)}\d{{4}}-\d{{2}}-\d{{2}}-\d{{2}}-\d{{2}}\.png$")

    ftps = FTP_TLS()
    ftps.connect(FTP_HOST, 21, timeout=30)
    ftps.login(user=FTP_USER, passwd=FTP_PASS)
    ftps.prot_p()

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


# === FETCH HELPERS ===
def robust_fetch_text(url: str, timeout: int = 60, tries: int = 6):
    """
    Returns (text, source) where source is one of: "network", "curl", "cache"
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

    last_err = None
    session = requests.Session()

    for attempt in range(1, tries + 1):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            text = r.text
            if not text or "Datetime" not in text:
                raise RuntimeError("Downloaded content looks empty or malformed.")
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
        text = cp.stdout
        if text and "Datetime" in text:
            return text, "curl"
        raise RuntimeError("curl returned empty/malformed content.")
    except Exception as e:
        print(f"[fetch] curl fallback failed: {e}")

    # cached fallback
    if os.path.exists(CACHE_TXT):
        print(f"[fetch] using cached file: {CACHE_TXT}")
        with open(CACHE_TXT, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "cache"

    # Never leak the URL
    raise RuntimeError("Failed to fetch feed") from last_err



# === ALTITUDE SAMPLING ===
def sample_altitude_m(vrt_path: str, lons, lats) -> np.ndarray:
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


# === IDW (FIXED validity) ===
def idw_optimized(x, y, z, xi, yi, power=2, max_distance=1.0, k=8) -> np.ndarray:
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


# === Minimum-distance thinning for snowflakes ===
def downsample_mask_points(grid_x, grid_y, mask, max_points=5000, min_sep_deg=0.04, seed=123):
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
            if d >= min_sep_deg:
                kept.append(p)
                if len(kept) % 150 == 0:
                    kept_tree = cKDTree(np.array(kept))

        if len(kept) >= max_points:
            break

    kept = np.array(kept)
    return kept[:, 0], kept[:, 1]


# === Local regression temperature field (spatially varying lapse) ===
def build_temperature_grid_local_lr(
    lon_min, lon_max, lat_min, lat_max,
    grid_lon, grid_lat,
    station_lons, station_lats,
    station_tnow, station_alt_m,
    alt_vrt_path,
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
        temp_nan = np.full_like(grid_lon, np.nan, dtype=float)
        b_nan = np.full_like(grid_lon, np.nan, dtype=float)
        a_nan = np.full_like(grid_lon, np.nan, dtype=float)
        return temp_nan, b_nan, a_nan, LAPSE_DEFAULT

    b_global = fit_global_lapse_rate(st_t, st_alt)
    t0_global = np.nanmedian(st_t - b_global * st_alt)

    # Coarse lon/lat grid for local fits
    c_lon = np.linspace(lon_min, lon_max, TEMP_COARSE_N)
    c_lat = np.linspace(lat_min, lat_max, TEMP_COARSE_N)
    c_grid_lon, c_grid_lat = np.meshgrid(c_lon, c_lat)

    # Altitude at coarse points
    c_alt = sample_altitude_m(alt_vrt_path, c_grid_lon.ravel(), c_grid_lat.ravel()).reshape(c_grid_lon.shape)

    # Project coarse points for neighbor search
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

    # Fallback where local fit failed
    c_alt_flat = c_alt.ravel()
    fallback_temp = t0_global + b_global * c_alt_flat
    pred = np.where(np.isfinite(pred), pred, fallback_temp)

    missing = ~np.isfinite(b_loc) | ~np.isfinite(a_loc)
    b_loc = np.where(missing, b_global, b_loc)
    a_loc = np.where(missing, t0_global, a_loc)

    pred_coarse = pred.reshape(c_grid_lon.shape)
    b_coarse = b_loc.reshape(c_grid_lon.shape)
    a_coarse = a_loc.reshape(c_grid_lon.shape)

    # Upsample to full grid
    zoom_factor = GRID_N / float(TEMP_COARSE_N)
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

    temp_fine = force_shape(temp_fine, grid_lon.shape)
    b_fine = force_shape(b_fine, grid_lon.shape)
    a_fine = force_shape(a_fine, grid_lon.shape)

    return temp_fine, b_fine, a_fine, b_global


def main():
    ensure_altitude_bundle()
    ensure_geojson()

    if not os.path.exists(GEOJSON_PATH):
        print(f"❌ Missing: {GEOJSON_PATH}")
        return
    if not os.path.exists(ALT_VRT_PATH):
        print(f"❌ Missing: {ALT_VRT_PATH}")
        return


    # === Fetch data ===
    try:
        text, source = robust_fetch_text(RAIN_URL, timeout=60, tries=6)
        print("source:", source)
        with open(CACHE_TXT, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print("❌ Failed to fetch data from feed.")
        return


    data = pd.read_csv(StringIO(text), delimiter="\t")

    # Convert columns
    for col in ["Latitude", "Longitude", "RainIntensity", "Datetime"]:
        if col not in data.columns:
            print(f"❌ Missing required column: {col}")
            return

    data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
    data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")
    data["RainIntensity"] = pd.to_numeric(data["RainIntensity"], errors="coerce")
    data["TNow"] = pd.to_numeric(data.get("TNow", np.nan), errors="coerce")
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")

    print(f"📥 Raw rows in feed: {len(data)}")
    print_latest_rows(data, n=8, title="(Raw feed preview)")

    # Drop invalid essentials
    before = len(data)
    data.dropna(subset=["Latitude", "Longitude", "RainIntensity", "Datetime"], inplace=True)
    print(f"🧹 dropna essentials: {before} -> {len(data)} (removed {before - len(data)})")

    before = len(data)
    data = data[(data["Latitude"] != 0) & (data["Longitude"] != 0)].copy()
    print(f"🧹 drop zero lat/lon: {before} -> {len(data)} (removed {before - len(data)})")

    bad_webcodes = ["agrivate_rizia", "metaxochori"]
    if "webcode" in data.columns:
        before = len(data)
        data = data[~data["webcode"].astype(str).str.lower().isin(bad_webcodes)].copy()
        print(f"🧹 remove bad webcodes {bad_webcodes}: {before} -> {len(data)} (removed {before - len(data)})")

    # Localize/convert datetime robustly
    dt = data["Datetime"]
    try:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT")
        else:
            dt = dt.dt.tz_convert("Europe/Athens")
    except Exception as e:
        print(f"⚠️ Datetime tz handling failed ({e}); leaving as-is (naive).")
    data["Datetime"] = dt

    # tz_localize with ambiguous/nonexistent can create NaT, so drop them now
    before = len(data)
    data = data.dropna(subset=["Datetime"]).copy()
    print(f"🧹 drop NaT after tz handling: {before} -> {len(data)} (removed {before - len(data)})")

    athens_now = datetime.now(ZoneInfo("Europe/Athens"))
    time_threshold = athens_now - timedelta(minutes=TIME_WINDOW_MIN)

    print("athens_now:", athens_now)
    print("max file datetime:", data["Datetime"].max())
    print("min file datetime:", data["Datetime"].min())
    print("time_threshold:", time_threshold)

    # Filter only recent stations
    before = len(data)
    filtered_data = data[data["Datetime"] >= time_threshold].copy()
    print(f"⏱️ time filter (last {TIME_WINDOW_MIN} min): {before} -> {len(filtered_data)} (removed {before - len(filtered_data)})")

    # Keep sane longitudes (your old guard)
    before = len(filtered_data)
    filtered_data = filtered_data[filtered_data["Longitude"] <= 30].copy()
    print(f"🧹 lon <= 30: {before} -> {len(filtered_data)} (removed {before - len(filtered_data)})")

    # Greece-only if column exists
    if "Country" in filtered_data.columns:
        before = len(filtered_data)
        c = filtered_data["Country"].astype(str).str.strip().str.upper()
        filtered_data = filtered_data[c.isin(["GR", "GREECE"])].copy()
        print(f"🇬🇷 Country filter: {before} -> {len(filtered_data)} (removed {before - len(filtered_data)})")
    else:
        print("ℹ️ 'Country' column not found; skipping country filter.")

    print_latest_rows(filtered_data, n=8, title="(After filters preview)")

    print(f"📊 Total rows after cleaning: {len(data)} | After filtering: {len(filtered_data)}")
    if filtered_data.empty:
        print("⚠️ No data points available after filters. Exiting.")
        return

    # Extract arrays
    lats = filtered_data["Latitude"].values.astype(float)
    lons = filtered_data["Longitude"].values.astype(float)
    intensities = filtered_data["RainIntensity"].values.astype(float)
    tnow = filtered_data["TNow"].values.astype(float)

    # Create plotting grid (degrees)
    grid_x, grid_y = np.meshgrid(
        np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_N),
        np.linspace(GRID_LAT_MIN, GRID_LAT_MAX, GRID_N)
    )

    # Load Greece polygon and mask (force EPSG:4326 to match grid)
    greece = gpd.read_file(GEOJSON_PATH)
    if greece.crs is None:
        greece = greece.set_crs("EPSG:4326")
    if greece.crs.to_string() != "EPSG:4326":
        greece = greece.to_crs("EPSG:4326")

    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()),
        crs="EPSG:4326"
    )
    try:
        greece_boundary = greece.geometry.union_all()
    except AttributeError:
        greece_boundary = greece.geometry.unary_union

    geo_mask = grid_points.geometry.within(greece_boundary).values.reshape(grid_x.shape)

    # Distance mask (degrees)
    station_tree = cKDTree(np.c_[lons, lats])
    distances, _ = station_tree.query(np.c_[grid_x.ravel(), grid_y.ravel()])
    distance_mask = distances.reshape(grid_x.shape) <= DISTANCE_MASK_DEG

    final_mask = geo_mask & distance_mask

    # Interpolate rain intensity (degrees)
    grid_intensity = idw_optimized(
        lons, lats, intensities,
        grid_x, grid_y,
        power=IDW_POWER,
        max_distance=MAX_DISTANCE_DEG,
        k=IDW_K
    )

    masked_intensity = np.full(grid_x.shape, np.nan, dtype=float)
    masked_intensity[final_mask] = grid_intensity[final_mask]

    # Temperature lapse: station altitude sampled via lon/lat
    station_alt = sample_altitude_m(ALT_VRT_PATH, lons, lats)

    grid_tnow_local, grid_b_local, grid_a_local, b_global = build_temperature_grid_local_lr(
        GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX,
        grid_x, grid_y,
        lons, lats,
        tnow, station_alt,
        ALT_VRT_PATH
    )

    # Snow condition on grid
    snow_grid_mask = (
        final_mask &
        np.isfinite(masked_intensity) &
        np.isfinite(grid_tnow_local) &
        (masked_intensity > RAIN_THRESH) &
        (grid_tnow_local <= SNOW_T_C)
    )

    # Isotherm height (altitude above which T <= threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        iso_alt_m = (SNOW_T_C - grid_a_local) / grid_b_local
    iso_alt_m = np.where(np.isfinite(iso_alt_m), iso_alt_m, np.nan)
    iso_alt_m = np.clip(iso_alt_m, ISO_ALT_MIN_M, ISO_ALT_MAX_M)

    # Colormap
    cmap = ListedColormap(["#deebf7", "#9ecae1", "#4292c6", "#08519c"])
    cmap.set_under("#ffffff")
    cmap.set_over("#08306b")
    cmap.set_bad("#ffffff")
    bounds = [0.1, 2, 6, 36, 100]
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

    # Output files
    timestamp = athens_now.strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join(BASE_DIR, "rainintensitymaps")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{PREFIX}{timestamp}.png")
    latest_output = os.path.join(output_dir, LATEST_NAME)

    # === PLOTTING ===
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    masked_array = ma.masked_invalid(masked_intensity)

    img = ax.imshow(
        masked_array,
        extent=(GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX),
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=0.7
    )

    greece.boundary.plot(ax=ax, color="black", linewidth=0.5)

    # Contour borders only (no labels)
    contour_levels = [0.1, 2, 6, 36, 100]
    ax.contour(grid_x, grid_y, masked_intensity, levels=contour_levels, colors="black", linewidths=1)

    # Snowflakes
    sx, sy = downsample_mask_points(
        grid_x, grid_y, snow_grid_mask,
        max_points=MAX_SNOWFLAKES,
        min_sep_deg=MIN_SEP_DEG,
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

    # Altitude labels at snowflake locations (sparse)
    if len(sx) > 0:
        x_axis = grid_x[0, :]
        y_axis = grid_y[:, 0]

        for k, (x, y) in enumerate(zip(sx, sy)):
            if LABEL_EVERY_N_SNOWFLAKES > 1 and (k % LABEL_EVERY_N_SNOWFLAKES) != 0:
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

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax, boundaries=bounds, extend="both")
    cbar.set_ticks([2, 6, 36, 100])
    cbar.set_ticklabels(["2", "6", "36", "100"])
    cbar.set_label("Ραγδαιότητα υετού (mm/h)", fontsize=12)

    ax.set_title("Υπολογ. τελευταία διαθέσιμη ραγδαιότητα υετού", fontsize=14, pad=10, loc="center")
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    # Footer
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

    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0)
    shutil.copy(output_file, latest_output)
    plt.close(fig)

    print(f"✅ Saved: {output_file}")
    print(f"✅ Saved: {latest_output}")

    # FTP upload + prune
    try:
        upload_to_ftp(output_file)
        upload_to_ftp(latest_output)
        prune_remote_pngs(keep=40, prefix=PREFIX, latest_name=LATEST_NAME)
    except Exception as e:
        print(f"⚠️ FTP upload/prune failed: {e}")


if __name__ == "__main__":
    main()
