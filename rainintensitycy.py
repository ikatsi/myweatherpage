#!/usr/bin/env python3
# rainintensity_cyprus.py
# Cyprus rain intensity map with altitude-adjusted snow mask + snowline altitude labels + FTP upload (FTPS explicit on 21)
#
# Debug additions (same logic as the Attica script):
# - robust_fetch_text() with retries, curl fallback, cached fallback
# - prints where the feed came from: network / curl / cache
# - prints Athens now, min/max datetime in the feed, and the time threshold
# - prints row counts after each filter step (dropna, time, bbox, etc.)
# - prints the latest few rows in the remote feed (after parsing) so you can see what "fresh" means

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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from pyproj import Transformer

import requests
from ftplib import FTP_TLS, error_perm

import rasterio
from rasterio.warp import transform as rio_transform
import matplotlib.patheffects as pe


# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Boundary (full island)
GEOJSON_PATH = os.path.join(BASE_DIR, "cyprus.geojson")

if not os.path.exists(GEOJSON_PATH):
    raise SystemExit(
        "❌ cyprus.geojson not found. "
        "Did you forget to decrypt cyprus.geojson.enc before running?"
    )

# DEM (Copernicus 30m DSM GeoTIFF)
VECTORS_DIR = os.path.join(BASE_DIR, "vectors")
ALT_TIF_PATH = os.path.join(VECTORS_DIR, "cy_Copernicus Global DSM 30m.tif")

# Station feed
RAIN_URL = os.environ.get("CURRENTWEATHER_URL")
if not RAIN_URL:
    raise SystemExit("❌ CURRENTWEATHER_URL environment variable is not set.")
CACHE_TXT = os.path.join(BASE_DIR, "weathernow_cyprus_cached.txt")

# Cyprus bbox in lon/lat (for station filtering only)
LON_MIN, LON_MAX = 32.0, 34.9
LAT_MIN, LAT_MAX = 34.4, 35.9

# Projections
CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32636"  # UTM zone 36N (Cyprus)

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "cyprusrainintensitymaps")
LATEST_NAME = "cyprusrainintensity_latest.png"
PREFIX = "cyprusrainintensity"

# FTP (EXPLICIT FTPS on port 21)
FTP_HOST = os.environ.get("FTP_HOST")
FTP_PORT = 21
FTP_USER = os.environ.get("FTP_USER")
FTP_PASS = os.environ.get("FTP_PASS")

if not FTP_HOST or not FTP_USER or not FTP_PASS:
    raise SystemExit("❌ FTP_HOST, FTP_USER, or FTP_PASS environment variable is not set.")

# Snow definition
SNOW_T_C = 2.0
RAIN_THRESH = 0.0

# Time filter window (minutes)
TIME_WINDOW_MIN = 30

# Grid + IDW (meters)
GRID_N = 300
IDW_POWER = 2
IDW_K = 8
MAX_DISTANCE_M = 40000.0   # IDW neighbors within 40 km
DISTANCE_MASK_M = 40000.0  # show only within 40 km of a station

# Snowflakes (meters)
SNOW_FONTSIZE = 6
SNOW_STROKE_W = 1.2
MIN_SEP_M = 1500.0
MAX_SNOWFLAKES = 5000
SNOW_SEED = 123

# Lapse rate bounds (degC per meter)
LAPSE_DEFAULT = -0.0065
LAPSE_MIN = -0.0120
LAPSE_MAX = -0.0010

# --- Snowline altitude labels (meters) ---
LABEL_EVERY_N_SNOWFLAKES = 10
LABEL_MIN_SEP_M = 12000.0
LABEL_Y_OFFSET_M = 2500.0
LABEL_FONTSIZE = 7
LABEL_STROKE_W = 1.2

ISO_ALT_ROUND_M = 50
ISO_ALT_MIN_M = 0
ISO_ALT_MAX_M = 5000


# =========================
# FETCH (same logic style as Attica)
# =========================
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

    raise RuntimeError(f"Failed to fetch {url}") from last_err


def print_latest_rows(df: pd.DataFrame, n: int = 8) -> None:
    if df.empty or "Datetime" not in df.columns:
        print("ℹ️ No rows to preview.")
        return

    cols = []
    for c in ["Datetime", "webcode", "Latitude", "Longitude", "TNow", "RainIntensity"]:
        if c in df.columns:
            cols.append(c)

    tmp = df.copy()
    tmp = tmp.dropna(subset=["Datetime"])
    tmp = tmp.sort_values("Datetime", ascending=False).head(n)

    print(f"🕒 Latest {min(n, len(tmp))} rows (sorted by Datetime desc):")
    for _, r in tmp.iterrows():
        parts = []
        for c in cols:
            v = r.get(c, "")
            if c == "Datetime":
                try:
                    parts.append(f"{c}={pd.to_datetime(v).strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception:
                    parts.append(f"{c}={v}")
            else:
                parts.append(f"{c}={v}")
        print("  - " + " | ".join(parts))


# =========================
# FTP
# =========================
def ftp_login_ftps() -> FTP_TLS:
    """
    Explicit FTPS (AUTH TLS) on port 21.
    """
    if not FTP_PASS:
        raise RuntimeError("FTP_PASS is empty. Set environment variable FTP_PASS.")

    ftps = FTP_TLS()
    ftps.connect(FTP_HOST, FTP_PORT, timeout=30)

    # Explicit TLS upgrade before login
    ftps.auth()
    ftps.login(user=FTP_USER, passwd=FTP_PASS)

    # Protect data channel
    ftps.prot_p()
    ftps.set_pasv(True)
    return ftps


def upload_to_ftp(local_file: str):
    remote_filename = os.path.basename(local_file)
    ftps = None
    try:
        ftps = ftp_login_ftps()
        with open(local_file, "rb") as f:
            ftps.storbinary("STOR " + remote_filename, f)
        print("📤 Uploaded:", remote_filename)
    except error_perm as e:
        raise RuntimeError(f"FTP login/upload failed: {e}") from e
    finally:
        if ftps is not None:
            try:
                ftps.quit()
            except Exception:
                pass


def prune_remote_cyprus_pngs(keep: int = 144):
    """
    Delete old timestamped PNGs:
      cyprusrainintensityYYYY-MM-DD-HH-MM.png
    Keep latest `keep`.
    """
    ftps = None
    try:
        ftps = ftp_login_ftps()
        names = ftps.nlst()

        pat = re.compile(r"^cyprusrainintensity\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.png$")
        timestamped = [os.path.basename(n) for n in names if pat.match(os.path.basename(n))]

        if not timestamped:
            print("ℹ️ No timestamped cyprusrainintensity PNGs found remotely.")
            return

        timestamped.sort()
        if len(timestamped) <= keep:
            print(f"ℹ️ {len(timestamped)} timestamped files ≤ keep={keep}. Nothing to delete.")
            return

        for fname in timestamped[:-keep]:
            try:
                ftps.delete(fname)
                print("🧹 Deleted old remote file:", fname)
            except Exception as e:
                print(f"⚠️ Failed to delete {fname}: {e}")
    except error_perm as e:
        raise RuntimeError(f"FTP prune failed: {e}") from e
    finally:
        if ftps is not None:
            try:
                ftps.quit()
            except Exception:
                pass


# =========================
# GEOMETRY HELPERS
# =========================
def bounds_reasonable(geom, lon_min=31.0, lon_max=36.0, lat_min=34.0, lat_max=36.5):
    try:
        minx, miny, maxx, maxy = geom.bounds
        return (lon_min <= minx <= lon_max) and (lon_min <= maxx <= lon_max) and \
               (lat_min <= miny <= lat_max) and (lat_min <= maxy <= lat_max)
    except Exception:
        return False


def swap_geom(geom):
    # swap x/y if GeoJSON came in lat/lon order
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


# =========================
# IDW
# =========================
def idw_optimized_m(x, y, z, xi, yi, power=2, max_distance=40000.0, k=8):
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

    valid = np.all(np.isfinite(dist), axis=1)

    weights = np.zeros_like(dist, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights[valid] = 1.0 / np.power(dist[valid], power)
        weights[valid] = np.where(dist[valid] == 0, 1e12, weights[valid])

    zvals = np.zeros_like(dist, dtype=float)
    zvals[valid] = z[idx[valid]]

    zi = np.full(xi.size, np.nan, dtype=float)
    num = np.sum(weights[valid] * zvals[valid], axis=1)
    den = np.sum(weights[valid], axis=1)
    zi[valid] = num / den

    return zi.reshape(xi.shape)


# =========================
# ALTITUDE / LAPSE / SNOW
# =========================
def sample_altitude_m(raster_path, xs, ys, input_crs):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise RuntimeError("Altitude GeoTIFF has no CRS defined.")

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


def fit_lapse_rate(tnow_c, alt_m):
    mask = np.isfinite(tnow_c) & np.isfinite(alt_m)
    if mask.sum() < 8:
        return LAPSE_DEFAULT
    x = alt_m[mask]
    y = tnow_c[mask]
    b, _a = np.polyfit(x, y, 1)  # y = b*x + a
    if not (LAPSE_MIN <= b <= LAPSE_MAX):
        return LAPSE_DEFAULT
    return float(b)


def downsample_mask_points_utm(grid_E, grid_N, mask, max_points=5000, min_sep_m=1500.0, seed=123):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([]), np.array([])

    pts = np.column_stack([grid_E[ys, xs], grid_N[ys, xs]])
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
            if d >= min_sep_m:
                kept.append(p)
                if len(kept) % 200 == 0:
                    kept_tree = cKDTree(np.array(kept))

        if len(kept) >= max_points:
            break

    kept = np.array(kept)
    return kept[:, 0], kept[:, 1]


def label_snowline_altitudes(
    ax,
    sx, sy,
    iso_alt_grid_m,
    grid_E, grid_N,
    every_n=10,
    min_sep_m=12000.0,
    y_offset_m=2500.0
):
    """
    Place labels "XXXXm+" near snowflake positions, sampling iso_alt_grid_m at nearest grid cell.
    Crowd-control with min distance between labels.
    """
    if len(sx) == 0:
        return

    x_axis = grid_E[0, :]
    y_axis = grid_N[:, 0]

    label_pts = []
    label_tree = None

    for k, (x, y) in enumerate(zip(sx, sy)):
        if every_n > 1 and (k % every_n) != 0:
            continue

        if min_sep_m and label_tree is not None and len(label_pts) > 0:
            d, _ = label_tree.query([x, y], k=1)
            if d < min_sep_m:
                continue

        j = int(np.argmin(np.abs(x_axis - x)))
        i = int(np.argmin(np.abs(y_axis - y)))

        z = iso_alt_grid_m[i, j]
        if not np.isfinite(z):
            continue

        z_round = int(round(z / ISO_ALT_ROUND_M) * ISO_ALT_ROUND_M)
        z_round = max(ISO_ALT_MIN_M, min(ISO_ALT_MAX_M, z_round))

        label = f"{z_round}m+"

        ax.text(
            x, y - y_offset_m,
            label,
            ha="center", va="top",
            fontsize=LABEL_FONTSIZE,
            color="black",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=LABEL_STROKE_W, foreground="white")]
        )

        label_pts.append([x, y])
        if len(label_pts) == 1 or (len(label_pts) % 50 == 0):
            label_tree = cKDTree(np.array(label_pts))


# =========================
# MAIN
# =========================
def main():
    # --- load boundary ---
    try:
        cyprus = gpd.read_file(GEOJSON_PATH)
        print("🔎 Loaded boundary from:", os.path.basename(GEOJSON_PATH))
    except Exception as e:
        raise SystemExit(f"❌ Failed to read cyprus.geojson: {e}")

    if cyprus.crs is None:
        cyprus = cyprus.set_crs(epsg=4326)

    cyprus = cyprus[~cyprus.geometry.is_empty]
    if not cyprus.geometry.is_valid.all():
        cyprus.geometry = cyprus.buffer(0)

    try:
        cyprus_boundary_ll = cyprus.union_all()
    except AttributeError:
        cyprus_boundary_ll = unary_union(cyprus.geometry)

    if not bounds_reasonable(cyprus_boundary_ll):
        cyprus.geometry = cyprus.geometry.apply(swap_geom)
        try:
            cyprus_boundary_ll = cyprus.union_all()
        except AttributeError:
            cyprus_boundary_ll = unary_union(cyprus.geometry)

    print("CRS:", cyprus.crs)
    print("Bounds (lon/lat):", cyprus.total_bounds)

    # project boundary to UTM
    cyprus_utm = cyprus.to_crs(CRS_UTM)
    try:
        cyprus_boundary_utm = cyprus_utm.union_all()
    except AttributeError:
        cyprus_boundary_utm = unary_union(cyprus_utm.geometry)

    # --- altitude file ---
    if not os.path.exists(ALT_TIF_PATH):
        raise SystemExit(f"❌ Missing altitude GeoTIFF: {ALT_TIF_PATH}")

    # --- fetch data (robust + cached) ---
    try:
        text, source = robust_fetch_text(RAIN_URL, timeout=60, tries=6)
        print("source:", source)
        with open(CACHE_TXT, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        raise SystemExit(f"❌ Failed to fetch {RAIN_URL}: {e}")

    data = pd.read_csv(StringIO(text), delimiter="\t")

    # --- required columns ---
    needed = ["Latitude", "Longitude", "RainIntensity", "Datetime"]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise SystemExit("❌ Missing columns in feed: " + ", ".join(missing))

    # --- conversions ---
    data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
    data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")
    data["RainIntensity"] = pd.to_numeric(data["RainIntensity"], errors="coerce")
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")

    if "TNow" in data.columns:
        data["TNow"] = pd.to_numeric(data["TNow"], errors="coerce")
    else:
        data["TNow"] = np.nan

    print(f"📥 Raw rows in feed: {len(data)}")
    print_latest_rows(data, n=8)

    # --- drop invalid essentials ---
    before = len(data)
    data.dropna(subset=["Latitude", "Longitude", "RainIntensity", "Datetime"], inplace=True)
    after = len(data)
    print(f"🧹 dropna essentials: {before} -> {after} (removed {before - after})")

    before = len(data)
    data = data[(data["Latitude"] != 0) & (data["Longitude"] != 0)].copy()
    after = len(data)
    print(f"🧹 drop zero lat/lon: {before} -> {after} (removed {before - after})")

    # optional: remove known bad station(s)
    if "webcode" in data.columns:
        bad = ["kardamyla"]
        before = len(data)
        data = data[~data["webcode"].astype(str).str.lower().isin(bad)].copy()
        after = len(data)
        print(f"🧹 remove bad webcodes {bad}: {before} -> {after} (removed {before - after})")

    # --- time filter (feed is Europe/Athens naive) ---
    athens_now_aware = datetime.now(ZoneInfo("Europe/Athens"))
    athens_now_naive = athens_now_aware.replace(tzinfo=None)
    time_threshold = athens_now_naive - timedelta(minutes=TIME_WINDOW_MIN)

    print("athens_now:", athens_now_aware)
    print("max file datetime:", data["Datetime"].max())
    print("min file datetime:", data["Datetime"].min())
    print("time_threshold:", time_threshold)

    before = len(data)
    filtered = data[data["Datetime"] >= time_threshold].copy()
    after = len(filtered)
    print(f"⏱️ time filter (last {TIME_WINDOW_MIN} min): {before} -> {after} (removed {before - after})")

    print_latest_rows(filtered, n=8)

    # --- bbox filter ---
    before = len(filtered)
    filtered = filtered[
        filtered["Longitude"].between(LON_MIN, LON_MAX) &
        filtered["Latitude"].between(LAT_MIN, LAT_MAX)
    ].copy()
    after = len(filtered)
    print(f"🗺️ bbox Cyprus lon[{LON_MIN},{LON_MAX}] lat[{LAT_MIN},{LAT_MAX}]: {before} -> {after} (removed {before - after})")

    if filtered.empty:
        print("⚠️ No data points after filters. Exiting.")
        return

    # Extra: unique stations count if webcode exists
    if "webcode" in filtered.columns:
        uniq = filtered["webcode"].astype(str).nunique()
        print(f"📍 Unique webcodes in filtered set: {uniq}")

    # --- station coords to UTM ---
    to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)
    lons = filtered["Longitude"].values
    lats = filtered["Latitude"].values
    easts, norths = to_utm.transform(lons, lats)

    intensities = filtered["RainIntensity"].values.astype(float)
    tnow = filtered["TNow"].values.astype(float)

    # --- grid over bbox corners ---
    corn_lon = np.array([LON_MIN, LON_MAX, LON_MIN, LON_MAX])
    corn_lat = np.array([LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX])
    corn_E, corn_N = to_utm.transform(corn_lon, corn_lat)

    E_MIN, E_MAX = float(np.min(corn_E)), float(np.max(corn_E))
    N_MIN, N_MAX = float(np.min(corn_N)), float(np.max(corn_N))

    grid_E, grid_N = np.meshgrid(
        np.linspace(E_MIN, E_MAX, GRID_N),
        np.linspace(N_MIN, N_MAX, GRID_N)
    )

    # --- interpolate intensity ---
    grid_intensity = idw_optimized_m(
        easts, norths, intensities, grid_E, grid_N,
        power=IDW_POWER, max_distance=MAX_DISTANCE_M, k=IDW_K
    )

    # --- mask: inside boundary + within distance ---
    grid_points_utm = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_E.ravel(), grid_N.ravel()),
        crs=CRS_UTM
    )
    geo_mask = grid_points_utm.geometry.within(cyprus_boundary_utm).values.reshape(grid_E.shape)

    station_tree = cKDTree(np.c_[easts, norths])
    dists, _ = station_tree.query(np.c_[grid_E.ravel(), grid_N.ravel()])
    distance_mask = dists.reshape(grid_E.shape) <= DISTANCE_MASK_M

    final_mask = geo_mask & distance_mask

    masked_intensity = np.full(grid_E.shape, np.nan, dtype=float)
    masked_intensity[final_mask] = grid_intensity[final_mask]

    # --- altitude + lapse + snow mask ---
    station_alt = sample_altitude_m(ALT_TIF_PATH, lons, lats, input_crs=CRS_WGS84)
    lapse = fit_lapse_rate(tnow, station_alt)

    ok_t = np.isfinite(tnow) & np.isfinite(station_alt)
    t0 = np.full_like(tnow, np.nan, dtype=float)
    t0[ok_t] = tnow[ok_t] - lapse * station_alt[ok_t]

    grid_t0 = idw_optimized_m(
        easts, norths, t0, grid_E, grid_N,
        power=IDW_POWER, max_distance=MAX_DISTANCE_M, k=IDW_K
    )

    grid_alt = sample_altitude_m(
        ALT_TIF_PATH, grid_E.ravel(), grid_N.ravel(), input_crs=CRS_UTM
    ).reshape(grid_E.shape)

    grid_tnow_adj = np.full(grid_E.shape, np.nan, dtype=float)
    ok_grid = np.isfinite(grid_t0) & np.isfinite(grid_alt)
    grid_tnow_adj[ok_grid] = grid_t0[ok_grid] + lapse * grid_alt[ok_grid]

    snow_grid_mask = (
        final_mask &
        np.isfinite(masked_intensity) &
        np.isfinite(grid_tnow_adj) &
        (masked_intensity > RAIN_THRESH) &
        (grid_tnow_adj <= SNOW_T_C)
    )

    # --- compute snowline altitude grid (altitude above which T <= SNOW_T_C) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        iso_alt_m = (SNOW_T_C - grid_t0) / lapse
    iso_alt_m = np.where(final_mask & np.isfinite(iso_alt_m), iso_alt_m, np.nan)
    iso_alt_m = np.clip(iso_alt_m, ISO_ALT_MIN_M, ISO_ALT_MAX_M)

    # --- colormap ---
    cmap = ListedColormap(["#ffffff", "#deebf7", "#9ecae1", "#4292c6", "#08519c"])
    bounds = [0.0, 0.2, 2, 6, 40, 100]
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

    # --- output paths ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = athens_now_aware.strftime("%Y-%m-%d-%H-%M")
    out_png = os.path.join(OUTPUT_DIR, f"{PREFIX}{timestamp}.png")
    out_latest = os.path.join(OUTPUT_DIR, LATEST_NAME)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # degree axes while plotting in meters
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

    masked_array = ma.masked_invalid(masked_intensity)

    img = ax.imshow(
        masked_array,
        extent=(E_MIN, E_MAX, N_MIN, N_MAX),
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=0.7
    )

    plot_boundary_proj(ax, cyprus_boundary_utm, linewidth=0.5, color="black")
    ax.set_aspect("equal")

    # contour borders only (NO labels)
    contour_levels = [0.2, 2, 6, 40]
    ax.contour(grid_E, grid_N, masked_intensity, levels=contour_levels, colors="black", linewidths=1)

    # snowflakes
    sx, sy = downsample_mask_points_utm(
        grid_E, grid_N, snow_grid_mask,
        max_points=MAX_SNOWFLAKES, min_sep_m=MIN_SEP_M, seed=SNOW_SEED
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

    # snowline altitude labels near snowflakes
    label_snowline_altitudes(
        ax,
        sx, sy,
        iso_alt_m,
        grid_E, grid_N,
        every_n=LABEL_EVERY_N_SNOWFLAKES,
        min_sep_m=LABEL_MIN_SEP_M,
        y_offset_m=LABEL_Y_OFFSET_M
    )

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label("Ραγδαιότητα υετού (mm/h)", fontsize=12)

    ax.set_title("Υπολογ. τελευταία διαθέσιμη ραγδαιότητα υετού", fontsize=14, pad=10, loc="center")
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    timestamp_text = athens_now_aware.strftime("Δημιουργήθηκε: %Y-%m-%d %H:%M %Z για το e-kairos.gr")
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

    # FTP upload + prune
    upload_to_ftp(out_png)
    upload_to_ftp(out_latest)
    prune_remote_cyprus_pngs(keep=144)


if __name__ == "__main__":
    main()
