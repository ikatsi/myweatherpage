#!/usr/bin/env python3
# rainintensityattiki.py
#
# Same map, but EVERYTHING is handled/plotted in Greek Grid (EGSA87) EPSG:2100:
# - Interpolation grid is in meters (EPSG:2100)
# - IDW distances are in meters
# - Snowflake thinning/min-separation is in meters
# - Greece boundary is reprojected and clipped in EPSG:2100
# - Plot geometry stays EPSG:2100, but AXES are LABELLED as lon/lat degrees (tick formatter)
#
# Output:
#   ./rainintensitymaps/rain_intensity_attica_YYYY-MM-DD-HH-MM.png
#   ./rainintensitymaps/latestattica.png
# Uploads both to FTP (explicit FTPS on port 21) and prunes old timestamped PNGs.

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
from matplotlib.ticker import FuncFormatter, MaxNLocator

from scipy.spatial import cKDTree
from scipy.ndimage import zoom
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe

import rasterio
from rasterio.warp import transform as rio_transform
from pyproj import Transformer
from ftplib import FTP_TLS

from shapely.geometry import box

import requests


# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
ALT_VRT_PATH = os.path.join(BASE_DIR, "GRC_alt.vrt")

RAIN_URL = os.environ.get("CURRENTWEATHER_URL")
if not RAIN_URL:
    raise SystemExit("❌ CURRENTWEATHER_URL not set")

CACHE_TXT = os.path.join(BASE_DIR, "weathernow_cached.txt")

FTP_HOST = os.environ.get("FTP_HOST")
FTP_USER = os.environ.get("FTP_USER")
FTP_PASS = os.environ.get("FTP_PASS")
if not FTP_HOST or not FTP_USER or not FTP_PASS:
    raise SystemExit("❌ FTP_HOST / FTP_USER / FTP_PASS not set")

PREFIX = "rain_intensity_attica_"
LATEST_NAME = "latestattica.png"

CRS_WGS84 = "EPSG:4326"
CRS_EGSA87 = "EPSG:2100"

# Area of interest in lon/lat (degrees); converted internally to EGSA meters
GRID_N = 300
GRID_LON_MIN, GRID_LON_MAX = 22.7, 25.0
GRID_LAT_MIN, GRID_LAT_MAX = 37.5, 38.7

# IDW settings (meters)
IDW_POWER = 2
IDW_K = 8
MAX_DISTANCE_M = 120_000
DISTANCE_MASK_M = 170_000

# Snow thresholds
SNOW_T_C = 2.0
RAIN_THRESH = 0.0

# Snowflake display (meters)
SNOW_FONTSIZE = 6
SNOW_STROKE_W = 1.2
MIN_SEP_M = 4_500
MAX_SNOWFLAKES = 5000
SNOW_SEED = 123

# Altitude label density (meters)
LABEL_EVERY_N_SNOWFLAKES = 10
LABEL_MIN_SEP_M = 12_000
LABEL_Y_OFFSET_M = 3_000
LABEL_FONTSIZE = 6

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


# =========================
# FTP HELPERS
# =========================
def upload_to_ftp(local_file: str) -> None:
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
      - Keep latestattica.png
      - Keep newest `keep` files matching: PREFIX + YYYY-MM-DD-HH-MM.png
      - Delete older ones
    Only touches files starting with `prefix`.
    """
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
            if not fname.startswith(prefix):
                continue
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


# =========================
# FETCH HELPERS
# =========================
def robust_fetch_text(url: str, timeout: int = 60, tries: int = 6):
    """
    Returns: (text, source)
      source is one of: "network", "curl", "cache"
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


# =========================
# ALTITUDE SAMPLING
# =========================
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


# =========================
# IDW (meters)
# =========================
def idw_optimized(x, y, z, xi, yi, power=2, max_distance=120_000, k=8) -> np.ndarray:
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
# Minimum-distance thinning for snowflakes (meters)
# =========================
def downsample_mask_points(grid_x, grid_y, mask, max_points=5000, min_sep=4500, seed=123):
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


# =========================
# Local regression temperature field
# =========================
def build_temperature_grid_local_lr(
    grid_x_m, grid_y_m,
    station_x_m, station_y_m,
    station_tnow, station_alt_m,
    transformer_egsa_to_wgs,
    alt_vrt_path,
    x_min, x_max, y_min, y_max
):
    st_x = np.asarray(station_x_m, dtype=float)
    st_y = np.asarray(station_y_m, dtype=float)
    st_t = np.asarray(station_tnow, dtype=float)
    st_alt = np.asarray(station_alt_m, dtype=float)

    ok = np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_t) & np.isfinite(st_alt)
    st_x, st_y, st_t, st_alt = st_x[ok], st_y[ok], st_t[ok], st_alt[ok]

    if len(st_t) < 10:
        temp_nan = np.full_like(grid_x_m, np.nan, dtype=float)
        b_nan = np.full_like(grid_x_m, np.nan, dtype=float)
        a_nan = np.full_like(grid_x_m, np.nan, dtype=float)
        return temp_nan, b_nan, a_nan, LAPSE_DEFAULT

    b_global = fit_global_lapse_rate(st_t, st_alt)
    t0_global = np.nanmedian(st_t - b_global * st_alt)

    # Coarse EGSA grid
    c_x = np.linspace(x_min, x_max, TEMP_COARSE_N)
    c_y = np.linspace(y_min, y_max, TEMP_COARSE_N)
    c_grid_x, c_grid_y = np.meshgrid(c_x, c_y)

    # Altitude at coarse points: EGSA -> lon/lat -> sample
    c_lon, c_lat = transformer_egsa_to_wgs.transform(c_grid_x.ravel().tolist(), c_grid_y.ravel().tolist())
    c_alt = sample_altitude_m(alt_vrt_path, np.array(c_lon), np.array(c_lat)).reshape(c_grid_x.shape)

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

    # Fallback where local fit failed
    c_alt_flat = c_alt.ravel()
    fallback_temp = t0_global + b_global * c_alt_flat
    pred = np.where(np.isfinite(pred), pred, fallback_temp)

    missing = ~np.isfinite(b_loc) | ~np.isfinite(a_loc)
    b_loc = np.where(missing, b_global, b_loc)
    a_loc = np.where(missing, t0_global, a_loc)

    pred_coarse = pred.reshape(c_grid_x.shape)
    b_coarse = b_loc.reshape(c_grid_x.shape)
    a_coarse = a_loc.reshape(c_grid_x.shape)

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

    temp_fine = force_shape(temp_fine, grid_x_m.shape)
    b_fine = force_shape(b_fine, grid_x_m.shape)
    a_fine = force_shape(a_fine, grid_x_m.shape)

    return temp_fine, b_fine, a_fine, b_global


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(GEOJSON_PATH):
        print(f"❌ Missing: {GEOJSON_PATH}")
        return
    if not os.path.exists(ALT_VRT_PATH):
        print(f"❌ Missing: {ALT_VRT_PATH}")
        return

    wgs_to_egsa = Transformer.from_crs(CRS_WGS84, CRS_EGSA87, always_xy=True)
    egsa_to_wgs = Transformer.from_crs(CRS_EGSA87, CRS_WGS84, always_xy=True)

    # Convert lon/lat bbox to EGSA bbox
    corners_lon = [GRID_LON_MIN, GRID_LON_MIN, GRID_LON_MAX, GRID_LON_MAX]
    corners_lat = [GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_MIN, GRID_LAT_MAX]
    corners_x, corners_y = wgs_to_egsa.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(corners_x)), float(np.max(corners_x))
    y_min, y_max = float(np.min(corners_y)), float(np.max(corners_y))

    # Fetch data
    try:
        text, source = robust_fetch_text(RAIN_URL, timeout=60, tries=6)
        print("source:", source)
        with open(CACHE_TXT, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return

    data = pd.read_csv(StringIO(text), delimiter="\t")

    # Convert columns
    data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
    data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")
    data["RainIntensity"] = pd.to_numeric(data["RainIntensity"], errors="coerce")
    data["TNow"] = pd.to_numeric(data.get("TNow", np.nan), errors="coerce")

    # Parse datetimes as naive first, then localize to Athens
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data["Datetime"] = data["Datetime"].dt.tz_localize(
        "Europe/Athens", ambiguous="NaT", nonexistent="NaT"
    )

    # Drop invalid entries
    data.dropna(subset=["Latitude", "Longitude", "RainIntensity", "Datetime"], inplace=True)
    data = data[(data["Latitude"] != 0) & (data["Longitude"] != 0)]

    bad_webcodes = ["agrivate_rizia", "metaxochori"]
    if "webcode" in data.columns:
        data = data[~data["webcode"].astype(str).str.lower().isin(bad_webcodes)]

    # Filter only recent stations (Athens time)
    athens_now = datetime.now(ZoneInfo("Europe/Athens"))
    time_threshold = athens_now - timedelta(minutes=45)

    print("athens_now:", athens_now)
    print("max file datetime:", data["Datetime"].max())
    print("min file datetime:", data["Datetime"].min())
    print("time_threshold:", time_threshold)

    filtered_data = data[data["Datetime"] >= time_threshold].copy()
    filtered_data = filtered_data[filtered_data["Longitude"] <= 30].copy()

    # Greece only if column exists
    if "Country" in filtered_data.columns:
        c = filtered_data["Country"].astype(str).str.strip().str.upper()
        filtered_data = filtered_data[c.isin(["GR", "GREECE"])].copy()

    print(f"📊 Total rows: {len(data)} | After filtering: {len(filtered_data)}")
    if filtered_data.empty:
        print("⚠️ No data points available within the last 45 minutes. Exiting.")
        return

    # Station arrays (lon/lat degrees)
    lats = filtered_data["Latitude"].values.astype(float)
    lons = filtered_data["Longitude"].values.astype(float)
    intensities = filtered_data["RainIntensity"].values.astype(float)
    tnow = filtered_data["TNow"].values.astype(float)

    # Project stations to EGSA (meters)
    st_x, st_y = wgs_to_egsa.transform(lons.tolist(), lats.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    # Grid in EGSA (meters)
    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, GRID_N),
        np.linspace(y_min, y_max, GRID_N)
    )

    # Greece outline in EGSA, clipped to bbox
    greece = gpd.read_file(GEOJSON_PATH)
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

    # Mask grid points inside Greece geometry (EGSA)
    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_EGSA87
    )
    try:
        greece_boundary = greece_clip.geometry.union_all()
    except AttributeError:
        greece_boundary = greece_clip.geometry.unary_union

    geo_mask = grid_points.geometry.within(greece_boundary).values.reshape(grid_x_m.shape)

    # Distance mask (meters)
    station_tree = cKDTree(np.c_[st_x, st_y])
    distances, _ = station_tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    distance_mask = distances.reshape(grid_x_m.shape) <= DISTANCE_MASK_M

    final_mask = geo_mask & distance_mask

    # Interpolate rain intensity (EGSA)
    grid_intensity = idw_optimized(
        st_x, st_y, intensities,
        grid_x_m, grid_y_m,
        power=IDW_POWER,
        max_distance=MAX_DISTANCE_M,
        k=IDW_K
    )

    masked_intensity = np.full(grid_x_m.shape, np.nan, dtype=float)
    masked_intensity[final_mask] = grid_intensity[final_mask]

    # Temperature lapse: station altitude sampled via lon/lat
    station_alt = sample_altitude_m(ALT_VRT_PATH, lons, lats)

    grid_tnow_local, grid_b_local, grid_a_local, b_global = build_temperature_grid_local_lr(
        grid_x_m, grid_y_m,
        st_x, st_y,
        tnow, station_alt,
        egsa_to_wgs,
        ALT_VRT_PATH,
        x_min, x_max, y_min, y_max
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

    # Plot (geometry in EPSG:2100, ticks shown as lon/lat degrees)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    masked_array = ma.masked_invalid(masked_intensity)

    img = ax.imshow(
        masked_array,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=0.7
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    # Show lon/lat degrees on axes (labels only; map stays EGSA)
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

    # Contour borders only (no labels)
    contour_levels = [0.1, 2, 6, 36, 100]
    ax.contour(grid_x_m, grid_y_m, masked_intensity, levels=contour_levels, colors="black", linewidths=1)

    # Snowflake overlay (EGSA coords)
    sx, sy = downsample_mask_points(
        grid_x_m, grid_y_m, snow_grid_mask,
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
            path_effects=[pe.withStroke(linewidth=SNOW_STROKE_W, foreground="white")]
        )

    if len(sx) > 0:
        print(f"❄ Snow flags placed at {len(sx)} point(s) (min_sep={MIN_SEP_M} m).")
    else:
        print("❄ No snow grid points found with current thresholds.")

    # Altitude labels (denser but controlled)
    if len(sx) > 0:
        x_axis = grid_x_m[0, :]
        y_axis = grid_y_m[:, 0]

        label_pts = []
        label_tree = None

        for k, (x, y) in enumerate(zip(sx, sy)):
            if LABEL_EVERY_N_SNOWFLAKES > 1 and (k % LABEL_EVERY_N_SNOWFLAKES) != 0:
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
            label = f"{z_round}m+"

            ax.text(
                x, y - LABEL_Y_OFFSET_M,
                label,
                ha="center", va="top",
                fontsize=LABEL_FONTSIZE,
                color="black",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")]
            )

            label_pts.append([x, y])
            if len(label_pts) == 1 or (len(label_pts) % 50 == 0):
                label_tree = cKDTree(np.array(label_pts))

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax, boundaries=bounds, extend="both")
    cbar.set_ticks([2, 6, 36, 100])
    cbar.set_ticklabels(["2", "6", "36", "100"])
    cbar.set_label("Ραγδαιότητα υετού (mm/h)", fontsize=12)

    # Titles and footer
    ax.set_title("Υπολογ. τελευταία διαθέσιμη ραγδαιότητα υετού", fontsize=14, pad=10, loc="center")
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    timestamp_text = athens_now.strftime("%Y-%m-%d %H:%M %Z")
    left_text = f"Δημιουργήθηκε για το e-kairos.gr\n{timestamp_text}"
    right_text = f"v4.2-egsa2100\nΧιονόπτωση ≤ {SNOW_T_C:.1f}°C\nΣυνολ. βαροβαθμιδα: {b_global*1000:.2f} °C/km"

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
