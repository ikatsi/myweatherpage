#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py

Independent map generator for:
  - Cloudiness
  - Rain

Regions:
  - Greece (WGS84)
  - Attica (EGSA87 / EPSG:2100)
  - Crete (EGSA87 / EPSG:2100)
  - NE Greece (EGSA87 / EPSG:2100)
  - SW Greece (EGSA87 / EPSG:2100)
  - Cyprus (UTM 36N / EPSG:32636)

Key design goals:
  - Same Greece/EGSA/Cyprus plotting logic as rainintensityall.py
  - Same regional bboxes as rainintensityall.py
  - Same right-side reserved legend space as rainintensityall.py
  - Timestamped PNG + static latest PNG
  - FTP upload + remote prune
  - Manual GitHub Action friendly
"""

import os
import re
import io
import sys
import math
import time
import shutil
import socket
import zipfile
import random
import argparse
import subprocess
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable

import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

from pyproj import Transformer

from ftplib import FTP_TLS


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATHENS_TZ = ZoneInfo("Europe/Athens")

# Optional FTP
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()

# Greece encrypted assets
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()
GREECE_GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
GREECE_GEOJSON_ENC  = os.path.join(BASE_DIR, "greece.geojson.enc")

# Optional Cyprus asset
CYPRUS_GEOJSON_PATH = os.path.join(BASE_DIR, "cyprus.geojson")

# Cloud source
EUMETVIEW_WMS = "https://view.eumetsat.int/geoserver/wms"
CLOUD_WMS_LAYER = os.environ.get("CLOUD_WMS_LAYER", "msg_fes:ir108").strip() or "msg_fes:ir108"

# Rain source
RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"
RAIN_ZOOM = int(os.environ.get("RAIN_ZOOM", "7").strip() or "7")
if RAIN_ZOOM < 0:
    RAIN_ZOOM = 0
if RAIN_ZOOM > 7:
    RAIN_ZOOM = 7

# Rendering controls
FIGSIZE = (10, 10)
DPI = 300

# source image sizes
SOURCE_WMS_W = 768
SOURCE_WMS_H = 768
TARGET_GRID_N_GREECE = 700
TARGET_GRID_N_REGION = 700
TARGET_GRID_N_CYPRUS = 700

# RainViewer rendering options
RAIN_TILE_SIZE = 256
RAIN_COLOR_SCHEME = 2      # Universal Blue
RAIN_OPTIONS = "1_1"       # smoothed + snow colors


# =============================================================================
# REGION CONFIGS
# =============================================================================
# IMPORTANT:
# bboxes below follow the EXACT SAME ORDER as rainintensityall.py:
# (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "greece": {
        "name": "Greece",
        "bbox": (19.0, 30.0, 34.5, 42.5),
        "mode": "greece_wgs84",
        "boundary": "greece",
        "cloud_prefix": "cloudiness_",
        "cloud_latest": "latestcloudiness.png",
        "rain_prefix": "rain_rate_",
        "rain_latest": "latestrainrate.png",
        "remote_keep": 200,
    },
    "attica": {
        "name": "Attica",
        "bbox": (22.7, 25.0, 37.5, 38.7),
        "mode": "egsa_region",
        "boundary": "greece",
        "cloud_prefix": "cloudiness_attica_",
        "cloud_latest": "latestcloudinessattica.png",
        "rain_prefix": "rain_rate_attica_",
        "rain_latest": "latestrainrateattica.png",
        "remote_keep": 200,
    },
    "crete": {
        "name": "Crete",
        "bbox": (23.37, 26.4, 34.7, 35.78),
        "mode": "egsa_region",
        "boundary": "greece",
        "cloud_prefix": "cloudiness_crete_",
        "cloud_latest": "latestcloudinesscrete.png",
        "rain_prefix": "rain_rate_crete_",
        "rain_latest": "latestrainratecrete.png",
        "remote_keep": 200,
    },
    "negreece": {
        "name": "NE Greece",
        "bbox": (22.0, 26.6, 39.7, 41.8),
        "mode": "egsa_region",
        "boundary": "greece",
        "cloud_prefix": "cloudiness_negreece_",
        "cloud_latest": "latestcloudinessnegreece.png",
        "rain_prefix": "rain_rate_negreece_",
        "rain_latest": "latestrainratenegreece.png",
        "remote_keep": 200,
    },
    "swgreece": {
        "name": "SW Greece",
        "bbox": (20.0, 24.0, 36.0, 39.0),
        "mode": "egsa_region",
        "boundary": "greece",
        "cloud_prefix": "cloudiness_swgreece_",
        "cloud_latest": "latestcloudinessswgreece.png",
        "rain_prefix": "rain_rate_swgreece_",
        "rain_latest": "latestrainrateswgreece.png",
        "remote_keep": 200,
    },
    "cyprus": {
        "name": "Cyprus",
        "bbox": (32.0, 34.9, 34.4, 35.9),
        "mode": "cyprus_utm",
        "boundary": "cyprus",
        "cloud_prefix": "cloudiness_cyprus_",
        "cloud_latest": "latestcloudinesscyprus.png",
        "rain_prefix": "rain_rate_cyprus_",
        "rain_latest": "latestrainratecyprus.png",
        "remote_keep": 200,
    },
}


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

def reserve_right_legend_space(fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    bg = fig.get_facecolor()
    blank_cmap = ListedColormap([bg, bg])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=blank_cmap)
    sm.set_array([0.0, 1.0])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)

    cax.set_facecolor(bg)
    cax.set_xticks([])
    cax.set_yticks([])
    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.set_frame_on(False)

    return cax

def athens_abbrev(dt: datetime) -> str:
    try:
        dt_ath = dt.astimezone(ATHENS_TZ)
        is_dst = bool(dt_ath.dst()) and dt_ath.dst() != timedelta(0)
        return "EEST" if is_dst else "EET"
    except Exception:
        return "EET"


def fmt_generated(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def ensure_greece_geojson():
    if os.path.exists(GREECE_GEOJSON_PATH):
        return

    if not os.path.exists(GREECE_GEOJSON_ENC):
        raise SystemExit("❌ Missing greece.geojson and greece.geojson.enc")

    if not GEOJSON_PASS:
        raise SystemExit("❌ GEOJSON_PASS not set")

    try:
        subprocess.check_call([
            "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
            "-in", GREECE_GEOJSON_ENC,
            "-out", GREECE_GEOJSON_PATH,
            "-pass", "pass:" + GEOJSON_PASS
        ])
    except FileNotFoundError:
        raise SystemExit("❌ OpenSSL not found on runner")
    except subprocess.CalledProcessError as e:
        raise SystemExit("❌ Failed to decrypt greece.geojson.enc: %s" % e)


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
            print("⚠️ FTPS connect failed (%s: %s). Retry in %ss..." % (type(e).__name__, e, sleep_s))
            time.sleep(sleep_s)
    raise last_err


def ftp_upload_many_and_prune(upload_files, prune_specs):
    if not ftp_enabled():
        print("ℹ️ FTP disabled (missing env). Skipping upload/prune.")
        return

    ftps = ftps_connect_with_retries(FTP_HOST, FTP_USER, FTP_PASS, attempts=6, base_sleep=5, timeout=60)
    try:
        for local_file in upload_files:
            remote_filename = os.path.basename(local_file)
            with open(local_file, "rb") as f:
                ftps.storbinary("STOR " + remote_filename, f)
            print("📤 Uploaded:", remote_filename)

        # list once
        try:
            names = ftps.nlst()
        except Exception as e:
            print("⚠️ Could not list remote directory for prune:", e)
            names = []

        basenames = [os.path.basename(n) for n in names if n]

        for prefix, latest_name, keep in prune_specs:
            pat = re.compile(r"^" + re.escape(prefix) + r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.png$")
            timestamped = [n for n in basenames if pat.match(n) and n != latest_name]
            timestamped.sort()

            if len(timestamped) <= keep:
                print("ℹ️ %s timestamped files <= keep=%s. Nothing to delete." % (prefix, keep))
                continue

            to_delete = timestamped[:-keep]
            for fname in to_delete:
                try:
                    ftps.delete(fname)
                    print("🧹 Deleted old remote file:", fname)
                except Exception as e:
                    print("⚠️ Failed to delete %s: %s" % (fname, e))

    finally:
        try:
            ftps.quit()
        except Exception:
            pass


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def load_greece_wgs84():
    ensure_greece_geojson()
    greece = gpd.read_file(GREECE_GEOJSON_PATH)
    if greece.crs is None:
        greece = greece.set_crs("EPSG:4326")
    if greece.crs.to_string() != "EPSG:4326":
        greece = greece.to_crs("EPSG:4326")
    return greece


def load_greece_egsa87():
    greece = load_greece_wgs84()
    return greece.to_crs("EPSG:2100")


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


def load_cyprus_wgs84_or_none():
    if not os.path.exists(CYPRUS_GEOJSON_PATH):
        return None

    cyprus = gpd.read_file(CYPRUS_GEOJSON_PATH)
    if cyprus.crs is None:
        cyprus = cyprus.set_crs("EPSG:4326")
    cyprus = cyprus[~cyprus.geometry.is_empty]
    if not cyprus.geometry.is_valid.all():
        cyprus.geometry = cyprus.buffer(0)

    try:
        geom = cyprus.geometry.union_all()
    except AttributeError:
        geom = unary_union(cyprus.geometry)

    if not bounds_reasonable(geom):
        cyprus.geometry = cyprus.geometry.apply(swap_geom)

    if cyprus.crs.to_string() != "EPSG:4326":
        cyprus = cyprus.to_crs("EPSG:4326")

    return cyprus


def plot_boundary_proj(ax, geom, linewidth=0.5, color="black"):
    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.plot(x, y, linewidth=linewidth, color=color, zorder=3)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=linewidth, color=color, zorder=3)


# =============================================================================
# SOURCE FETCHERS
# =============================================================================
def fetch_cloud_wms_rgba(lon_min, lon_max, lat_min, lat_max, width=SOURCE_WMS_W, height=SOURCE_WMS_H):
    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": CLOUD_WMS_LAYER,
        "styles": "",
        "srs": "EPSG:4326",
        "bbox": "%.6f,%.6f,%.6f,%.6f" % (lon_min, lat_min, lon_max, lat_max),
        "width": str(width),
        "height": str(height),
        "format": "image/png",
        "transparent": "true",
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/png,*/*;q=0.8",
    }

    r = requests.get(EUMETVIEW_WMS, params=params, headers=headers, timeout=90)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)

    return arr, (lon_min, lon_max, lat_min, lat_max)


def lonlat_to_tile_xy(lon_deg, lat_deg, z):
    lat_rad = math.radians(max(min(lat_deg, 85.05112878), -85.05112878))
    n = 2.0 ** z
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def tile_xy_to_lonlat(xtile, ytile, z):
    n = 2.0 ** z
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lon_deg, lat_deg


def fetch_latest_rainviewer_frame_meta():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,*/*;q=0.8",
    }
    r = requests.get(RAINVIEWER_API, headers=headers, timeout=60)
    r.raise_for_status()
    js = r.json()

    host = js.get("host", "").strip()
    radar = js.get("radar", {})
    past = radar.get("past", []) or []
    if not host or not past:
        raise RuntimeError("RainViewer API returned no usable host/frame")

    frame = past[-1]
    path = str(frame.get("path", "")).strip()
    frame_time = frame.get("time")
    if not path:
        raise RuntimeError("RainViewer latest frame has no path")

    return host, path, frame_time, js.get("generated")


def fetch_rainviewer_rgba_for_bbox(lon_min, lon_max, lat_min, lat_max, z=RAIN_ZOOM, tile_size=RAIN_TILE_SIZE):
    host, path, frame_time, api_generated = fetch_latest_rainviewer_frame_meta()

    x0, y0 = lonlat_to_tile_xy(lon_min, lat_max, z)
    x1, y1 = lonlat_to_tile_xy(lon_max, lat_min, z)

    tx_min = int(math.floor(min(x0, x1)))
    tx_max = int(math.floor(max(x0, x1)))
    ty_min = int(math.floor(min(y0, y1)))
    ty_max = int(math.floor(max(y0, y1)))

    ntiles = 2 ** z
    tx_min = max(0, min(tx_min, ntiles - 1))
    tx_max = max(0, min(tx_max, ntiles - 1))
    ty_min = max(0, min(ty_min, ntiles - 1))
    ty_max = max(0, min(ty_max, ntiles - 1))

    mosaic_w = (tx_max - tx_min + 1) * tile_size
    mosaic_h = (ty_max - ty_min + 1) * tile_size
    mosaic = np.zeros((mosaic_h, mosaic_w, 4), dtype=np.uint8)

    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/png,*/*;q=0.8",
    }

    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            url = "%s%s/%s/%s/%s/%s/%s/%s.png" % (
                host,
                path,
                tile_size,
                z,
                tx,
                ty,
                RAIN_COLOR_SCHEME,
                RAIN_OPTIONS
            )
            r = session.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            arr = np.array(img, dtype=np.uint8)

            x_off = (tx - tx_min) * tile_size
            y_off = (ty - ty_min) * tile_size
            mosaic[y_off:y_off + tile_size, x_off:x_off + tile_size, :] = arr

    lon_left, lat_top = tile_xy_to_lonlat(tx_min, ty_min, z)
    lon_right, lat_bottom = tile_xy_to_lonlat(tx_max + 1, ty_max + 1, z)

    meta = {
        "frame_time": frame_time,
        "api_generated": api_generated,
        "host": host,
        "path": path,
        "z": z,
    }

    return mosaic, (lon_left, lon_right, lat_bottom, lat_top), meta


# =============================================================================
# SAMPLING / REPROJECTION
# =============================================================================
def bilinear_sample_rgba(src_rgba, src_extent, lon_grid, lat_grid):
    """
    src_extent = (lon_min, lon_max, lat_min, lat_max)
    src_rgba shape = (H, W, 4)
    lon_grid, lat_grid shape = (Ny, Nx)
    """
    lon_min, lon_max, lat_min, lat_max = src_extent
    h, w, _ = src_rgba.shape

    x = (lon_grid - lon_min) / (lon_max - lon_min) * (w - 1)
    y = (lat_max - lat_grid) / (lat_max - lat_min) * (h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (
        np.isfinite(x) & np.isfinite(y) &
        (x >= 0) & (x <= (w - 1)) &
        (y >= 0) & (y <= (h - 1))
    )

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    Ia = src_rgba[y0, x0].astype(np.float32)
    Ib = src_rgba[y0, x1].astype(np.float32)
    Ic = src_rgba[y1, x0].astype(np.float32)
    Id = src_rgba[y1, x1].astype(np.float32)

    out = (
        Ia * wa[..., None] +
        Ib * wb[..., None] +
        Ic * wc[..., None] +
        Id * wd[..., None]
    )

    out = np.clip(out, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def sample_rgba_to_projected_grid(src_rgba, src_extent, grid_x, grid_y, proj_to_wgs):
    lon_grid, lat_grid = proj_to_wgs.transform(grid_x, grid_y)
    lon_grid = np.asarray(lon_grid, dtype=np.float64)
    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    return bilinear_sample_rgba(src_rgba, src_extent, lon_grid, lat_grid)


# =============================================================================
# PLOTTING
# =============================================================================
def plot_greece_wgs84(product_label, region_name, bbox, src_rgba, src_extent, boundary_gdf, footer_right, footer_left, out_png):
    lon_min, lon_max, lat_min, lat_max = bbox

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.imshow(
        src_rgba,
        extent=(src_extent[0], src_extent[1], src_extent[2], src_extent[3]),
        origin="upper",
        interpolation="nearest"
    )

    boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_title(product_label, fontsize=14, pad=4, loc="center")
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    ax.grid(True, linewidth=0.3, alpha=0.4)

    ax.text(
        0.01, 0.01, footer_left,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )
    ax.text(
        0.99, 0.01, footer_right,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="right",
        va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )

    spacer_ax = reserve_right_legend_space(fig, ax)

    plt.subplots_adjust(top=0.98, bottom=0.08, left=0.08, right=0.92)
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight", bbox_extra_artists=[spacer_ax], pad_inches=0)
    plt.close(fig)


def plot_egsa_region(product_label, region_name, bbox, src_rgba, src_extent, greece_egsa, footer_right, footer_left, out_png):
    lon_min, lon_max, lat_min, lat_max = bbox

    wgs_to_egsa = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)
    egsa_to_wgs = Transformer.from_crs("EPSG:2100", "EPSG:4326", always_xy=True)

    corners_lon = [lon_min, lon_min, lon_max, lon_max]
    corners_lat = [lat_min, lat_max, lat_min, lat_max]
    corners_x, corners_y = wgs_to_egsa.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(corners_x)), float(np.max(corners_x))
    y_min, y_max = float(np.min(corners_y)), float(np.max(corners_y))

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, TARGET_GRID_N_REGION),
        np.linspace(y_min, y_max, TARGET_GRID_N_REGION)
    )

    proj_rgba = sample_rgba_to_projected_grid(src_rgba, src_extent, grid_x, grid_y, egsa_to_wgs)

    bbox_poly = box(x_min, y_min, x_max, y_max)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:2100")
    try:
        greece_clip = gpd.clip(greece_egsa, bbox_gdf)
    except Exception:
        greece_clip = greece_egsa

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.imshow(
        proj_rgba,
        extent=(x_min, x_max, y_min, y_max),
        origin="upper",
        interpolation="nearest"
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _ = egsa_to_wgs.transform(x, y_ref_for_lon)
        return "%.2f" % lon

    def fmt_lat(y, pos):
        _, lat = egsa_to_wgs.transform(x_ref_for_lat, y)
        return "%.2f" % lat

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_title(product_label, fontsize=14, pad=4, loc="center")
    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    ax.grid(True, linewidth=0.3, alpha=0.4)

    ax.text(
        0.01, 0.01, footer_left,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )
    ax.text(
        0.99, 0.01, footer_right,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="right",
        va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )

    spacer_ax = reserve_right_legend_space(fig, ax)

    plt.subplots_adjust(top=0.98, bottom=0.08, left=0.08, right=0.92)
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight", bbox_extra_artists=[spacer_ax], pad_inches=0)
    plt.close(fig)


def plot_cyprus_utm(product_label, region_name, bbox, src_rgba, src_extent, cyprus_wgs, footer_right, footer_left, out_png):
    lon_min, lon_max, lat_min, lat_max = bbox

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    utm_to_wgs = Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)

    corn_lon = np.array([lon_min, lon_max, lon_min, lon_max])
    corn_lat = np.array([lat_min, lat_min, lat_max, lat_max])
    corn_E, corn_N = to_utm.transform(corn_lon, corn_lat)
    e_min, e_max = float(np.min(corn_E)), float(np.max(corn_E))
    n_min, n_max = float(np.min(corn_N)), float(np.max(corn_N))

    grid_E, grid_N = np.meshgrid(
        np.linspace(e_min, e_max, TARGET_GRID_N_CYPRUS),
        np.linspace(n_min, n_max, TARGET_GRID_N_CYPRUS)
    )

    proj_rgba = sample_rgba_to_projected_grid(src_rgba, src_extent, grid_E, grid_N, utm_to_wgs)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.imshow(
        proj_rgba,
        extent=(e_min, e_max, n_min, n_max),
        origin="upper",
        interpolation="nearest"
    )

    if cyprus_wgs is not None:
        cyprus_utm = cyprus_wgs.to_crs("EPSG:32636")
        try:
            boundary_utm = cyprus_utm.geometry.union_all()
        except AttributeError:
            boundary_utm = unary_union(cyprus_utm.geometry)
        plot_boundary_proj(ax, boundary_utm, linewidth=0.5, color="black")
    else:
        print("⚠️ cyprus.geojson not found. Cyprus maps will be plotted without coastline overlay.")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)

    lon0 = (lon_min + lon_max) / 2.0
    lat0 = (lat_min + lat_max) / 2.0
    lon_step = 0.5
    lat_step = 0.5

    lon_ticks = np.arange(np.floor(lon_min / lon_step) * lon_step, lon_max + 1e-9, lon_step)
    lat_ticks = np.arange(np.floor(lat_min / lat_step) * lat_step, lat_max + 1e-9, lat_step)

    x_ticks_m, _ = to_utm.transform(lon_ticks, np.full_like(lon_ticks, lat0))
    _, y_ticks_m = to_utm.transform(np.full_like(lat_ticks, lon0), lat_ticks)

    ax.xaxis.set_major_locator(FixedLocator(x_ticks_m))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks_m))
    ax.set_xticklabels(["%.2f" % lon for lon in lon_ticks])
    ax.set_yticklabels(["%.2f" % lat for lat in lat_ticks])

    ax.set_title(product_label, fontsize=14, pad=4, loc="center")
    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    ax.grid(True, linewidth=0.3, alpha=0.4)

    ax.text(
        0.01, 0.01, footer_left,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )
    ax.text(
        0.99, 0.01, footer_right,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="right",
        va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )

    spacer_ax = reserve_right_legend_space(fig, ax)

    plt.subplots_adjust(top=0.98, bottom=0.08, left=0.08, right=0.92)
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight", bbox_extra_artists=[spacer_ax], pad_inches=0)
    plt.close(fig)


# =============================================================================
# PRODUCT RUNNERS
# =============================================================================
def build_and_save_cloud(region_key, cfg, output_dir, now_athens, greece_wgs, greece_egsa, cyprus_wgs):
    lon_min, lon_max, lat_min, lat_max = cfg["bbox"]

    src_rgba, src_extent = fetch_cloud_wms_rgba(lon_min, lon_max, lat_min, lat_max)

    ts = now_athens.strftime("%Y-%m-%d-%H-%M")
    out_png = os.path.join(output_dir, cfg["cloud_prefix"] + ts + ".png")
    out_latest = os.path.join(output_dir, cfg["cloud_latest"])

    footer_left = "Δημιουργήθηκε για το e-kairos.gr\n%s %s\nΠηγή: EUMETView WMS" % (
        fmt_generated(now_athens),
        athens_abbrev(now_athens)
    )
    footer_right = "Layer: %s" % CLOUD_WMS_LAYER

    if cfg["mode"] == "greece_wgs84":
        plot_greece_wgs84("Υπολογ. τελευταία διαθέσιμη νεφοκάλυψη", cfg["name"], cfg["bbox"], src_rgba, src_extent, greece_wgs, footer_right, footer_left, out_png)
    elif cfg["mode"] == "egsa_region":
        plot_egsa_region("Υπολογ. τελευταία διαθέσιμη νεφοκάλυψη", cfg["name"], cfg["bbox"], src_rgba, src_extent, greece_egsa, footer_right, footer_left, out_png)
    elif cfg["mode"] == "cyprus_utm":
        plot_cyprus_utm("Υπολογ. τελευταία διαθέσιμη νεφοκάλυψη", cfg["name"], cfg["bbox"], src_rgba, src_extent, cyprus_wgs, footer_right, footer_left, out_png)
    else:
        raise RuntimeError("Unknown mode for cloud: %s" % cfg["mode"])

    shutil.copy(out_png, out_latest)
    print("✅ Saved:", out_png)
    print("✅ Saved:", out_latest)

    return [out_png, out_latest], (cfg["cloud_prefix"], cfg["cloud_latest"], int(cfg.get("remote_keep", 200)))


def build_and_save_rain(region_key, cfg, output_dir, now_athens, greece_wgs, greece_egsa, cyprus_wgs):
    lon_min, lon_max, lat_min, lat_max = cfg["bbox"]

    src_rgba, src_extent, meta = fetch_rainviewer_rgba_for_bbox(lon_min, lon_max, lat_min, lat_max, z=RAIN_ZOOM, tile_size=RAIN_TILE_SIZE)

    frame_time = meta.get("frame_time")
    api_generated = meta.get("api_generated")

    frame_time_str = "unknown"
    api_gen_str = "unknown"
    if frame_time is not None:
        try:
            dt_frame = datetime.fromtimestamp(int(frame_time), tz=ZoneInfo("UTC")).astimezone(ATHENS_TZ)
            frame_time_str = dt_frame.strftime("%Y-%m-%d %H:%M") + " " + athens_abbrev(dt_frame)
        except Exception:
            pass
    if api_generated is not None:
        try:
            dt_gen = datetime.fromtimestamp(int(api_generated), tz=ZoneInfo("UTC")).astimezone(ATHENS_TZ)
            api_gen_str = dt_gen.strftime("%Y-%m-%d %H:%M") + " " + athens_abbrev(dt_gen)
        except Exception:
            pass

    ts = now_athens.strftime("%Y-%m-%d-%H-%M")
    out_png = os.path.join(output_dir, cfg["rain_prefix"] + ts + ".png")
    out_latest = os.path.join(output_dir, cfg["rain_latest"])

    footer_left = "Δημιουργήθηκε για το e-kairos.gr\n%s %s\nΠηγή: RainViewer radar mosaic" % (
        fmt_generated(now_athens),
        athens_abbrev(now_athens)
    )
    footer_right = "Frame: %s\nAPI generated: %s" % (frame_time_str, api_gen_str)

    if cfg["mode"] == "greece_wgs84":
        plot_greece_wgs84("Υπολογ. τελευταία διαθέσιμη εκτίμηση υετού", cfg["name"], cfg["bbox"], src_rgba, src_extent, greece_wgs, footer_right, footer_left, out_png)
    elif cfg["mode"] == "egsa_region":
        plot_egsa_region("Υπολογ. τελευταία διαθέσιμη εκτίμηση υετού", cfg["name"], cfg["bbox"], src_rgba, src_extent, greece_egsa, footer_right, footer_left, out_png)
    elif cfg["mode"] == "cyprus_utm":
        plot_cyprus_utm("Υπολογ. τελευταία διαθέσιμη εκτίμηση υετού", cfg["name"], cfg["bbox"], src_rgba, src_extent, cyprus_wgs, footer_right, footer_left, out_png)
    else:
        raise RuntimeError("Unknown mode for rain: %s" % cfg["mode"])

    shutil.copy(out_png, out_latest)
    print("✅ Saved:", out_png)
    print("✅ Saved:", out_latest)

    return [out_png, out_latest], (cfg["rain_prefix"], cfg["rain_latest"], int(cfg.get("remote_keep", 200)))


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        default="all",
        choices=["all", "greece", "attica", "crete", "negreece", "swgreece", "cyprus"],
        help="Which region to run."
    )
    args = parser.parse_args()

    output_dir = os.path.join(BASE_DIR, "testmaps")
    os.makedirs(output_dir, exist_ok=True)

    now_athens = datetime.now(ATHENS_TZ)

    # Load boundaries once
    greece_wgs = load_greece_wgs84()
    greece_egsa = greece_wgs.to_crs("EPSG:2100")
    cyprus_wgs = load_cyprus_wgs84_or_none()

    selected = []
    if args.region == "all":
        selected = ["greece", "attica", "crete", "negreece", "swgreece", "cyprus"]
    else:
        selected = [args.region]

    upload_files = []
    prune_specs = []

    for rk in selected:
        cfg = REGIONS[rk]

        print("\n====================")
        print("RUN:", cfg["name"])
        print("====================")

        try:
            files, prune = build_and_save_cloud(rk, cfg, output_dir, now_athens, greece_wgs, greece_egsa, cyprus_wgs)
            upload_files.extend(files)
            prune_specs.append(prune)
        except Exception as e:
            print("❌ Cloud generation failed for %s: %s" % (cfg["name"], e))

        try:
            files, prune = build_and_save_rain(rk, cfg, output_dir, now_athens, greece_wgs, greece_egsa, cyprus_wgs)
            upload_files.extend(files)
            prune_specs.append(prune)
        except Exception as e:
            print("❌ Rain generation failed for %s: %s" % (cfg["name"], e))

    # de-duplicate prune specs
    seen = set()
    prune_specs_unique = []
    for item in prune_specs:
        key = (item[0], item[1], item[2])
        if key not in seen:
            seen.add(key)
            prune_specs_unique.append(item)

    # Upload and prune once
    ftp_upload_many_and_prune(upload_files, prune_specs_unique)


if __name__ == "__main__":
    main()
