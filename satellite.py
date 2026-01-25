#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# satellite.py
#
# Satellite precipitation (IMERG) map over the FULL bbox (land + sea),
# with the SAME look as your station-based rain_intensity_ map:
# - same bbox, same GRID_N, same legend steps, same labels, same colormap
# - only the data source changes (NASA IMERG via Earthdata GIS ArcGIS ImageServer)
#
# Also:
# - decrypts greece.geojson.enc -> greece.geojson using GEOJSON_PASS (same pattern as your other scripts)
# - uses FTP_HOST/FTP_USER/FTP_PASS env secrets for optional upload + remote prune
#
# Source (ArcGIS ImageServer):
#   https://gis.earthdata.nasa.gov/image/rest/services/GESDISC/GPM_3IMERGHHE/ImageServer
#
# Outputs:
#   rainintensitymaps/
#     rain_intensity_sat_YYYY-MM-DD-HH-MM.png
#     latest_sat.png
#
# Required secrets/env in GitHub Actions:
#   GEOJSON_PASS  (to decrypt greece.geojson.enc)
# Optional for upload:
#   FTP_HOST, FTP_USER, FTP_PASS

import os
import re
import time
import random
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO

import numpy as np
import numpy.ma as ma
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import requests
import rasterio
from ftplib import FTP_TLS
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========================
# CONFIG (match your station map)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Greece outline: encrypted in repo, decrypted at runtime
GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
GEOJSON_ENC = os.path.join(BASE_DIR, "greece.geojson.enc")
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()

# Map grid/extent (EXACTLY as in your code)
GRID_N = 300
GRID_LON_MIN, GRID_LON_MAX = 19.0, 30.0
GRID_LAT_MIN, GRID_LAT_MAX = 34.5, 42.5

# Colormap and bounds (EXACTLY as in your code)
CMAP = ListedColormap(["#deebf7", "#9ecae1", "#4292c6", "#08519c"])
CMAP.set_under("#ffffff")
CMAP.set_over("#08306b")
CMAP.set_bad("#ffffff")
BOUNDS = [0.1, 2, 6, 36, 100]
NORM = BoundaryNorm(boundaries=BOUNDS, ncolors=CMAP.N)

# Output naming (separate product for overlay)
OUTPUT_DIR = os.path.join(BASE_DIR, "rainintensitymaps")
PREFIX = "rain_intensity_sat_"
LATEST_NAME = "latest_sat.png"
KEEP_REMOTE = 40

# IMERG ImageServer (Early Run half-hourly)
IMERG_IMAGESERVER = "https://gis.earthdata.nasa.gov/image/rest/services/GESDISC/GPM_3IMERGHHE/ImageServer"

# Robustness
TIMEOUT = 60
TRIES = 6


# =========================
# Decryption helpers
# =========================
def ensure_geojson():
    """
    Decrypt greece.geojson.enc -> greece.geojson (repo root) using GEOJSON_PASS.
    """
    if os.path.exists(GEOJSON_PATH):
        return
    if not os.path.exists(GEOJSON_ENC):
        return
    if not GEOJSON_PASS:
        raise SystemExit("GeoJSON missing and GEOJSON_PASS not set to decrypt greece.geojson.enc")

    try:
        subprocess.check_call([
            "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
            "-in", GEOJSON_ENC, "-out", GEOJSON_PATH, "-pass", "pass:" + GEOJSON_PASS
        ])
        print("🔓 Decrypted greece.geojson.enc -> greece.geojson")
    except FileNotFoundError:
        raise SystemExit("OpenSSL not found. Install it or decrypt greece.geojson.enc in a prior CI step.")
    except subprocess.CalledProcessError as e:
        raise SystemExit("OpenSSL decryption failed for greece.geojson.enc: %s" % e)


# =========================
# FTP (optional)
# =========================
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()

def ftp_enabled():
    return bool(FTP_HOST and FTP_USER and FTP_PASS)

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
      - Keep latest_sat.png
      - Keep newest `keep` files matching rain_intensity_sat_YYYY-MM-DD-HH-MM.png
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


# =========================
# Robust GET
# =========================
def robust_get(url: str, params=None, timeout: int = 60, tries: int = 6, stream: bool = False):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    last_err = None
    s = requests.Session()

    for attempt in range(1, tries + 1):
        try:
            r = s.get(url, params=params, headers=headers, timeout=timeout, stream=stream)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 20) + random.random()
            print(f"[get] attempt {attempt}/{tries} failed: {e}. Retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed GET: {url}") from last_err


# =========================
# ImageServer helpers
# =========================
def get_latest_stdtime_ms() -> int:
    """
    Query the ImageServer catalog for the latest StdTime (milliseconds since epoch, UTC).
    """
    q_url = IMERG_IMAGESERVER + "/query"
    params = {
        "where": "1=1",
        "outFields": "StdTime",
        "orderByFields": "StdTime DESC",
        "resultRecordCount": 1,
        "f": "pjson",
    }
    r = robust_get(q_url, params=params, timeout=TIMEOUT, tries=TRIES, stream=False)
    js = r.json()
    feats = js.get("features", [])
    if not feats:
        raise RuntimeError("No features returned by ImageServer /query (cannot determine latest time).")
    attrs = feats[0].get("attributes", {})
    stdtime = attrs.get("StdTime")
    if stdtime is None:
        stdtime = attrs.get("stdtime")
    if stdtime is None:
        raise RuntimeError("Latest feature did not contain StdTime.")
    return int(stdtime)

def export_imerg_tiff(bbox, size_wh, time_ms: int) -> bytes:
    """
    Export the raster for a given bbox and time slice as TIFF bytes.
    bbox: (xmin, ymin, xmax, ymax) in EPSG:4326
    size_wh: (width, height) pixels
    """
    ex_url = IMERG_IMAGESERVER + "/exportImage"
    xmin, ymin, xmax, ymax = bbox
    w, h = size_wh

    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{w},{h}",
        "format": "tiff",
        "time": str(time_ms),
        "f": "image",
    }
    r = robust_get(ex_url, params=params, timeout=TIMEOUT, tries=TRIES, stream=True)
    return r.content


def main():
    ensure_geojson()

    if not os.path.exists(GEOJSON_PATH):
        print(f"❌ Missing: {GEOJSON_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Greece boundary (same as your code)
    greece = gpd.read_file(GEOJSON_PATH)
    if greece.crs is None:
        greece = greece.set_crs("EPSG:4326")
    if greece.crs.to_string() != "EPSG:4326":
        greece = greece.to_crs("EPSG:4326")

    # Find latest available IMERG time on the service (UTC millis)
    latest_ms = get_latest_stdtime_ms()
    latest_utc = datetime.fromtimestamp(latest_ms / 1000.0, tz=ZoneInfo("UTC"))
    athens_now = datetime.now(ZoneInfo("Europe/Athens"))
    print("Latest IMERG StdTime (UTC):", latest_utc.isoformat())
    print("Athens now:", athens_now.isoformat())

    # Export TIFF for your exact bbox and grid size
    bbox = (GRID_LON_MIN, GRID_LAT_MIN, GRID_LON_MAX, GRID_LAT_MAX)
    tiff_bytes = export_imerg_tiff(bbox=bbox, size_wh=(GRID_N, GRID_N), time_ms=latest_ms)

    # Read TIFF bytes to array
    with rasterio.open(BytesIO(tiff_bytes)) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

    # ArcGIS export often returns top-to-bottom rows; your plotting uses origin="lower".
    # Flip vertically to align geography.
    arr = np.flipud(arr)

    # Mask invalids for plotting
    masked_array = ma.masked_invalid(arr)

    # Output filenames
    timestamp = athens_now.strftime("%Y-%m-%d-%H-%M")
    output_file = os.path.join(OUTPUT_DIR, f"{PREFIX}{timestamp}.png")
    latest_output = os.path.join(OUTPUT_DIR, LATEST_NAME)

    # === PLOTTING (match your existing layout) ===
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        masked_array,
        extent=(GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX),
        origin="lower",
        cmap=CMAP,
        norm=NORM,
        alpha=0.7
    )

    greece.boundary.plot(ax=ax, color="black", linewidth=0.5)

    # Contour borders only (same levels)
    contour_levels = [0.1, 2, 6, 36, 100]
    grid_x, grid_y = np.meshgrid(
        np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_N),
        np.linspace(GRID_LAT_MIN, GRID_LAT_MAX, GRID_N)
    )
    ax.contour(grid_x, grid_y, arr, levels=contour_levels, colors="black", linewidths=1)

    # Colorbar (same)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax, boundaries=BOUNDS, extend="both")
    cbar.set_ticks([2, 6, 36, 100])
    cbar.set_ticklabels(["2", "6", "36", "100"])
    cbar.set_label("Ραγδαιότητα υετού (mm/h)", fontsize=12)

    # Titles/labels (same)
    ax.set_title("Υπολογ. τελευταία διαθέσιμη ραγδαιότητα υετού", fontsize=14, pad=10, loc="center")
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    # Footer (same style)
    timestamp_text = athens_now.strftime("%Y-%m-%d %H:%M %Z")
    left_text = f"Δημιουργήθηκε για το e-kairos.gr\n{timestamp_text}"
    right_text = f"v_sat-imerg\nStdTime: {latest_utc.strftime('%Y-%m-%d %H:%M UTC')}"

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

    # FTP upload + prune (optional)
    try:
        upload_to_ftp(output_file)
        upload_to_ftp(latest_output)
        prune_remote_pngs(keep=KEEP_REMOTE, prefix=PREFIX, latest_name=LATEST_NAME)
    except Exception as e:
        print(f"⚠️ FTP upload/prune failed: {e}")


if __name__ == "__main__":
    main()
