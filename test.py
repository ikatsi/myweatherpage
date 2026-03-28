#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py

Independent cloudiness and rain-layer test maps for the same domains used in
rainintensityall.py.

What it does
- Creates cloudiness PNGs from EUMETView WMS for:
    Greece, Attica, Crete, NE Greece, SW Greece, Cyprus
- Creates rain/radar PNGs from RainViewer mosaic tiles for the same domains
- Uses greece.geojson (with the same decryption pattern as rainintensityall.py)
  for Greece + Attica + Crete + NE Greece + SW Greece
- Uses cyprus.geojson if present for Cyprus. If missing, Cyprus is still plotted
  on its bbox without a coastline overlay.
- Saves timestamped and static latest PNGs locally
- Uploads both via FTPS using the same env secrets as rainintensityall.py
- Prunes older remote timestamped PNGs per prefix, same logic as rainintensityall.py

Notes
- Cloud maps are image overlays from EUMETView WMS.
- Rain maps are RainViewer radar mosaics, suitable for visual testing.
- This script does not depend on docs/ or GitHub Pages.
"""

import io
import math
import os
import re
import shutil
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from ftplib import FTP_TLS


# =============================================================================
# GLOBAL CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATHENS_TZ = ZoneInfo("Europe/Athens")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) e-kairos-cloud-rain-test/2.0"

# Optional FTP, same pattern as rainintensityall.py
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()

# Assets
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()
GREECE_GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
GREECE_GEOJSON_ENC = os.path.join(BASE_DIR, "greece.geojson.enc")
CYPRUS_GEOJSON_PATH = os.path.join(BASE_DIR, "cyprus.geojson")

# Local output folder only; upload is via FTPS
OUTPUT_DIR = os.path.join(BASE_DIR, "testmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cloud source: EUMETView WMS
CLOUD_WMS_URL = os.environ.get("CLOUD_WMS_URL", "https://view.eumetsat.int/geoserver/wms").strip()
CLOUD_WMS_LAYER = os.environ.get("CLOUD_WMS_LAYER", "msg_fes:clm").strip()
CLOUD_WIDTH = int(os.environ.get("CLOUD_WIDTH", "1400"))
CLOUD_HEIGHT = int(os.environ.get("CLOUD_HEIGHT", "1000"))

# Rain source: RainViewer latest radar composite tiles
RAINVIEWER_API = os.environ.get("RAINVIEWER_API", "https://api.rainviewer.com/public/weather-maps.json").strip()
RAIN_ZOOM = int(os.environ.get("RAIN_ZOOM", "6"))
RAIN_TILE_SIZE = int(os.environ.get("RAIN_TILE_SIZE", "512"))
RAIN_COLOR_SCHEME = int(os.environ.get("RAIN_COLOR_SCHEME", "2"))
RAIN_SMOOTH = int(os.environ.get("RAIN_SMOOTH", "1"))
RAIN_SNOW = int(os.environ.get("RAIN_SNOW", "1"))


REGIONS = {
    "greece": {
        "name": "Greece",
        "bbox": (19.0, 34.5, 30.0, 42.5),
        "boundary": "greece",
        "cloud_prefix": "cloudiness_",
        "cloud_latest": "latestcloudiness.png",
        "rain_prefix": "rain_rate_",
        "rain_latest": "latestrainrate.png",
        "remote_keep": 200,
    },
    "attica": {
        "name": "Attica",
        "bbox": (22.7, 37.5, 25.0, 38.7),
        "boundary": "greece",
        "cloud_prefix": "cloudiness_attica_",
        "cloud_latest": "latestcloudinessattica.png",
        "rain_prefix": "rain_rate_attica_",
        "rain_latest": "latestrainrateattica.png",
        "remote_keep": 200,
    },
    "crete": {
        "name": "Crete",
        "bbox": (23.37, 34.7, 26.4, 35.78),
        "boundary": "greece",
        "cloud_prefix": "cloudiness_crete_",
        "cloud_latest": "latestcloudinesscrete.png",
        "rain_prefix": "rain_rate_crete_",
        "rain_latest": "latestrainratecrete.png",
        "remote_keep": 200,
    },
    "negreece": {
        "name": "NE Greece",
        "bbox": (22.0, 39.7, 26.6, 41.8),
        "boundary": "greece",
        "cloud_prefix": "cloudiness_negreece_",
        "cloud_latest": "latestcloudinessnegreece.png",
        "rain_prefix": "rain_rate_negreece_",
        "rain_latest": "latestrainratenegreece.png",
        "remote_keep": 200,
    },
    "swgreece": {
        "name": "SW Greece",
        "bbox": (20.0, 36.0, 24.0, 39.0),
        "boundary": "greece",
        "cloud_prefix": "cloudiness_swgreece_",
        "cloud_latest": "latestcloudinessswgreece.png",
        "rain_prefix": "rain_rate_swgreece_",
        "rain_latest": "latestrainrateswgreece.png",
        "remote_keep": 200,
    },
    "cyprus": {
        "name": "Cyprus",
        "bbox": (32.0, 34.4, 34.9, 35.9),
        "boundary": "cyprus",
        "cloud_prefix": "cloudiness_cyprus_",
        "cloud_latest": "latestcloudinesscyprus.png",
        "rain_prefix": "rain_rate_cyprus_",
        "rain_latest": "latestrainratecyprus.png",
        "remote_keep": 144,
    },
}


# =============================================================================
# HELPERS
# =============================================================================
def ftp_enabled():
    return bool(FTP_HOST and FTP_USER and FTP_PASS)


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
            print(f"ℹ️ No timestamped PNGs to prune remotely for prefix {prefix}")
            return

        timestamped.sort()
        if len(timestamped) <= keep:
            print(f"ℹ️ {len(timestamped)} timestamped files ≤ keep={keep} for {prefix}. Nothing to delete.")
            return

        for fname in timestamped[:-keep]:
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


def athens_abbrev(dt: datetime) -> str:
    try:
        dt_ath = dt.astimezone(ATHENS_TZ)
        is_dst = bool(dt_ath.dst()) and dt_ath.dst() != timedelta(0)
        return "EEST" if is_dst else "EET"
    except Exception:
        return "EET"


def transparent_bbox(pad=0.3, rounded=True):
    boxstyle = ("round,pad=" + str(pad)) if rounded else ("square,pad=" + str(pad))
    return dict(
        facecolor=(1, 1, 1, 0.0),
        edgecolor=(0, 0, 0, 0.0),
        boxstyle=boxstyle,
    )


def ensure_greece_geojson():
    if os.path.exists(GREECE_GEOJSON_PATH):
        return
    if not os.path.exists(GREECE_GEOJSON_ENC):
        raise SystemExit("❌ Missing greece.geojson and greece.geojson.enc")
    if not GEOJSON_PASS:
        raise SystemExit("❌ GEOJSON_PASS secret/env not set")

    try:
        subprocess.check_call([
            "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
            "-in", GREECE_GEOJSON_ENC,
            "-out", GREECE_GEOJSON_PATH,
            "-pass", "pass:" + GEOJSON_PASS,
        ])
    except FileNotFoundError:
        raise SystemExit("❌ OpenSSL not available on runner")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"❌ Failed to decrypt greece.geojson.enc: {e}")


def load_boundary_geojson(path: str):
    if not os.path.exists(path):
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def fetch_bytes(url: str, params=None, timeout=180):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.content, r.headers


def fetch_json(url: str, timeout=120):
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# =============================================================================
# CLOUDINESS (EUMETView WMS)
# =============================================================================
def get_latest_wms_time(endpoint: str, layer_name: str):
    params = {
        "service": "WMS",
        "request": "GetCapabilities",
        "version": "1.3.0",
    }
    data, _ = fetch_bytes(endpoint, params=params, timeout=180)
    root = ET.fromstring(data)

    def strip(tag):
        return tag.split("}", 1)[-1] if "}" in tag else tag

    latest = None
    for layer in root.iter():
        if strip(layer.tag) != "Layer":
            continue
        name = None
        for child in layer:
            if strip(child.tag) == "Name":
                name = (child.text or "").strip()
                break
        if name != layer_name:
            continue
        for child in layer:
            tag = strip(child.tag)
            if tag not in ("Dimension", "Extent"):
                continue
            dim_name = (child.attrib.get("name") or child.attrib.get("Name") or "").lower()
            if dim_name != "time":
                continue
            txt = (child.text or "").strip()
            if not txt:
                continue
            if "," in txt:
                latest = txt.split(",")[-1].strip()
            elif "/" in txt:
                latest = None
            else:
                latest = txt
            break
        break
    return latest


def fetch_cloud_image(extent):
    lon_min, lat_min, lon_max, lat_max = extent
    latest_time = None
    try:
        latest_time = get_latest_wms_time(CLOUD_WMS_URL, CLOUD_WMS_LAYER)
    except Exception as e:
        print(f"⚠️ Could not determine WMS latest time for {CLOUD_WMS_LAYER}: {e}")

    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": CLOUD_WMS_LAYER,
        "styles": "",
        "crs": "EPSG:4326",
        "bbox": f"{lat_min},{lon_min},{lat_max},{lon_max}",
        "width": str(CLOUD_WIDTH),
        "height": str(CLOUD_HEIGHT),
        "format": "image/png",
        "transparent": "FALSE",
    }
    if latest_time:
        params["time"] = latest_time

    data, headers = fetch_bytes(CLOUD_WMS_URL, params=params, timeout=240)
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    return img, latest_time, headers


# =============================================================================
# RAIN (RainViewer mosaic)
# =============================================================================
def lonlat_to_tile_xy(lon, lat, zoom):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y


def fetch_rainviewer_latest_frame():
    meta = fetch_json(RAINVIEWER_API, timeout=180)
    radar = meta.get("radar", {})
    past = radar.get("past", [])
    if not past:
        raise RuntimeError("RainViewer returned no radar.past frames")
    frame = past[-1]
    host = meta.get("host", "https://tilecache.rainviewer.com")
    return {
        "host": host.rstrip("/"),
        "path": frame["path"],
        "time": int(frame["time"]),
        "generated": int(meta.get("generated", frame["time"])),
    }


def fetch_rain_mosaic(extent, zoom=6, tile_size=512):
    lon_min, lat_min, lon_max, lat_max = extent
    frame = fetch_rainviewer_latest_frame()

    x0f, y1f = lonlat_to_tile_xy(lon_min, lat_min, zoom)
    x1f, y0f = lonlat_to_tile_xy(lon_max, lat_max, zoom)

    x0 = int(math.floor(min(x0f, x1f)))
    x1 = int(math.floor(max(x0f, x1f)))
    y0 = int(math.floor(min(y0f, y1f)))
    y1 = int(math.floor(max(y0f, y1f)))

    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    mosaic = Image.new("RGBA", (cols * tile_size, rows * tile_size), (255, 255, 255, 0))

    for xt in range(x0, x1 + 1):
        for yt in range(y0, y1 + 1):
            tile_url = (
                f"{frame['host']}{frame['path']}/"
                f"{tile_size}/{zoom}/{xt}/{yt}/{RAIN_COLOR_SCHEME}/{RAIN_SMOOTH}_{RAIN_SNOW}.png"
            )
            try:
                data, _ = fetch_bytes(tile_url, timeout=180)
                tile = Image.open(io.BytesIO(data)).convert("RGBA")
            except Exception as e:
                print(f"⚠️ Failed tile z={zoom} x={xt} y={yt}: {e}")
                tile = Image.new("RGBA", (tile_size, tile_size), (255, 255, 255, 0))
            mosaic.paste(tile, ((xt - x0) * tile_size, (yt - y0) * tile_size))

    px0 = (x0f - x0) * tile_size
    px1 = (x1f - x0) * tile_size
    py0 = (y0f - y0) * tile_size
    py1 = (y1f - y0) * tile_size

    left = int(round(min(px0, px1)))
    right = int(round(max(px0, px1)))
    top = int(round(min(py0, py1)))
    bottom = int(round(max(py0, py1)))

    crop = mosaic.crop((left, top, right, bottom))
    return crop, frame


# =============================================================================
# PLOTTING
# =============================================================================
def fmt_athens_from_unix(ts_utc):
    if ts_utc is None:
        return "—"
    dt = datetime.fromtimestamp(int(ts_utc), tz=timezone.utc).astimezone(ATHENS_TZ)
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def save_map(image_rgba, boundary_gdf, extent, title, left_text, right_text, out_path):
    lon_min, lat_min, lon_max, lat_max = extent
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    ax.imshow(np.asarray(image_rgba), extent=(lon_min, lon_max, lat_min, lat_max), origin="upper")

    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=11)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=11)
    ax.set_title(title, fontsize=14, pad=10, loc="center")
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)

    ax.text(
        0.01, 0.01, left_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="left",
        va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )
    ax.text(
        0.99, 0.01, right_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        ha="right",
        va="bottom",
        bbox=transparent_bbox(pad=0.3, rounded=True)
    )

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.92)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


# =============================================================================
# RUNNERS
# =============================================================================
def render_one_product(kind: str, region_key: str, region_cfg: dict, boundary_gdf):
    athens_now = datetime.now(ATHENS_TZ)
    ts = athens_now.strftime("%Y-%m-%d-%H-%M")
    timestamp_text = athens_now.strftime("%Y-%m-%d %H:%M") + f" {athens_abbrev(athens_now)}"
    extent = region_cfg["bbox"]

    if kind == "cloud":
        prefix = region_cfg["cloud_prefix"]
        latest_name = region_cfg["cloud_latest"]
        out_png = os.path.join(OUTPUT_DIR, f"{prefix}{ts}.png")
        out_latest = os.path.join(OUTPUT_DIR, latest_name)

        cloud_img, cloud_time, _headers = fetch_cloud_image(extent)
        left_text = (
            f"Δημιουργήθηκε για το e-kairos.gr\n"
            f"{timestamp_text}\n"
            f"Πηγή: EUMETView WMS"
        )
        right_text = (
            f"Layer: {CLOUD_WMS_LAYER}\n"
            f"Χρόνος layer: {cloud_time or 'latest/default'}"
        )
        title = f"Υπολογ. τελευταία διαθέσιμη νέφωση {region_cfg['name']}"

        save_map(cloud_img, boundary_gdf, extent, title, left_text, right_text, out_png)
        shutil.copy(out_png, out_latest)
        print(f"✅ Saved: {out_latest}")

        try:
            ftp_upload_file(out_png)
            ftp_upload_file(out_latest)
            ftp_prune_timestamped(prefix=prefix, latest_name=latest_name, keep=int(region_cfg["remote_keep"]))
        except Exception as e:
            print(f"⚠️ FTP upload/prune failed for {region_key} cloud: {e}")

    elif kind == "rain":
        prefix = region_cfg["rain_prefix"]
        latest_name = region_cfg["rain_latest"]
        out_png = os.path.join(OUTPUT_DIR, f"{prefix}{ts}.png")
        out_latest = os.path.join(OUTPUT_DIR, latest_name)

        rain_img, rain_meta = fetch_rain_mosaic(extent, zoom=RAIN_ZOOM, tile_size=RAIN_TILE_SIZE)
        left_text = (
            f"Δημιουργήθηκε για το e-kairos.gr\n"
            f"{timestamp_text}\n"
            f"Πηγή: RainViewer radar mosaic"
        )
        right_text = (
            f"Frame UTC: {fmt_athens_from_unix(rain_meta.get('time'))}\n"
            f"API generated: {fmt_athens_from_unix(rain_meta.get('generated'))}"
        )
        title = f"Υπολογ. τελευταία διαθέσιμη βροχή {region_cfg['name']}"

        save_map(rain_img, boundary_gdf, extent, title, left_text, right_text, out_png)
        shutil.copy(out_png, out_latest)
        print(f"✅ Saved: {out_latest}")

        try:
            ftp_upload_file(out_png)
            ftp_upload_file(out_latest)
            ftp_prune_timestamped(prefix=prefix, latest_name=latest_name, keep=int(region_cfg["remote_keep"]))
        except Exception as e:
            print(f"⚠️ FTP upload/prune failed for {region_key} rain: {e}")

    else:
        raise ValueError(f"Unknown product kind: {kind}")


def main():
    ensure_greece_geojson()
    greece_gdf = load_boundary_geojson(GREECE_GEOJSON_PATH)
    cyprus_gdf = load_boundary_geojson(CYPRUS_GEOJSON_PATH)

    for region_key, cfg in REGIONS.items():
        print("\n====================")
        print(f"RUN: {cfg['name']}")
        print("====================")

        if cfg["boundary"] == "greece":
            boundary = greece_gdf
        elif cfg["boundary"] == "cyprus":
            boundary = cyprus_gdf
            if boundary is None:
                print("⚠️ cyprus.geojson not found. Cyprus maps will be plotted without coastline overlay.")
        else:
            boundary = None

        render_one_product("cloud", region_key, cfg, boundary)
        render_one_product("rain", region_key, cfg, boundary)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
        raise
