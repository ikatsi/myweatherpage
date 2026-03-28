#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py

Creates two independent Greece-area test maps:
  1) Cloudiness from EUMETView WMS (default: MSG cloud mask)
  2) Rain / radar overlay from RainViewer latest radar mosaic

Output files are written to:
  docs/testmaps/

Assets:
  - greece.geojson.enc -> greece.geojson using GEOJSON_PASS

Notes:
  - The cloud layer is requested over the Greece bounding box from EUMETView WMS.
  - The rain layer in this test script is a radar mosaic image layer, not a true OPERA
    NIMBUS mm/h data service. It is suitable for visual testing, not for a quantitative
    public rain-rate product.
"""

import io
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from zoneinfo import ZoneInfo


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATHENS_TZ = ZoneInfo("Europe/Athens")

# Greece assets
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()
GREECE_GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
GREECE_GEOJSON_ENC = os.path.join(BASE_DIR, "greece.geojson.enc")

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "docs", "testmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Greece bbox in lon/lat degrees
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = 19.0, 34.5, 30.0, 42.5
MAP_EXTENT = (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

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
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) e-kairos-test/1.0"


def ensure_geojson():
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


def load_greece():
    ensure_geojson()
    gdf = gpd.read_file(GREECE_GEOJSON_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def fetch_bytes(url: str, params=None, timeout=120):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.content, r.headers


# -----------------------------------------------------------------------------
# Cloud map via EUMETView WMS
# -----------------------------------------------------------------------------
def get_latest_wms_time(endpoint: str, layer_name: str):
    params = {
        "service": "WMS",
        "request": "GetCapabilities",
        "version": "1.3.0",
    }
    data, _headers = fetch_bytes(endpoint, params=params, timeout=120)
    root = ET.fromstring(data)

    ns = {
        "wms": "http://www.opengis.net/wms",
    }

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
            if (child.attrib.get("name") or child.attrib.get("Name") or "").lower() != "time":
                continue
            txt = (child.text or "").strip()
            if not txt:
                continue
            # Typical cases: comma-separated timestamps, or start/end/period
            if "," in txt:
                latest = txt.split(",")[-1].strip()
            elif "/" in txt:
                # If interval-like, keep None and let WMS serve default latest.
                latest = None
            else:
                latest = txt
            break
        break
    return latest


def fetch_cloud_image():
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
        # WMS 1.3.0 axis order for EPSG:4326 is lat,lon
        "bbox": f"{LAT_MIN},{LON_MIN},{LAT_MAX},{LON_MAX}",
        "width": str(CLOUD_WIDTH),
        "height": str(CLOUD_HEIGHT),
        "format": "image/png",
        "transparent": "FALSE",
    }
    if latest_time:
        params["time"] = latest_time

    data, headers = fetch_bytes(CLOUD_WMS_URL, params=params, timeout=180)
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    return img, latest_time, headers


# -----------------------------------------------------------------------------
# Rain map via RainViewer tile stitching
# -----------------------------------------------------------------------------
def lonlat_to_tile_xy(lon, lat, zoom):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return x, y


def tile_xy_to_lonlat(x, y, zoom):
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def fetch_json(url):
    r = SESSION.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_rainviewer_latest_frame():
    meta = fetch_json(RAINVIEWER_API)
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
                data, _headers = fetch_bytes(tile_url, timeout=120)
                tile = Image.open(io.BytesIO(data)).convert("RGBA")
            except Exception as e:
                print(f"⚠️ Failed tile z={zoom} x={xt} y={yt}: {e}")
                tile = Image.new("RGBA", (tile_size, tile_size), (255, 255, 255, 0))
            mosaic.paste(tile, ((xt - x0) * tile_size, (yt - y0) * tile_size))

    # Pixel coordinates of the requested bbox within the stitched mosaic
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


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def fmt_athens(ts_utc):
    if ts_utc is None:
        return "—"
    dt = datetime.fromtimestamp(int(ts_utc), tz=timezone.utc).astimezone(ATHENS_TZ)
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def save_map(image_rgba, greece_gdf, title, subtitle_left, subtitle_right, out_path):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    ax.imshow(np.asarray(image_rgba), extent=MAP_EXTENT, origin="upper")
    greece_gdf.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=11)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=11)
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(True, linewidth=0.25, alpha=0.35)

    ax.text(
        0.01, 0.01, subtitle_left,
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8,
        bbox=dict(facecolor=(1, 1, 1, 0.65), edgecolor=(0, 0, 0, 0.15), boxstyle="round,pad=0.3")
    )
    ax.text(
        0.99, 0.01, subtitle_right,
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8,
        bbox=dict(facecolor=(1, 1, 1, 0.65), edgecolor=(0, 0, 0, 0.15), boxstyle="round,pad=0.3")
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


def main():
    greece = load_greece()
    run_ts = datetime.now(ATHENS_TZ).strftime("%Y-%m-%d %H:%M %Z")

    # Cloud map
    cloud_img, cloud_time, cloud_headers = fetch_cloud_image()
    cloud_out = os.path.join(OUTPUT_DIR, "cloudiness_latest.png")
    save_map(
        cloud_img,
        greece,
        title="Δοκιμή νέφωσης Ελλάδας (δορυφορικό layer)",
        subtitle_left=f"Δημιουργήθηκε: {run_ts}\nΠηγή: EUMETView WMS\nLayer: {CLOUD_WMS_LAYER}",
        subtitle_right=f"Χρόνος layer: {cloud_time or 'latest/default'}",
        out_path=cloud_out,
    )

    # Rain map
    rain_img, rain_meta = fetch_rain_mosaic(MAP_EXTENT, zoom=RAIN_ZOOM, tile_size=RAIN_TILE_SIZE)
    rain_out = os.path.join(OUTPUT_DIR, "rain_latest.png")
    save_map(
        rain_img,
        greece,
        title="Δοκιμή βροχής Ελλάδας (radar mosaic layer)",
        subtitle_left=(
            f"Δημιουργήθηκε: {run_ts}\n"
            f"Πηγή: RainViewer radar mosaic\n"
            f"Zoom: {RAIN_ZOOM}"
        ),
        subtitle_right=(
            f"Frame UTC: {fmt_athens(rain_meta.get('time'))}\n"
            f"API generated: {fmt_athens(rain_meta.get('generated'))}"
        ),
        out_path=rain_out,
    )

    metadata = {
        "run_generated_athens": run_ts,
        "bbox": {
            "lon_min": LON_MIN,
            "lat_min": LAT_MIN,
            "lon_max": LON_MAX,
            "lat_max": LAT_MAX,
        },
        "cloud": {
            "source": "EUMETView WMS",
            "wms_url": CLOUD_WMS_URL,
            "layer": CLOUD_WMS_LAYER,
            "time_used": cloud_time,
            "response_date_header": cloud_headers.get("Date"),
            "output": os.path.relpath(cloud_out, BASE_DIR),
        },
        "rain": {
            "source": "RainViewer Weather Maps API",
            "api_url": RAINVIEWER_API,
            "frame_path": rain_meta.get("path"),
            "frame_time_utc_unix": rain_meta.get("time"),
            "api_generated_utc_unix": rain_meta.get("generated"),
            "zoom": RAIN_ZOOM,
            "tile_size": RAIN_TILE_SIZE,
            "color_scheme": RAIN_COLOR_SCHEME,
            "smooth": RAIN_SMOOTH,
            "snow": RAIN_SNOW,
            "output": os.path.relpath(rain_out, BASE_DIR),
        },
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved: {meta_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
        raise
