#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tnow.py
#
# Produces 4 files (Attica + Greece), ALL in ONE folder: ./Tnowmaps/
#   1) tnow_attica.png
#   2) tnow_attica_YYYYMMDD_HHMM.png
#   3) tnow.png
#   4) tnow_YYYYMMDD_HHMM.png
#
# Attica map is done like
# - Grid + interpolation + distance masks in EGSA87 (EPSG:2100 meters)
# - Greece boundary reprojected + clipped in EPSG:2100
# - Plot in EPSG:2100, but axis ticks formatted as lon/lat degrees
#
# Greece map is left in WGS84 degrees (as in your previous tnow script).
#
# Temperature palette: SAME as today.py (shared Tmin/Tmax palette).
#
# Contours:
# - 0°C thicker
# - Every 3°C thinner
#
#
# Requirements:
#   pip install numpy pandas geopandas matplotlib scipy requests rasterio pyproj

import os
import time
import shutil
import subprocess, zipfile
import builtins
import warnings
from time import perf_counter
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
from common_abbrev import shorten_for_box

import numpy as np
import numpy.ma as ma
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

from scipy.spatial import cKDTree

import requests
import rasterio
from pyproj import Transformer
from ftplib import FTP_TLS

# =========================
# PUBLIC LOG CONTROL
# =========================
# GitHub Actions logs are visible in a public repository.
# Default: keep script output quiet so HTTP details, paths, filenames,
# coverage percentages, body previews, and upload names are not printed.
QUIET_PUBLIC_LOGS = os.environ.get("QUIET_PUBLIC_LOGS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off"
)

_real_print = builtins.print


def print(*args, **kwargs):
    if not QUIET_PUBLIC_LOGS:
        _real_print(*args, **kwargs)


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in divide"
)


# =========================
# CONFIG (no secrets here)
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
DEM_PATH = os.path.join(BASE_DIR, "GRC_alt.vrt")

# All sensitive values are injected via environment variables by the CI runner.
# You created these in GitHub → Settings → Secrets and variables → Actions.
DATA_URL = os.environ.get("PRIVATE_WEATHERNOW_URL", "").strip()
PRIVATE_WEATHERNOW_TOKEN = os.environ.get("PRIVATE_WEATHERNOW_TOKEN", "").strip()

FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()  # empty disables uploads

# ALL outputs here
OUT_DIR = os.path.join(BASE_DIR, "Tnowmaps")

# --- Greece (keep as-is: WGS84 degrees) ---
GR_LON_MIN, GR_LON_MAX = 19.0, 30.0
GR_LAT_MIN, GR_LAT_MAX = 34.5, 42.5
GR_N = 300

# Fixed rasterized Greek land area for the 300 x 300 national grid.
# Calculated once from the same greece.geojson boundary and latitude-weighted
# grid-cell areas. It is used as the denominator for national area percentages.
GREECE_RASTERIZED_LAND_AREA_KM2 = 131595.026512276

# --- Attica bbox: EXACTLY like your rain Attica script ---
AT_LON_MIN, AT_LON_MAX = 22.7, 25.0
AT_LAT_MIN, AT_LAT_MAX = 37.5, 38.7
AT_N = 300

# --- Crete bbox (same EGSA approach as Attica) ---
CR_LON_MIN, CR_LON_MAX = 23.37, 26.4
CR_LAT_MIN, CR_LAT_MAX = 34.7, 35.78
CR_N = 300

# --- NE Greece bbox (EGSA approach like Attica/Crete) ---
NE_LON_MIN, NE_LON_MAX = 22.0, 26.6
NE_LAT_MIN, NE_LAT_MAX = 39.7, 41.8
NE_N = 300

# --- NW Greece bbox (EGSA approach like Attica/Crete/NE/SW Greece) ---
NW_LON_MIN, NW_LON_MAX = 19.36, 21.94
NW_LAT_MIN, NW_LAT_MAX = 38.53, 40.94
NW_N = 300

# --- SW Greece bbox (EGSA approach like Attica/Crete/NE/NW Greece) ---
SW_LON_MIN, SW_LON_MAX = 20.25, 23.78
SW_LAT_MIN, SW_LAT_MAX = 36.0, 38.45
SW_N = 300

# --- Cyprus bbox (UTM 36N approach, same as rainintensityall.py) ---
CY_LON_MIN, CY_LON_MAX = 32.0, 34.9
CY_LAT_MIN, CY_LAT_MAX = 34.4, 35.9
CY_N = 300

CYPRUS_GEOJSON_PATH = os.path.join(BASE_DIR, "cyprus.geojson")
CYPRUS_ALT_TIF_PATH = os.path.join(BASE_DIR, "cyprus_dsm_90m.tif")

# Fetch/retry
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/plain, text/*;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8,el;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "X-EKairos-Token": PRIVATE_WEATHERNOW_TOKEN,
}

MAX_RETRIES = 5
DELAY = 10
TIMEOUT = 20

SENTINEL_TEMP = -67.8

ALT_ENC  = os.path.join(BASE_DIR, "altitude.zip.enc")
ALT_ZIP  = os.path.join(BASE_DIR, "altitude.zip")
ALT_PASS = os.environ.get("GEOJSON_PASS", "").strip()

def ensure_altitude_bundle():
    # If the VRT is already present at repo root, nothing to do
    if os.path.exists(DEM_PATH):
        return

    # Try to decrypt and unzip if the encrypted bundle exists at repo root
    if not os.path.exists(ALT_ENC):
        return
    if not ALT_PASS:
        raise SystemExit("DEM bundle missing and GEOJSON_PASS not set to decrypt altitude.zip.enc")

    # Decrypt altitude.zip.enc → altitude.zip at repo root
    try:
        subprocess.check_call([
            "openssl","enc","-d","-aes-256-cbc","-pbkdf2",
            "-in", ALT_ENC, "-out", ALT_ZIP, "-pass", "pass:" + ALT_PASS
        ])
    except FileNotFoundError:
        raise SystemExit("OpenSSL not found. Install it or decrypt altitude.zip.enc in your CI step.")
    except subprocess.CalledProcessError as e:
        raise SystemExit("OpenSSL decryption failed for altitude.zip.enc: %s" % e)

    # Unzip into repo root so GRC_alt.vrt, .grd, .gri land next to tnow.py
    with zipfile.ZipFile(ALT_ZIP, "r") as zf:
        zf.extractall(BASE_DIR)

    # << RIGHT HERE: verify the VRT exists >>
    if not os.path.exists(DEM_PATH):
        raise SystemExit("Decrypted bundle didn’t contain GRC_alt.vrt at repo root. Check DEM_PATH or the zip contents.")

    # Remove the plaintext zip
    try:
        os.remove(ALT_ZIP)
    except Exception:
        pass



# ---------- TOP-10 BOX formatting (shared via common_abbrev.py) ----------
TOPBOX_NAME_MAX = 26


# Shared palette (same as today.py)
TEMP_VMIN = -25.0
TEMP_VMAX = 45.0

# Attica / Greece regional projection settings
CRS_WGS84 = "EPSG:4326"
CRS_EGSA87 = "EPSG:2100"
CRS_UTM36N = "EPSG:32636"

WGS_TO_EGSA = Transformer.from_crs(CRS_WGS84, CRS_EGSA87, always_xy=True)
EGSA_TO_WGS = Transformer.from_crs(CRS_EGSA87, CRS_WGS84, always_xy=True)

WGS_TO_UTM36N = Transformer.from_crs(CRS_WGS84, CRS_UTM36N, always_xy=True)
UTM36N_TO_WGS = Transformer.from_crs(CRS_UTM36N, CRS_WGS84, always_xy=True)

# IDW / masks in meters for Attica
AT_IDW_K = 8
AT_IDW_POWER = 2
AT_MAX_DISTANCE_M = 120_000
AT_MIN_NEIGHBORS = 3
AT_DISTANCE_MASK_M = 170_000

# IDW / masks in meters for Cyprus
CY_IDW_K = 8
CY_IDW_POWER = 2
CY_MAX_DISTANCE_M = 40_000
CY_MIN_NEIGHBORS = 3
CY_DISTANCE_MASK_M = 40_000

# Cyprus stations sometimes lag behind the latest Greece feed time.
# Use a longer freshness window only for the Cyprus TNow map.
CY_MAX_AGE_MINUTES = 180

# Lapse-rate estimation in Attica (meters)
LAPSE_DEFAULT = -0.0065
LAPSE_MIN = -0.0150
LAPSE_MAX = 0.0050
LAPSE_K = 25
LAPSE_RADIUS_M = 150_000
LAPSE_MIN_NBR = 8
LAPSE_ALT_RANGE_MIN_M = 200


# =========================
# SHARED TEMP PALETTE
# =========================
def build_shared_temp_cmap_norm():
    anchors = [
        (-25.0, "#0b1d5c"),  # deep cold navy
        (-18.0, "#123b8a"),  # dark blue
        (-12.0, "#1f63c6"),  # blue
        (-6.0,  "#2f8fe6"),  # lighter blue
        (-2.0,  "#44b6ff"),  # icy blue
        (0.0,   "#2b7bff"),  # 0°C = BLUE (important!)
        (3.0,   "#2fb8d6"),  # blue-cyan
        (7.0,   "#2fc4a0"),  # cyan-green
        (12.0,  "#34c759"),  # green
        (18.0,  "#b7dd2a"),  # yellow-green
        (24.0,  "#ffe11a"),  # yellow
        (30.0,  "#ff9a1a"),  # orange
        (35.0,  "#ff4d1a"),  # red-orange
        (40.0,  "#d1166f"),  # hot magenta
        (45.0,  "#6a00a8"),  # purple (extreme heat)
    ]
    vals = np.array([v for v, _ in anchors], dtype=float)
    cols = [c for _, c in anchors]
    t = (vals - TEMP_VMIN) / (TEMP_VMAX - TEMP_VMIN)
    t = np.clip(t, 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("t_shared", list(zip(t, cols)), N=256)
    norm = Normalize(vmin=TEMP_VMIN, vmax=TEMP_VMAX, clip=True)
    return cmap, norm

TEMP_CMAP, TEMP_NORM = build_shared_temp_cmap_norm()


# =========================
# FTP (FTPS)
# =========================
def upload_to_ftp(local_file: str) -> None:
    # Upload only if all credentials are present
    if not (FTP_HOST and FTP_USER and FTP_PASS):
        return

    remote_name = os.path.basename(local_file)
    ftps = FTP_TLS()
    ftps.connect(FTP_HOST, 21, timeout=30)
    ftps.login(user=FTP_USER, passwd=FTP_PASS)
    ftps.prot_p()
    try:
        with open(local_file, "rb") as f:
            ftps.storbinary("STOR " + remote_name, f)
        print(f"📤 Uploaded: {remote_name}")
    finally:
        try:
            ftps.quit()
        except Exception:
            pass



# =========================
# IO / PARSING
# =========================
def fetch_text(url: str) -> str:
    if not url:
        raise SystemExit("PRIVATE_WEATHERNOW_URL is not set.")

    if not PRIVATE_WEATHERNOW_TOKEN:
        raise SystemExit("PRIVATE_WEATHERNOW_TOKEN is not set.")

    last_exc = None

    def _looks_like_tsv(payload: str) -> bool:
        if not payload:
            return False
        head = payload.lstrip().lower()
        if head.startswith("<!doctype") or head.startswith("<html") or "<html" in head[:500]:
            return False
        lines = [ln for ln in payload.splitlines() if ln.strip()][:15]
        if not lines:
            return False
        joined = "\n".join(lines).lower()
        if "datetime" not in joined:
            return False
        if not any("\t" in ln for ln in lines):
            return False
        return True

    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            ct = r.headers.get("Content-Type", "")
            print("ℹ️ HTTP:", r.status_code, "| Content-Type:", ct)

            if r.status_code >= 400:
                raise requests.exceptions.RequestException(
                    "HTTP error while fetching protected weather feed: {}".format(r.status_code)
                )

            r.raise_for_status()

            raw = r.content  # bytes
            text = None

            # 1) Correct for your feed: UTF-8 (with BOM safety)
            for enc in ("utf-8-sig", "utf-8"):
                try:
                    text = raw.decode(enc)
                    break
                except Exception:
                    pass

            # 2) Fallbacks (Greek legacy encodings)
            if text is None:
                for enc in ("cp1253", "iso-8859-7", "latin-1"):
                    try:
                        text = raw.decode(enc)
                        break
                    except Exception:
                        pass

            if text is None:
                # last resort: don't crash, but you'll see replacement chars
                text = raw.decode("utf-8", errors="replace")
                
            if "Î" in text or "Ã" in text:
                print("⚠️ Suspected mojibake in decoded text (check encoding/headers).")

            if not _looks_like_tsv(text):
                raise requests.exceptions.RequestException(
                    "Response did not look like tab-delimited weather data."
                )

            return text

        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"🌧️ Attempt {i+1} failed: {e}")
            time.sleep(DELAY)

    raise SystemExit(last_exc)


def read_tabbed_df(text: str) -> pd.DataFrame:
    # First try: fast C engine
    try:
        df = pd.read_csv(StringIO(text), sep="\t")
    except Exception:
        # Fallback: python engine + skip malformed lines so one broken row cannot kill the run
        df = pd.read_csv(
            StringIO(text),
            sep="\t",
            engine="python",
            on_bad_lines="skip"
        )

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )
    for c in list(df.columns):
        if c.lower() == "datetime" and c != "Datetime":
            df.rename(columns={c: "Datetime"}, inplace=True)
    return df

# =========================
# GENERIC IDW (works for meters or degrees)
# =========================
def idw_fast(x, y, z, xi, yi, k=8, power=2, max_distance=1.0, min_neighbors=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(z)

    tree = cKDTree(np.c_[x, y])
    dist, idx = tree.query(
        np.c_[xi.ravel(), yi.ravel()],
        k=min(k, n),
        distance_upper_bound=max_distance
    )

    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    finite = np.isfinite(dist) & (idx < n)
    neigh_count = np.sum(finite, axis=1)

    zi = np.full(xi.size, np.nan, dtype=float)
    ok_pts = neigh_count >= min_neighbors
    if not np.any(ok_pts):
        return zi.reshape(xi.shape)

    idx_safe = np.where(finite, idx, 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(dist == 0, 1e12, 1.0 / (dist ** power))
        w = np.where(np.isfinite(w) & finite, w, 0.0)

    z_nei = z[idx_safe]
    num = np.sum(w * z_nei, axis=1)
    den = np.sum(w, axis=1)

    zi_ok = np.divide(
        num,
        den,
        out=np.full_like(num, np.nan, dtype=float),
        where=den > 0
    )
    zi[ok_pts] = zi_ok[ok_pts]
    return zi.reshape(xi.shape)

# =========================
# DEM SAMPLING (lon/lat arrays)
# =========================
def sample_dem_lonlat(dem_path: str, lons, lats) -> np.ndarray:
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found at: {dem_path}")

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        samples = list(src.sample(zip(lons.tolist(), lats.tolist())))
        elev = np.array([s[0] for s in samples], dtype=float)

        if nodata is not None:
            elev = np.where(elev == nodata, np.nan, elev)
        elev = np.where(elev < -100, np.nan, elev)
        elev = np.where(np.isfinite(elev), elev, 0.0)

    return elev

def sample_raster_xy(raster_path: str, xs, ys, input_crs: str) -> np.ndarray:
    """
    Sample a raster using coordinates in input_crs.
    Reprojects sample coordinates to the raster CRS if needed.
    """
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster not found at: {raster_path}")

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise RuntimeError("Raster has no CRS defined.")

        if str(src.crs) != str(input_crs):
            from rasterio.warp import transform as rio_transform
            xs2, ys2 = rio_transform(str(input_crs), src.crs, xs.tolist(), ys.tolist())
        else:
            xs2, ys2 = xs.tolist(), ys.tolist()

        samples = list(src.sample(zip(xs2, ys2)))
        arr = np.array(samples, dtype=float).reshape(-1)

        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        arr = np.where(arr < -100, np.nan, arr)
        arr = np.where(np.isfinite(arr), arr, 0.0)

    return arr

# =========================
# MASKS / CONTOURS / STAMP
# =========================
def stamp_text(athens_now: datetime) -> str:
    ts = athens_now.strftime("%Y-%m-%d %H:%M %Z")
    return "Δημιουργήθηκε για το e-kairos.gr\n" + ts


def fmt_decimal_comma(value: float, decimals: int = 1) -> str:
    """Format a number for map text using a comma as the decimal separator."""
    return f"{float(value):.{decimals}f}".replace(".", ",")


def pick_station_label_column(df: pd.DataFrame) -> str:
    """
    Picks the best column to display for station name/area.
    Preference: citygr -> CityGR -> station -> name -> webcode.
    """
    for c in ["citygr", "Citygr", "CityGR", "station", "name", "webcode"]:
        if c in df.columns:
            return c
    return "webcode"


def add_top10_box_greece(ax, tt0: pd.DataFrame, frost_text: str = "") -> None:
    """
    Transparent top-right info + map markers (1..10).
    - Cold 10: blue numbers
    - Hot 10: red numbers
    """
    if tt0 is None or tt0.empty:
        return
    if "TNow" not in tt0.columns:
        return

    label_col = pick_station_label_column(tt0)

    tmp = tt0.copy()
    tmp["TNow"] = pd.to_numeric(tmp["TNow"], errors="coerce")
    tmp = tmp.dropna(subset=["TNow", "Latitude", "Longitude"])
    if tmp.empty:
        return

    if label_col not in tmp.columns:
        tmp[label_col] = "station"

    cold10 = tmp.nsmallest(10, "TNow").copy()
    hot10  = tmp.nlargest(10, "TNow").copy()

    # ---- build text block (shortened names with today.py abbreviations)
    def fmt_block(dfx: pd.DataFrame, title: str) -> str:
        lines = [title]
        i = 1
        for _, r in dfx.iterrows():
            try:
                t = float(r["TNow"])
            except Exception:
                continue
            name = shorten_for_box(str(r.get(label_col, "–")), max_chars=TOPBOX_NAME_MAX)
            lines.append(f"{i}. {name}: {fmt_decimal_comma(t, 1)}°C")
            i += 1
        return "\n".join(lines)

    box_text = (
        fmt_block(cold10, "Ψυχρότερες 10 περιοχές")
    )

    # ---- add frost line (only when provided)
    if frost_text:
        box_text = box_text + "\n\n" + frost_text

    # Transparent box (no background), keep readability with white stroke
    ax.text(
        0.99, 0.99, box_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.2,
        color="black",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.25"),
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
    )

    # ---- draw map markers 1..10 for cold/hot (with white outline)
    def draw_rank_markers(dfx: pd.DataFrame, color: str):
        rank = 1
        for _, r in dfx.iterrows():
            try:
                lon = float(r["Longitude"])
                lat = float(r["Latitude"])
            except Exception:
                continue

            # subtle ring so the point is visible
            ax.scatter([lon], [lat], s=70, facecolors="none", edgecolors=color,
                       linewidths=1.2, zorder=12)

            txt = ax.text(
                lon, lat, str(rank),
                ha="center", va="center",
                fontsize=8, fontweight="normal",
                color=color, zorder=13
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
            rank += 1
            if rank > 10:
                break

    ### draw_rank_markers(cold10, color="#1d4ed8")  # blue-ish ### δείχνει τα τοπ 10 ψυχρότερα πάνω στον χάρτη
    ### draw_rank_markers(hot10,  color="#dc2626")  # red-ish  ### δείχνει τα τοπ 10 θερμότερα πάνω στον χάρτη

def add_top5_box_cyprus(ax, tt0: pd.DataFrame) -> None:
    """
    Transparent top-right Cyprus info box.
    - Cold 5
    - Hot 5
    Uses citygr where available, like the Greece TNow box.
    """
    if tt0 is None or tt0.empty:
        return
    if "TNow" not in tt0.columns:
        return

    label_col = "citygr" if "citygr" in tt0.columns else pick_station_label_column(tt0)

    tmp = tt0.copy()
    tmp["TNow"] = pd.to_numeric(tmp["TNow"], errors="coerce")
    tmp = tmp.dropna(subset=["TNow", "Latitude", "Longitude"])

    if tmp.empty:
        return

    if label_col not in tmp.columns:
        tmp[label_col] = "station"

    cold5 = tmp.nsmallest(5, "TNow").copy()
    hot5 = tmp.nlargest(5, "TNow").copy()

    def fmt_block(dfx: pd.DataFrame, title: str) -> str:
        lines = [title]
        i = 1
        for _, r in dfx.iterrows():
            try:
                t = float(r["TNow"])
            except Exception:
                continue

            name = shorten_for_box(str(r.get(label_col, "–")), max_chars=TOPBOX_NAME_MAX)
            lines.append(f"{i}. {name}: {fmt_decimal_comma(t, 1)}°C")
            i += 1

        return "\n".join(lines)

    box_text = (
        fmt_block(cold5, "Ψυχρότερες 5 περιοχές")
        + "\n\n"
        + fmt_block(hot5, "Θερμότερες 5 περιοχές")
    )

    ax.text(
        0.99, 0.01, box_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="black",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.25"),
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        zorder=30
    )

def add_contours(ax, X, Y, field):
    """
    Draw ordinary contours every 3°C without labels.
    Draw selected prominent contours more strongly and label only those.
    """
    levels = np.arange(-30, 46, 3, dtype=float)
    special_levels = [0.0, 10.0, 20.0, 30.0, 37.0, 40.0]

    thin_levels = [
        lv for lv in levels
        if not any(np.isclose(lv, special) for special in special_levels)
    ]

    try:
        ax.contour(
            X, Y, field,
            levels=thin_levels,
            colors="black",
            linewidths=0.6,
            alpha=0.70
        )
    except Exception:
        pass

    cs_special = None

    try:
        cs_special = ax.contour(
            X, Y, field,
            levels=special_levels,
            colors="black",
            linewidths=1.3,
            alpha=0.95
        )
    except Exception:
        cs_special = None

    if cs_special is not None:
        try:
            texts_special = ax.clabel(
                cs_special,
                levels=cs_special.levels[:],
                inline=True,
                inline_spacing=2,
                fmt="%d",
                fontsize=5
            )

            for t in texts_special:
                t.set_rotation(0)
                t.set_rotation_mode("anchor")
                t.set_path_effects([
                    pe.withStroke(linewidth=1.8, foreground="white")
                ])
        except Exception:
            pass

def add_contours_attica(ax, X, Y, field):
    """
    Attica-only contours:
    - Draw contours every 3°C
    - Label all 3°C contours
    - Keep selected prominent contours thicker
    """
    levels = np.arange(-30, 46, 3, dtype=float)
    special_levels = [0.0, 10.0, 20.0, 30.0, 37.0, 40.0]

    thin_levels = [
        lv for lv in levels
        if not any(np.isclose(lv, special) for special in special_levels)
    ]

    cs_thin = None
    cs_special = None

    try:
        cs_thin = ax.contour(
            X, Y, field,
            levels=thin_levels,
            colors="black",
            linewidths=0.6,
            alpha=0.70
        )
    except Exception:
        cs_thin = None

    try:
        cs_special = ax.contour(
            X, Y, field,
            levels=special_levels,
            colors="black",
            linewidths=1.3,
            alpha=0.95
        )
    except Exception:
        cs_special = None

    # Label the ordinary 3°C contours too
    if cs_thin is not None:
        try:
            texts_thin = ax.clabel(
                cs_thin,
                levels=cs_thin.levels[:],
                inline=True,
                inline_spacing=2,
                fmt="%d",
                fontsize=4.5
            )

            for t in texts_thin:
                t.set_rotation(0)
                t.set_rotation_mode("anchor")
                t.set_path_effects([
                    pe.withStroke(linewidth=1.6, foreground="white")
                ])
        except Exception:
            pass

    # Label the prominent contours slightly more clearly
    if cs_special is not None:
        try:
            texts_special = ax.clabel(
                cs_special,
                levels=cs_special.levels[:],
                inline=True,
                inline_spacing=2,
                fmt="%d",
                fontsize=5
            )

            for t in texts_special:
                t.set_rotation(0)
                t.set_rotation_mode("anchor")
                t.set_path_effects([
                    pe.withStroke(linewidth=1.8, foreground="white")
                ])
        except Exception:
            pass

def bounds_reasonable_cyprus(geom, lon_min=31.0, lon_max=36.0, lat_min=34.0, lat_max=36.5):
    try:
        minx, miny, maxx, maxy = geom.bounds
        return (
            (lon_min <= minx <= lon_max) and
            (lon_min <= maxx <= lon_max) and
            (lat_min <= miny <= lat_max) and
            (lat_min <= maxy <= lat_max)
        )
    except Exception:
        return False


def swap_geom_xy(geom):
    from shapely.geometry import Polygon, MultiPolygon

    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        return Polygon(np.column_stack([y, x]))

    if isinstance(geom, MultiPolygon):
        return MultiPolygon([swap_geom_xy(g) for g in geom.geoms])

    return geom


def fit_lapse_rate_simple(st_temp, st_elev):
    st_temp = np.asarray(st_temp, dtype=float)
    st_elev = np.asarray(st_elev, dtype=float)

    ok = np.isfinite(st_temp) & np.isfinite(st_elev)

    if np.sum(ok) < 8:
        return LAPSE_DEFAULT

    try:
        b, _a = np.polyfit(st_elev[ok], st_temp[ok], 1)
        b = float(np.clip(b, LAPSE_MIN, LAPSE_MAX))
        return b
    except Exception:
        return LAPSE_DEFAULT

def _temp_colorbar(ax, img):
    ticks = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    cbar = plt.colorbar(img, ax=ax, orientation="vertical", extend="both")
    cbar.set_ticks(ticks)
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)
    return cbar


def save_with_timestamp(fig, out_dir: str, out_name: str, athens_now: datetime):
    os.makedirs(out_dir, exist_ok=True)

    main_path = os.path.join(out_dir, out_name)
    fig.savefig(main_path, dpi=300, bbox_inches="tight")
    ts = athens_now.strftime("%Y%m%d_%H%M")
    root, ext = os.path.splitext(out_name)
    ts_path = os.path.join(out_dir, f"{root}_{ts}{ext}")
    try:
        shutil.copy2(main_path, ts_path)
    except Exception as e:
        print(f"⚠️ Could not create timestamped copy: {e}")
        ts_path = None

    return main_path, ts_path


# =========================
# GREECE MAP (KEEP WGS84 STYLE)
# =========================
def build_geo_mask_wgs(grid_x, grid_y, greece_gdf_wgs) -> np.ndarray:
    if hasattr(greece_gdf_wgs.geometry, "union_all"):
        boundary = greece_gdf_wgs.geometry.union_all()
    else:
        boundary = greece_gdf_wgs.unary_union

    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()),
        crs=greece_gdf_wgs.crs
    )
    return pts.geometry.within(boundary).values.reshape(grid_x.shape)


def build_distance_mask(xgrid, ygrid, xs, ys, max_dist):
    tree = cKDTree(np.c_[xs, ys])
    d, _ = tree.query(np.c_[xgrid.ravel(), ygrid.ravel()])
    return (d.reshape(xgrid.shape) <= max_dist)


def estimate_local_lapse_rates_wgs(st_lons, st_lats, st_temp, st_elev,
                                  k=12, max_deg=1.2,
                                  default_lapse=LAPSE_DEFAULT,
                                  clip_min=LAPSE_MIN, clip_max=LAPSE_MAX) -> np.ndarray:
    tree = cKDTree(np.c_[st_lons, st_lats])
    d, idx = tree.query(np.c_[st_lons, st_lats], k=min(k, len(st_temp)), distance_upper_bound=max_deg)

    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]

    lapses = np.full(len(st_temp), np.nan, dtype=float)

    for i in range(len(st_temp)):
        neigh = idx[i]
        dist = d[i]
        ok = np.isfinite(dist) & (neigh < len(st_temp))
        neigh = neigh[ok]

        if neigh.size < 4:
            lapses[i] = default_lapse
            continue

        elev_n = st_elev[neigh]
        t_n = st_temp[neigh]
        good = np.isfinite(elev_n) & np.isfinite(t_n)

        elev_n = elev_n[good]
        t_n = t_n[good]

        if elev_n.size < 4 or float(np.nanstd(elev_n)) < 50:
            lapses[i] = default_lapse
            continue

        try:
            b, _a = np.polyfit(elev_n, t_n, 1)
            b = float(np.clip(b, clip_min, clip_max))
            lapses[i] = b
        except Exception:
            lapses[i] = default_lapse

    return lapses


def make_tnow_greece_wgs(df, greece_gdf_wgs, dem_path, athens_now):
    if "TNow" not in df.columns:
        print("❌ TNow missing.")
        return (None, None)

    tt0 = df.copy()
    tt0["TNow"] = pd.to_numeric(tt0["TNow"], errors="coerce")
    tt0.dropna(subset=["TNow", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TNow"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]

    # Greece national map: use Greece-only stations for interpolation,
    # top-10 boxes, observed extrema, and observed threshold/frost checks.
    if "Country" in tt0.columns:
        tt0 = tt0[
            tt0["Country"].astype(str).str.strip().str.lower() == "greece"
        ].copy()

    if tt0.empty:
        print("❌ No valid TNow data for Greece.")
        return (None, None)
    
    grid_x, grid_y = np.meshgrid(
        np.linspace(GR_LON_MIN, GR_LON_MAX, GR_N),
        np.linspace(GR_LAT_MIN, GR_LAT_MAX, GR_N)
    )

    geo_mask = build_geo_mask_wgs(grid_x, grid_y, greece_gdf_wgs)

    st_lons = tt0["Longitude"].to_numpy(dtype=float)
    st_lats = tt0["Latitude"].to_numpy(dtype=float)
    st_t = tt0["TNow"].to_numpy(dtype=float)

    st_elev = sample_dem_lonlat(dem_path, st_lons, st_lats)

    ok = np.isfinite(st_t) & np.isfinite(st_lons) & np.isfinite(st_lats) & np.isfinite(st_elev)
    st_lons = st_lons[ok]
    st_lats = st_lats[ok]
    st_t = st_t[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 5:
        print("❌ Too few stations for Greece interpolation.")
        return (None, None)

    # Estimate a local lapse rate at each station, reduce each observation to a
    # sea-level-equivalent temperature, interpolate, and adjust back to the
    # DEM elevation of each national-grid cell.
    st_lapse = estimate_local_lapse_rates_wgs(st_lons, st_lats, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    t0_grid = idw_fast(
        st_lons, st_lats, st_t0, grid_x, grid_y,
        k=8, power=2, max_distance=1.2, min_neighbors=3
    )
    lapse_grid = idw_fast(
        st_lons, st_lats, st_lapse, grid_x, grid_y,
        k=8, power=2, max_distance=1.2, min_neighbors=3
    )

    grid_elev = sample_dem_lonlat(
        dem_path,
        grid_x.ravel(),
        grid_y.ravel()
    ).reshape(grid_x.shape)

    t_grid = t0_grid + (lapse_grid * grid_elev)

    # Display only Greek land cells with a reporting station within 0.8 degrees.
    dist_mask = build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_dist=0.8)
    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    # The national map now uses this single ordinary altitude-adjusted IDW field.
    out = np.full(grid_x.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    # =========================
    # MIN/MAX FROM THE DISPLAYED FIELD ONLY
    # =========================
    interp_min = None
    interp_max = None

    try:
        if np.any(np.isfinite(out)):
            interp_min = float(np.nanmin(out))
            interp_max = float(np.nanmax(out))
    except Exception:
        interp_min, interp_max = None, None

    # =========================
    # LATITUDE-WEIGHTED LAND-AREA STATISTICS FROM out ONLY
    # =========================
    earth_radius_km = 6371.0088
    dlon_rad = np.deg2rad((GR_LON_MAX - GR_LON_MIN) / max(GR_N - 1, 1))
    dlat_rad = np.deg2rad((GR_LAT_MAX - GR_LAT_MIN) / max(GR_N - 1, 1))

    cell_area_km2 = (
        (earth_radius_km ** 2)
        * dlon_rad
        * dlat_rad
        * np.cos(np.deg2rad(grid_y))
    )

    mapped_mask = final_mask & np.isfinite(out)
    mapped_area_km2 = float(np.sum(cell_area_km2[mapped_mask]))
    coverage_pct = 100.0 * mapped_area_km2 / GREECE_RASTERIZED_LAND_AREA_KM2

    def pct_area_above(threshold_c: float) -> float:
        threshold_mask = mapped_mask & (out > threshold_c)
        area_km2 = float(np.sum(cell_area_km2[threshold_mask]))
        return 100.0 * area_km2 / GREECE_RASTERIZED_LAND_AREA_KM2

    pct_above_30 = pct_area_above(30.0)
    pct_above_37 = pct_area_above(37.0)
    pct_above_40 = pct_area_above(40.0)

    print(f"ℹ️ National interpolation coverage: {mapped_area_km2:,.0f} km² ({coverage_pct:.1f}% of Greece)")
    print(f"ℹ️ Area >30°C: {pct_above_30:.1f}% of Greece")
    print(f"ℹ️ Area >37°C: {pct_above_37:.1f}% of Greece")
    print(f"ℹ️ Area >40°C: {pct_above_40:.1f}% of Greece")

    def format_pct_with_observed_floor(pct_value: float, threshold_c: float) -> str:
        """
        Show <0,1% when the interpolated percentage is below 0.1%,
        but the threshold has genuinely been crossed either by the
        interpolated field or by at least one valid station observation.
        """
        observed_exceedance = bool((tt0["TNow"] > threshold_c).any())

        if pct_value < 0.1 and (pct_value > 0.0 or observed_exceedance):
            return "<0,1%"

        return fmt_decimal_comma(pct_value, 1) + "%"

    pct_above_30_text = format_pct_with_observed_floor(pct_above_30, 30.0)
    pct_above_37_text = format_pct_with_observed_floor(pct_above_37, 37.0)
    pct_above_40_text = format_pct_with_observed_floor(pct_above_40, 40.0)

    # =========================
    # FROST % FROM out ONLY, WITH OBSERVED-STATION DISPLAY FLOOR
    # =========================
    frost_text = ""
    try:
        frost_mask = mapped_mask & (out <= 0.0)
        frost_area_km2 = float(np.sum(cell_area_km2[frost_mask]))
        frost_pct = 100.0 * frost_area_km2 / GREECE_RASTERIZED_LAND_AREA_KM2
        observed_frost = bool((tt0["TNow"] <= 0.0).any())

        if frost_pct < 0.1 and (frost_pct > 0.0 or observed_frost):
            frost_pct_text = "<0,1%"
        else:
            frost_pct_text = fmt_decimal_comma(frost_pct, 1) + "%"

        if frost_pct > 0.0 or observed_frost:
            frost_text = f"{frost_pct_text} της επικράτειας\nμε παγετό αέρα"
    except Exception:
        frost_text = ""

    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(GR_LON_MIN, GR_LON_MAX, GR_LAT_MIN, GR_LAT_MAX),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_gdf_wgs.boundary.plot(ax=ax, color="black", linewidth=0.6)
    add_contours(ax, grid_x, grid_y, out)
    _temp_colorbar(ax, img)

    ax.set_title("Τρέχουσα θερμοκρασία (προσαρμογή υψομέτρου)", fontsize=16)

    if interp_min is not None and interp_max is not None:
        display_min = interp_min
        display_max = interp_max

        # Show actual valid station extrema when they extend beyond
        # the interpolated land-grid range. The interpolation itself
        # remains unchanged.
        if not tt0.empty:
            actual_station_min = float(tt0["TNow"].min())
            actual_station_max = float(tt0["TNow"].max())

            if np.isfinite(actual_station_min):
                display_min = min(display_min, actual_station_min)
            if np.isfinite(actual_station_max):
                display_max = max(display_max, actual_station_max)

        mm_text = (
            "Εύρος θερμοκρασιών στην ξηρά:\n"
            f"{fmt_decimal_comma(display_min, 1)} έως {fmt_decimal_comma(display_max, 1)}°C\n\n"
            "Ποσοστό έκτασης επικράτειας βάσει παρεμβολής:\n"
            f">30°C: {pct_above_30_text}\n"
            f">37°C: {pct_above_37_text}\n"
            f">40°C: {pct_above_40_text}"
        )
        ax.text(
            0.01, 0.985, mm_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8.2,
            color="black",
            bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.2"),
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")]
        )

    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)

    add_top10_box_greece(ax, tt0, frost_text=frost_text)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=10, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")]
    )

    main_path, ts_path = save_with_timestamp(fig, OUT_DIR, "tnow.png", athens_now)
    plt.close(fig)

    print("✅ Saved:", main_path)
    if ts_path:
        print("✅ Saved:", ts_path)

    return main_path, ts_path



# =========================
# ATTICA MAP (EGSA2100 STYLE)
# =========================
def estimate_local_lapse_rates_egsa(st_x, st_y, st_temp, st_elev,
                                   k=LAPSE_K, radius_m=LAPSE_RADIUS_M,
                                   default_lapse=LAPSE_DEFAULT,
                                   clip_min=LAPSE_MIN, clip_max=LAPSE_MAX) -> np.ndarray:
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)
    st_temp = np.asarray(st_temp, dtype=float)
    st_elev = np.asarray(st_elev, dtype=float)

    tree = cKDTree(np.c_[st_x, st_y])
    d, idx = tree.query(np.c_[st_x, st_y], k=min(k, len(st_temp)), distance_upper_bound=radius_m)

    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]

    lapses = np.full(len(st_temp), np.nan, dtype=float)

    for i in range(len(st_temp)):
        neigh = idx[i]
        dist = d[i]
        ok = np.isfinite(dist) & (neigh < len(st_temp))
        neigh = neigh[ok]
        if neigh.size < LAPSE_MIN_NBR:
            lapses[i] = default_lapse
            continue

        elev_n = st_elev[neigh]
        t_n = st_temp[neigh]
        good = np.isfinite(elev_n) & np.isfinite(t_n)
        elev_n = elev_n[good]
        t_n = t_n[good]

        if elev_n.size < LAPSE_MIN_NBR or (np.nanmax(elev_n) - np.nanmin(elev_n)) < LAPSE_ALT_RANGE_MIN_M:
            lapses[i] = default_lapse
            continue

        try:
            b, _a = np.polyfit(elev_n, t_n, 1)
            b = float(np.clip(b, clip_min, clip_max))
            lapses[i] = b
        except Exception:
            lapses[i] = default_lapse

    return lapses


def make_tnow_attica_egsa(df, greece_gdf_wgs, dem_path, athens_now):
    if "TNow" not in df.columns:
        print("❌ TNow missing.")
        return (None, None)

    tt0 = df.copy()
    tt0["TNow"] = pd.to_numeric(tt0["TNow"], errors="coerce")
    tt0.dropna(subset=["TNow", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TNow"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]
    if tt0.empty:
        print("❌ No valid TNow data for Attica.")
        return (None, None)

    # Convert bbox corners to EGSA meters
    corners_lon = [AT_LON_MIN, AT_LON_MIN, AT_LON_MAX, AT_LON_MAX]
    corners_lat = [AT_LAT_MIN, AT_LAT_MAX, AT_LAT_MIN, AT_LAT_MAX]
    cx, cy = WGS_TO_EGSA.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(cx)), float(np.max(cx))
    y_min, y_max = float(np.min(cy)), float(np.max(cy))

    # Stations projected to EGSA
    st_lon = tt0["Longitude"].to_numpy(dtype=float)
    st_lat = tt0["Latitude"].to_numpy(dtype=float)
    st_t = tt0["TNow"].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    # Prefilter to nearby stations (buffer 200 km)
    buf = 200_000.0
    near = (st_x >= (x_min - buf)) & (st_x <= (x_max + buf)) & (st_y >= (y_min - buf)) & (st_y <= (y_max + buf))
    st_lon = st_lon[near]
    st_lat = st_lat[near]
    st_t = st_t[near]
    st_x = st_x[near]
    st_y = st_y[near]

    if len(st_t) < 8:
        print("❌ Too few nearby stations for Attica interpolation.")
        return (None, None)

    # DEM altitude at stations (lon/lat sampling)
    st_elev = sample_dem_lonlat(dem_path, st_lon, st_lat)

    ok = np.isfinite(st_t) & np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_elev)
    st_t = st_t[ok]
    st_x = st_x[ok]
    st_y = st_y[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 8:
        print("❌ Too few valid stations (after DEM) for Attica interpolation.")
        return (None, None)

    # Lapse per station (in meters space)
    st_lapse = estimate_local_lapse_rates_egsa(st_x, st_y, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    # Grid in EGSA meters
    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, AT_N),
        np.linspace(y_min, y_max, AT_N)
    )

    # Interpolate t0 and lapse in meters
    t0_grid = idw_fast(st_x, st_y, st_t0, grid_x_m, grid_y_m,
                       k=AT_IDW_K, power=AT_IDW_POWER,
                       max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS)

    lapse_grid = idw_fast(st_x, st_y, st_lapse, grid_x_m, grid_y_m,
                          k=AT_IDW_K, power=AT_IDW_POWER,
                          max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS)

    # DEM on grid: EGSA -> lon/lat -> sample
    glon, glat = EGSA_TO_WGS.transform(grid_x_m.ravel().tolist(), grid_y_m.ravel().tolist())
    grid_elev = sample_dem_lonlat(dem_path, np.array(glon, dtype=float), np.array(glat, dtype=float)).reshape(grid_x_m.shape)

    t_grid = t0_grid + (lapse_grid * grid_elev)

    # Greece boundary in EGSA and clipped to bbox
    greece_egsa = greece_gdf_wgs.to_crs(CRS_EGSA87)
    greece_clip = greece_egsa.cx[x_min:x_max, y_min:y_max].copy()

    if hasattr(greece_clip.geometry, "union_all"):
        boundary = greece_clip.geometry.union_all()
    else:
        boundary = greece_clip.geometry.unary_union

    # Geo mask on grid (EGSA)
    grid_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_EGSA87
    )
    geo_mask = grid_pts.geometry.within(boundary).values.reshape(grid_x_m.shape)

    # Distance mask in meters
    tree = cKDTree(np.c_[st_x, st_y])
    d, _ = tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    dist_mask = (d.reshape(grid_x_m.shape) <= AT_DISTANCE_MASK_M)

    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x_m.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    # Plot in EGSA meters, ticks shown as lon/lat degrees
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = EGSA_TO_WGS.transform(x, y_ref_for_lon)
        return fmt_decimal_comma(lon, 2)

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return fmt_decimal_comma(lat, 2)

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours_attica(ax, grid_x_m, grid_y_m, out)

    # slimmer colorbar so the map stays large (Attica figure is square)
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", extend="both",
                        fraction=0.035, pad=0.02)
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)

    ax.set_title("Τρέχουσα θερμοκρασία Αττικής (προσαρμογή υψομέτρου)", fontsize=16, pad=10)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
    )


    main_path, ts_path = save_with_timestamp(fig, OUT_DIR, "tnow_attica.png", athens_now)
    plt.close(fig)

    print("✅ Saved:", main_path)
    if ts_path:
        print("✅ Saved:", ts_path)

    return main_path, ts_path


def make_tnow_crete_egsa(df, greece_gdf_wgs, dem_path, athens_now):
    if "TNow" not in df.columns:
        print("❌ TNow missing.")
        return (None, None)

    tt0 = df.copy()
    tt0["TNow"] = pd.to_numeric(tt0["TNow"], errors="coerce")
    tt0.dropna(subset=["TNow", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TNow"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]
    if tt0.empty:
        print("❌ No valid TNow data for Crete.")
        return (None, None)

    # Convert bbox corners to EGSA meters
    corners_lon = [CR_LON_MIN, CR_LON_MIN, CR_LON_MAX, CR_LON_MAX]
    corners_lat = [CR_LAT_MIN, CR_LAT_MAX, CR_LAT_MIN, CR_LAT_MAX]
    cx, cy = WGS_TO_EGSA.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(cx)), float(np.max(cx))
    y_min, y_max = float(np.min(cy)), float(np.max(cy))

    # Stations projected to EGSA
    st_lon = tt0["Longitude"].to_numpy(dtype=float)
    st_lat = tt0["Latitude"].to_numpy(dtype=float)
    st_t = tt0["TNow"].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    # Prefilter to nearby stations (buffer 200 km)
    buf = 200_000.0
    near = (st_x >= (x_min - buf)) & (st_x <= (x_max + buf)) & (st_y >= (y_min - buf)) & (st_y <= (y_max + buf))
    st_lon = st_lon[near]
    st_lat = st_lat[near]
    st_t = st_t[near]
    st_x = st_x[near]
    st_y = st_y[near]

    if len(st_t) < 8:
        print("❌ Too few nearby stations for Crete interpolation.")
        return (None, None)

    # DEM altitude at stations (lon/lat sampling)
    st_elev = sample_dem_lonlat(dem_path, st_lon, st_lat)

    ok = np.isfinite(st_t) & np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_elev)
    st_t = st_t[ok]
    st_x = st_x[ok]
    st_y = st_y[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 8:
        print("❌ Too few valid stations (after DEM) for Crete interpolation.")
        return (None, None)

    # Lapse per station (in meters space)
    st_lapse = estimate_local_lapse_rates_egsa(st_x, st_y, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    # Grid in EGSA meters
    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, CR_N),
        np.linspace(y_min, y_max, CR_N)
    )

    # Interpolate t0 and lapse in meters
    t0_grid = idw_fast(st_x, st_y, st_t0, grid_x_m, grid_y_m,
                       k=AT_IDW_K, power=AT_IDW_POWER,
                       max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS)

    lapse_grid = idw_fast(st_x, st_y, st_lapse, grid_x_m, grid_y_m,
                          k=AT_IDW_K, power=AT_IDW_POWER,
                          max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS)

    # DEM on grid: EGSA -> lon/lat -> sample
    glon, glat = EGSA_TO_WGS.transform(grid_x_m.ravel().tolist(), grid_y_m.ravel().tolist())
    grid_elev = sample_dem_lonlat(dem_path, np.array(glon, dtype=float), np.array(glat, dtype=float)).reshape(grid_x_m.shape)

    t_grid = t0_grid + (lapse_grid * grid_elev)

    # Greece boundary in EGSA and clipped to bbox
    greece_egsa = greece_gdf_wgs.to_crs(CRS_EGSA87)
    greece_clip = greece_egsa.cx[x_min:x_max, y_min:y_max].copy()

    if hasattr(greece_clip.geometry, "union_all"):
        boundary = greece_clip.geometry.union_all()
    else:
        boundary = greece_clip.geometry.unary_union

    # Geo mask on grid (EGSA)
    grid_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_EGSA87
    )
    geo_mask = grid_pts.geometry.within(boundary).values.reshape(grid_x_m.shape)

    # Distance mask in meters
    tree = cKDTree(np.c_[st_x, st_y])
    d, _ = tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    dist_mask = (d.reshape(grid_x_m.shape) <= AT_DISTANCE_MASK_M)

    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x_m.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    # Plot in EGSA meters, ticks shown as lon/lat degrees
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = EGSA_TO_WGS.transform(x, y_ref_for_lon)
        return fmt_decimal_comma(lon, 2)

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return fmt_decimal_comma(lat, 2)

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours_attica(ax, grid_x_m, grid_y_m, out)
    
    # slimmer colorbar so the map stays large (Attica figure is square)
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", extend="both",
                        fraction=0.035, pad=0.02)
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)

    ax.set_title("Τρέχουσα θερμοκρασία Κρήτης (προσαρμογή υψομέτρου)", fontsize=16, pad=10)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
    )


    main_path, ts_path = save_with_timestamp(fig, OUT_DIR, "tnow_crete.png", athens_now)
    plt.close(fig)

    print("✅ Saved:", main_path)
    if ts_path:
        print("✅ Saved:", ts_path)

    return main_path, ts_path


def make_tnow_negreece_egsa(df, greece_gdf_wgs, dem_path, athens_now):
    if "TNow" not in df.columns:
        print("❌ TNow missing.")
        return (None, None)

    tt0 = df.copy()
    tt0["TNow"] = pd.to_numeric(tt0["TNow"], errors="coerce")
    tt0.dropna(subset=["TNow", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TNow"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]
    if tt0.empty:
        print("❌ No valid TNow data for NE Greece.")
        return (None, None)

    # Convert bbox corners to EGSA meters
    corners_lon = [NE_LON_MIN, NE_LON_MIN, NE_LON_MAX, NE_LON_MAX]
    corners_lat = [NE_LAT_MIN, NE_LAT_MAX, NE_LAT_MIN, NE_LAT_MAX]
    cx, cy = WGS_TO_EGSA.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(cx)), float(np.max(cx))
    y_min, y_max = float(np.min(cy)), float(np.max(cy))

    # Stations projected to EGSA
    st_lon = tt0["Longitude"].to_numpy(dtype=float)
    st_lat = tt0["Latitude"].to_numpy(dtype=float)
    st_t = tt0["TNow"].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    # Prefilter to nearby stations (buffer 200 km)
    buf = 200_000.0
    near = (st_x >= (x_min - buf)) & (st_x <= (x_max + buf)) & (st_y >= (y_min - buf)) & (st_y <= (y_max + buf))
    st_lon = st_lon[near]
    st_lat = st_lat[near]
    st_t = st_t[near]
    st_x = st_x[near]
    st_y = st_y[near]

    if len(st_t) < 8:
        print("❌ Too few nearby stations for NE Greece interpolation.")
        return (None, None)

    # DEM altitude at stations (lon/lat sampling)
    st_elev = sample_dem_lonlat(dem_path, st_lon, st_lat)

    ok = np.isfinite(st_t) & np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_elev)
    st_t = st_t[ok]
    st_x = st_x[ok]
    st_y = st_y[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 8:
        print("❌ Too few valid stations (after DEM) for NE Greece interpolation.")
        return (None, None)

    # Lapse per station (in meters space)
    st_lapse = estimate_local_lapse_rates_egsa(st_x, st_y, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    # Grid in EGSA meters
    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, NE_N),
        np.linspace(y_min, y_max, NE_N)
    )

    # Interpolate t0 and lapse in meters
    t0_grid = idw_fast(st_x, st_y, st_t0, grid_x_m, grid_y_m,
                       k=AT_IDW_K, power=AT_IDW_POWER,
                       max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS)

    lapse_grid = idw_fast(st_x, st_y, st_lapse, grid_x_m, grid_y_m,
                          k=AT_IDW_K, power=AT_IDW_POWER,
                          max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS)

    # DEM on grid: EGSA -> lon/lat -> sample
    glon, glat = EGSA_TO_WGS.transform(grid_x_m.ravel().tolist(), grid_y_m.ravel().tolist())
    grid_elev = sample_dem_lonlat(dem_path, np.array(glon, dtype=float), np.array(glat, dtype=float)).reshape(grid_x_m.shape)

    t_grid = t0_grid + (lapse_grid * grid_elev)

    # Greece boundary in EGSA and clipped to bbox
    greece_egsa = greece_gdf_wgs.to_crs(CRS_EGSA87)
    greece_clip = greece_egsa.cx[x_min:x_max, y_min:y_max].copy()

    if hasattr(greece_clip.geometry, "union_all"):
        boundary = greece_clip.geometry.union_all()
    else:
        boundary = greece_clip.geometry.unary_union

    # Geo mask on grid (EGSA)
    grid_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_EGSA87
    )
    geo_mask = grid_pts.geometry.within(boundary).values.reshape(grid_x_m.shape)

    # Distance mask in meters
    tree = cKDTree(np.c_[st_x, st_y])
    d, _ = tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    dist_mask = (d.reshape(grid_x_m.shape) <= AT_DISTANCE_MASK_M)

    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x_m.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    # Plot in EGSA meters, ticks shown as lon/lat degrees
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = EGSA_TO_WGS.transform(x, y_ref_for_lon)
        return fmt_decimal_comma(lon, 2)

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return fmt_decimal_comma(lat, 2)

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours_attica(ax, grid_x_m, grid_y_m, out)

    cbar = fig.colorbar(img, ax=ax, orientation="vertical", extend="both",
                        fraction=0.035, pad=0.02)
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)

    ax.set_title("Τρέχουσα θερμοκρασία ΒΑ Ελλάδας (προσαρμογή υψομέτρου)", fontsize=16, pad=10)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
    )

    main_path, ts_path = save_with_timestamp(fig, OUT_DIR, "tnow_negreece.png", athens_now)
    plt.close(fig)

    print("✅ Saved:", main_path)
    if ts_path:
        print("✅ Saved:", ts_path)

    return main_path, ts_path



def make_tnow_nwgreece_egsa(df, greece_gdf_wgs, dem_path, athens_now):
    if "TNow" not in df.columns:
        print("❌ TNow missing.")
        return (None, None)

    tt0 = df.copy()
    tt0["TNow"] = pd.to_numeric(tt0["TNow"], errors="coerce")
    tt0.dropna(subset=["TNow", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TNow"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]
    if tt0.empty:
        print("❌ No valid TNow data for NW Greece.")
        return (None, None)

    # Convert bbox corners to EGSA meters
    corners_lon = [NW_LON_MIN, NW_LON_MIN, NW_LON_MAX, NW_LON_MAX]
    corners_lat = [NW_LAT_MIN, NW_LAT_MAX, NW_LAT_MIN, NW_LAT_MAX]
    cx, cy = WGS_TO_EGSA.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(cx)), float(np.max(cx))
    y_min, y_max = float(np.min(cy)), float(np.max(cy))

    # Stations projected to EGSA
    st_lon = tt0["Longitude"].to_numpy(dtype=float)
    st_lat = tt0["Latitude"].to_numpy(dtype=float)
    st_t = tt0["TNow"].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    # Prefilter to nearby stations, buffer 200 km
    buf = 200_000.0
    near = (st_x >= (x_min - buf)) & (st_x <= (x_max + buf)) & (st_y >= (y_min - buf)) & (st_y <= (y_max + buf))
    st_lon = st_lon[near]
    st_lat = st_lat[near]
    st_t = st_t[near]
    st_x = st_x[near]
    st_y = st_y[near]

    if len(st_t) < 8:
        print("❌ Too few nearby stations for NW Greece interpolation.")
        return (None, None)

    # DEM altitude at stations, lon/lat sampling
    st_elev = sample_dem_lonlat(dem_path, st_lon, st_lat)

    ok = np.isfinite(st_t) & np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_elev)
    st_t = st_t[ok]
    st_x = st_x[ok]
    st_y = st_y[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 8:
        print("❌ Too few valid stations after DEM for NW Greece interpolation.")
        return (None, None)

    # Lapse per station, in meters space
    st_lapse = estimate_local_lapse_rates_egsa(st_x, st_y, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    # Grid in EGSA meters
    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, NW_N),
        np.linspace(y_min, y_max, NW_N)
    )

    # Interpolate t0 and lapse in meters
    t0_grid = idw_fast(
        st_x, st_y, st_t0,
        grid_x_m, grid_y_m,
        k=AT_IDW_K,
        power=AT_IDW_POWER,
        max_distance=AT_MAX_DISTANCE_M,
        min_neighbors=AT_MIN_NEIGHBORS
    )

    lapse_grid = idw_fast(
        st_x, st_y, st_lapse,
        grid_x_m, grid_y_m,
        k=AT_IDW_K,
        power=AT_IDW_POWER,
        max_distance=AT_MAX_DISTANCE_M,
        min_neighbors=AT_MIN_NEIGHBORS
    )

    # DEM on grid: EGSA -> lon/lat -> sample
    glon, glat = EGSA_TO_WGS.transform(grid_x_m.ravel().tolist(), grid_y_m.ravel().tolist())
    grid_elev = sample_dem_lonlat(
        dem_path,
        np.array(glon, dtype=float),
        np.array(glat, dtype=float)
    ).reshape(grid_x_m.shape)

    t_grid = t0_grid + (lapse_grid * grid_elev)

    # Greece boundary in EGSA and clipped to bbox
    greece_egsa = greece_gdf_wgs.to_crs(CRS_EGSA87)
    greece_clip = greece_egsa.cx[x_min:x_max, y_min:y_max].copy()

    if hasattr(greece_clip.geometry, "union_all"):
        boundary = greece_clip.geometry.union_all()
    else:
        boundary = greece_clip.geometry.unary_union

    # Geo mask on grid, EGSA
    grid_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_EGSA87
    )
    geo_mask = grid_pts.geometry.within(boundary).values.reshape(grid_x_m.shape)

    # Distance mask in meters
    tree = cKDTree(np.c_[st_x, st_y])
    d, _ = tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    dist_mask = (d.reshape(grid_x_m.shape) <= AT_DISTANCE_MASK_M)

    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x_m.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    # Plot in EGSA meters, ticks shown as lon/lat degrees
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = EGSA_TO_WGS.transform(x, y_ref_for_lon)
        return fmt_decimal_comma(lon, 2)

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return fmt_decimal_comma(lat, 2)

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours_attica(ax, grid_x_m, grid_y_m, out)

    cbar = fig.colorbar(
        img,
        ax=ax,
        orientation="vertical",
        extend="both",
        fraction=0.035,
        pad=0.02
    )
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)

    ax.set_title("Τρέχουσα θερμοκρασία ΒΔ Ελλάδας (προσαρμογή υψομέτρου)", fontsize=16, pad=10)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes,
        fontsize=9,
        color="black",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
    )

    main_path, ts_path = save_with_timestamp(fig, OUT_DIR, "tnow_nwgreece.png", athens_now)
    plt.close(fig)

    print("✅ Saved:", main_path)
    if ts_path:
        print("✅ Saved:", ts_path)

    return main_path, ts_path



def make_tnow_swgreece_egsa(df, greece_gdf_wgs, dem_path, athens_now):
    if "TNow" not in df.columns:
        print("❌ TNow missing.")
        return (None, None)

    tt0 = df.copy()
    tt0["TNow"] = pd.to_numeric(tt0["TNow"], errors="coerce")
    tt0.dropna(subset=["TNow", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TNow"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]
    if tt0.empty:
        print("❌ No valid TNow data for SW Greece.")
        return (None, None)

    # Convert bbox corners to EGSA meters
    corners_lon = [SW_LON_MIN, SW_LON_MIN, SW_LON_MAX, SW_LON_MAX]
    corners_lat = [SW_LAT_MIN, SW_LAT_MAX, SW_LAT_MIN, SW_LAT_MAX]
    cx, cy = WGS_TO_EGSA.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(cx)), float(np.max(cx))
    y_min, y_max = float(np.min(cy)), float(np.max(cy))

    # Stations projected to EGSA
    st_lon = tt0["Longitude"].to_numpy(dtype=float)
    st_lat = tt0["Latitude"].to_numpy(dtype=float)
    st_t = tt0["TNow"].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    # Prefilter to nearby stations (buffer 200 km)
    buf = 200_000.0
    near = (st_x >= (x_min - buf)) & (st_x <= (x_max + buf)) & (st_y >= (y_min - buf)) & (st_y <= (y_max + buf))
    st_lon = st_lon[near]
    st_lat = st_lat[near]
    st_t = st_t[near]
    st_x = st_x[near]
    st_y = st_y[near]

    if len(st_t) < 8:
        print("❌ Too few nearby stations for SW Greece interpolation.")
        return (None, None)

    # DEM altitude at stations (lon/lat sampling)
    st_elev = sample_dem_lonlat(dem_path, st_lon, st_lat)

    ok = np.isfinite(st_t) & np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_elev)
    st_t = st_t[ok]
    st_x = st_x[ok]
    st_y = st_y[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 8:
        print("❌ Too few valid stations (after DEM) for SW Greece interpolation.")
        return (None, None)

    # Lapse per station (in meters space)
    st_lapse = estimate_local_lapse_rates_egsa(st_x, st_y, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    # Grid in EGSA meters
    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, SW_N),
        np.linspace(y_min, y_max, SW_N)
    )

    # Interpolate t0 and lapse in meters
    t0_grid = idw_fast(st_x, st_y, st_t0, grid_x_m, grid_y_m,
                       k=AT_IDW_K, power=AT_IDW_POWER,
                       max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS)

    lapse_grid = idw_fast(st_x, st_y, st_lapse, grid_x_m, grid_y_m,
                          k=AT_IDW_K, power=AT_IDW_POWER,
                          max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS)

    # DEM on grid: EGSA -> lon/lat -> sample
    glon, glat = EGSA_TO_WGS.transform(grid_x_m.ravel().tolist(), grid_y_m.ravel().tolist())
    grid_elev = sample_dem_lonlat(dem_path, np.array(glon, dtype=float), np.array(glat, dtype=float)).reshape(grid_x_m.shape)

    t_grid = t0_grid + (lapse_grid * grid_elev)

    # Greece boundary in EGSA and clipped to bbox
    greece_egsa = greece_gdf_wgs.to_crs(CRS_EGSA87)
    greece_clip = greece_egsa.cx[x_min:x_max, y_min:y_max].copy()

    if hasattr(greece_clip.geometry, "union_all"):
        boundary = greece_clip.geometry.union_all()
    else:
        boundary = greece_clip.geometry.unary_union

    # Geo mask on grid (EGSA)
    grid_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_EGSA87
    )
    geo_mask = grid_pts.geometry.within(boundary).values.reshape(grid_x_m.shape)

    # Distance mask in meters
    tree = cKDTree(np.c_[st_x, st_y])
    d, _ = tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    dist_mask = (d.reshape(grid_x_m.shape) <= AT_DISTANCE_MASK_M)

    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x_m.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    # Plot in EGSA meters, ticks shown as lon/lat degrees
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = EGSA_TO_WGS.transform(x, y_ref_for_lon)
        return fmt_decimal_comma(lon, 2)

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return fmt_decimal_comma(lat, 2)

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours_attica(ax, grid_x_m, grid_y_m, out)

    cbar = fig.colorbar(img, ax=ax, orientation="vertical", extend="both",
                        fraction=0.035, pad=0.02)
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)

    ax.set_title("Τρέχουσα θερμοκρασία ΝΔ Ελλάδας (προσαρμογή υψομέτρου)", fontsize=16, pad=10)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
    )

    main_path, ts_path = save_with_timestamp(fig, OUT_DIR, "tnow_swgreece.png", athens_now)
    plt.close(fig)

    print("✅ Saved:", main_path)
    if ts_path:
        print("✅ Saved:", ts_path)

    return main_path, ts_path

def make_tnow_cyprus_utm(df, athens_now):
    if "TNow" not in df.columns:
        print("❌ TNow missing.")
        return (None, None)

    if not os.path.exists(CYPRUS_GEOJSON_PATH):
        print(f"❌ Missing Cyprus GeoJSON: {CYPRUS_GEOJSON_PATH}")
        return (None, None)

    if not os.path.exists(CYPRUS_ALT_TIF_PATH):
        print(f"❌ Missing Cyprus altitude raster: {CYPRUS_ALT_TIF_PATH}")
        return (None, None)

    tt0 = df.copy()
    tt0["TNow"] = pd.to_numeric(tt0["TNow"], errors="coerce")
    tt0.dropna(subset=["TNow", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TNow"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]

    tt0 = tt0[
        tt0["Longitude"].between(CY_LON_MIN, CY_LON_MAX) &
        tt0["Latitude"].between(CY_LAT_MIN, CY_LAT_MAX)
    ].copy()

    if tt0.empty:
        print("❌ No valid TNow data for Cyprus.")
        return (None, None)

    st_lon = tt0["Longitude"].to_numpy(dtype=float)
    st_lat = tt0["Latitude"].to_numpy(dtype=float)
    st_t = tt0["TNow"].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_UTM36N.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    st_elev = sample_raster_xy(
        CYPRUS_ALT_TIF_PATH,
        st_lon,
        st_lat,
        input_crs=CRS_WGS84
    )

    ok = np.isfinite(st_t) & np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_elev)
    st_t = st_t[ok]
    st_x = st_x[ok]
    st_y = st_y[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 5:
        print("❌ Too few valid stations for Cyprus interpolation.")
        return (None, None)

    lapse = fit_lapse_rate_simple(st_t, st_elev)
    st_t0 = st_t - (lapse * st_elev)

    corners_lon = [CY_LON_MIN, CY_LON_MAX, CY_LON_MIN, CY_LON_MAX]
    corners_lat = [CY_LAT_MIN, CY_LAT_MIN, CY_LAT_MAX, CY_LAT_MAX]
    cx, cy = WGS_TO_UTM36N.transform(corners_lon, corners_lat)

    x_min, x_max = float(np.min(cx)), float(np.max(cx))
    y_min, y_max = float(np.min(cy)), float(np.max(cy))

    grid_x_m, grid_y_m = np.meshgrid(
        np.linspace(x_min, x_max, CY_N),
        np.linspace(y_min, y_max, CY_N)
    )

    t0_grid = idw_fast(
        st_x, st_y, st_t0,
        grid_x_m, grid_y_m,
        k=CY_IDW_K,
        power=CY_IDW_POWER,
        max_distance=CY_MAX_DISTANCE_M,
        min_neighbors=CY_MIN_NEIGHBORS
    )

    grid_elev = sample_raster_xy(
        CYPRUS_ALT_TIF_PATH,
        grid_x_m.ravel(),
        grid_y_m.ravel(),
        input_crs=CRS_UTM36N
    ).reshape(grid_x_m.shape)

    t_grid = t0_grid + (lapse * grid_elev)

    cyprus = gpd.read_file(CYPRUS_GEOJSON_PATH)

    if cyprus.crs is None:
        cyprus = cyprus.set_crs(CRS_WGS84)

    cyprus = cyprus[~cyprus.geometry.is_empty].copy()

    if not cyprus.geometry.is_valid.all():
        cyprus.geometry = cyprus.buffer(0)

    if cyprus.crs.to_string() != CRS_WGS84:
        cyprus_ll = cyprus.to_crs(CRS_WGS84)
    else:
        cyprus_ll = cyprus.copy()

    if hasattr(cyprus_ll.geometry, "union_all"):
        cyprus_boundary_ll = cyprus_ll.geometry.union_all()
    else:
        cyprus_boundary_ll = cyprus_ll.geometry.unary_union

    if not bounds_reasonable_cyprus(cyprus_boundary_ll):
        cyprus_ll.geometry = cyprus_ll.geometry.apply(swap_geom_xy)

    cyprus_utm = cyprus_ll.to_crs(CRS_UTM36N)

    if hasattr(cyprus_utm.geometry, "union_all"):
        cyprus_boundary_utm = cyprus_utm.geometry.union_all()
    else:
        cyprus_boundary_utm = cyprus_utm.geometry.unary_union

    grid_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
        crs=CRS_UTM36N
    )

    geo_mask = grid_pts.geometry.within(cyprus_boundary_utm).values.reshape(grid_x_m.shape)

    tree = cKDTree(np.c_[st_x, st_y])
    d, _ = tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    dist_mask = (d.reshape(grid_x_m.shape) <= CY_DISTANCE_MASK_M)

    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x_m.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    cyprus_utm.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    y_ref_for_lon = y_min
    x_ref_for_lat = x_min
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    def fmt_lon(x, pos):
        lon, _lat = UTM36N_TO_WGS.transform(x, y_ref_for_lon)
        return fmt_decimal_comma(lon, 2)

    def fmt_lat(y, pos):
        _lon, lat = UTM36N_TO_WGS.transform(x_ref_for_lat, y)
        return fmt_decimal_comma(lat, 2)

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours_attica(ax, grid_x_m, grid_y_m, out)

    cbar = fig.colorbar(
        img,
        ax=ax,
        orientation="vertical",
        extend="both",
        fraction=0.035,
        pad=0.02
    )
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)

    ax.set_title("Τρέχουσα θερμοκρασία Κύπρου (προσαρμογή υψομέτρου)", fontsize=16, pad=10)

    occupied_disclaimer = (
        "Δεδομένα από τα κατεχόμενα τμήματα της Κύπρου χρησιμοποιούνται\n"
        "μόνο για μετεωρολογικούς σκοπούς, χωρίς αναγνώριση καθεστώτος,\n"
        "σύμφωνα με τα Ψηφίσματα 541 και 550 του ΣΑ/ΟΗΕ."
    )

    ax.text(
        0.01, 0.99, occupied_disclaimer,
        transform=ax.transAxes,
        fontsize=5.8,
        color="black",
        ha="left",
        va="top",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.2"),
        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        zorder=30
    )

    add_top5_box_cyprus(ax, tt0)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes,
        fontsize=9,
        color="black",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
    )

    main_path, ts_path = save_with_timestamp(fig, OUT_DIR, "tnow_cyprus.png", athens_now)
    plt.close(fig)

    print("✅ Saved:", main_path)
    if ts_path:
        print("✅ Saved:", ts_path)

    return main_path, ts_path

def filter_fresh_rows(data: pd.DataFrame, athens_now: datetime, max_age_minutes: int = 60) -> pd.DataFrame:
    """
    Keep only rows with a valid Datetime and age <= max_age_minutes, using Athens timezone.
    Assumes feed timestamps are Athens local time (naive).
    """
    if "Datetime" not in data.columns:
        return data

    d = data.copy()

    dt = pd.to_datetime(d["Datetime"], errors="coerce")

    # If timestamps are naive (no timezone), assume they are Athens local time
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("Europe/Athens", nonexistent="shift_forward", ambiguous="NaT")
    else:
        dt = dt.dt.tz_convert("Europe/Athens")

    d["Datetime"] = dt

    delta = athens_now - d["Datetime"]
    age_min = delta.dt.total_seconds() / 60.0

    d = d[d["Datetime"].notna()]
    d = d[(age_min >= 0.0) & (age_min <= float(max_age_minutes))]

    return d



# =========================
# MAIN
# =========================
def main():
    print("✅ RUNNING:", os.path.abspath(__file__))
    print("✅ Output folder:", OUT_DIR)
    print("✅ FTP enabled:", bool(FTP_HOST and FTP_USER and FTP_PASS))

    # Make sure the DEM bundle is in place if needed
    ensure_altitude_bundle()

    if not os.path.exists(GEOJSON_PATH):
        raise FileNotFoundError(f"Missing {GEOJSON_PATH}")
    if not os.path.exists(DEM_PATH):
        raise FileNotFoundError(f"Missing DEM VRT at {DEM_PATH}")

    text = fetch_text(DATA_URL)
    data = read_tabbed_df(text)

    if "Datetime" not in data.columns:
        print("❌ Datetime column missing. Parsed columns:", list(data.columns))
        raise SystemExit(1)

    for col in ["Latitude", "Longitude", "TNow", "WindSpeedNow", "RHNow"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data[(data["Latitude"].notna()) & (data["Longitude"].notna())]
    data = data[(data["Latitude"] != 0) & (data["Longitude"] != 0)]

    # Keep both Greece and Cyprus rows.
    # Greece regional/national functions filter by their own bboxes later.
    # Cyprus needs longitudes around 32-35E, so do NOT globally cut at <=30.
    data = data[
        (data["Longitude"] >= 19.0) &
        (data["Longitude"] <= 35.0) &
        (data["Latitude"] >= 34.0) &
        (data["Latitude"] <= 42.8)
    ]
    
    # Exclude specific stations from all maps and top-10 lists
    if "webcode" in data.columns:
        data = data[~data["webcode"].astype(str).str.strip().isin(["wu_panoramavoulas", "wu_tilos"])]
    
    # Parse Datetime (timezone handling is done inside filter_fresh_rows)
    # data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")

    if data.empty:
        print("❌ No usable rows after cleaning.")
        return

    # Freshness filters:
    # - Greece and Greek regional maps keep the strict 60-minute rule.
    # - Cyprus gets a separate, longer window because some Cyprus stations lag.
    athens_now = datetime.now(ZoneInfo("Europe/Athens"))

    data_all_clean = data.copy()

    data = filter_fresh_rows(data_all_clean, athens_now, max_age_minutes=60)

    cyprus_data = filter_fresh_rows(
        data_all_clean,
        athens_now,
        max_age_minutes=CY_MAX_AGE_MINUTES
    )

    # Keep only Cyprus bbox rows in the relaxed Cyprus dataframe.
    cyprus_data = cyprus_data[
        cyprus_data["Longitude"].between(CY_LON_MIN, CY_LON_MAX) &
        cyprus_data["Latitude"].between(CY_LAT_MIN, CY_LAT_MAX)
    ].copy()

    # If the feed ever contains more than one row per station within the Cyprus
    # relaxed window, keep only the newest one per station.
    if "webcode" in cyprus_data.columns and "Datetime" in cyprus_data.columns:
        cyprus_data = (
            cyprus_data
            .sort_values("Datetime")
            .drop_duplicates(subset=["webcode"], keep="last")
        )

    if data.empty:
        print("❌ No usable rows after Greece freshness filter (older than 60 minutes).")
        return

    if cyprus_data.empty:
        print(f"❌ No usable Cyprus rows after freshness filter (older than {CY_MAX_AGE_MINUTES} minutes).")


    greece = gpd.read_file(GEOJSON_PATH)
    if greece.crs is None:
        greece = greece.set_crs(CRS_WGS84)

    # 1) Attica first
    att_main, att_ts = make_tnow_attica_egsa(data, greece, DEM_PATH, athens_now)

    # 2) Extra regional maps
    crete_main, crete_ts = make_tnow_crete_egsa(data, greece, DEM_PATH, athens_now)
    ne_main, ne_ts = make_tnow_negreece_egsa(data, greece, DEM_PATH, athens_now)
    nw_main, nw_ts = make_tnow_nwgreece_egsa(data, greece, DEM_PATH, athens_now)
    sw_main, sw_ts = make_tnow_swgreece_egsa(data, greece, DEM_PATH, athens_now)

    # 3) Cyprus, using the Cyprus-only relaxed freshness window
    if not cyprus_data.empty:
        cy_main, cy_ts = make_tnow_cyprus_utm(cyprus_data, athens_now)
    else:
        cy_main, cy_ts = None, None
    # 4) Greece last
    gr_main, gr_ts = make_tnow_greece_wgs(data, greece, DEM_PATH, athens_now)

    # Upload ONLY the stable filenames, keep timestamped copies local only
    for p in [att_main, crete_main, ne_main, nw_main, sw_main, cy_main, gr_main]:
        if p and os.path.exists(p):
            try:
                upload_to_ftp(p)
            except Exception as e:
                print(f"⚠️ FTP upload failed for {os.path.basename(p)}: {e}")

if __name__ == "__main__":
    _t0 = perf_counter()
    try:
        main()
    finally:
        elapsed = perf_counter() - _t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = elapsed % 60
        print(f"⏱️ Total runtime: {h:02d}:{m:02d}:{s:05.2f} (hh:mm:ss.ss)")
