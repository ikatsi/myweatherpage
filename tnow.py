#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tnow.py
#
# Produces EIGHT files (Attica, NE Greece, Crete, Greece), ALL in ONE folder:  ./Tnowmaps/
#   1) tnow_attica.png
#   2) tnow_attica_YYYYMMDD_HHMM.png
#   3) tnow_negreece.png
#   4) tnow_negreece_YYYYMMDD_HHMM.png
#   5) tnow_crete.png
#   6) tnow_crete_YYYYMMDD_HHMM.png
#   7) tnow.png
#   8) tnow_YYYYMMDD_HHMM.png
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
import re
import time
import shutil
import subprocess, zipfile
from time import perf_counter
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo


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
# CONFIG (no secrets here)
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__) or ".")
GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
DEM_PATH = os.path.join(BASE_DIR, "GRC_alt.vrt")

# All sensitive values are injected via environment variables by the CI runner.
# You created these in GitHub → Settings → Secrets and variables → Actions.
DATA_URL = os.environ.get("CURRENTWEATHER_URL", "").strip()  # your secret name
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()  # empty disables uploads

# ALL outputs here
OUT_DIR = os.path.join(BASE_DIR, "Tnowmaps")

# --- Greece (keep as-is: WGS84 degrees) ---
GR_LON_MIN, GR_LON_MAX = 19.0, 30.0
GR_LAT_MIN, GR_LAT_MAX = 34.5, 42.5
GR_N = 300

# --- Attica bbox: EXACTLY like your rain Attica script ---
AT_LON_MIN, AT_LON_MAX = 22.7, 25.0
AT_LAT_MIN, AT_LAT_MAX = 37.5, 38.7
AT_N = 300

# --- Crete bbox (same EGSA approach as Attica) ---
CR_LON_MIN, CR_LON_MAX = 23.0, 26.5
CR_LAT_MIN, CR_LAT_MAX = 34.5, 36.0
CR_N = 300

# --- NE Greece bbox (EGSA approach like Attica/Crete) ---
NE_LON_MIN, NE_LON_MAX = 22.0, 26.8
NE_LAT_MIN, NE_LAT_MAX = 39.8, 41.9
NE_N = 300

# Fetch/retry
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/plain, text/*;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8,el;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
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



# ---------- TOP-5 BOX formatting (same abbreviations as today.py) ----------
TOPBOX_NAME_MAX = 26

def prettify_station_name(s: str) -> str:
    if s is None:
        return "–"
    s = str(s).strip()
    s = s.replace("Μητροπολιτικό Πάρκο", "Πάρκο")
    s = s.replace("Διεθνές Αεροδρόμιο", "Α/Δ")
    s = s.replace("Αεροδρόμιο", "Α/Δ")
    s = s.replace("Πανεπιστήμιο", "Παν.")
    s = s.replace("Νοσοκομείο", "Νοσ.")
    s = s.replace("Χιονοδρομικό κέντρο", "Χ/Κ")
    s = s.replace("Καταφύγιο", "Καταφ.")
    s = s.replace("Όρος", "Όρ.")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ellipsize(s: str, max_chars: int = 42) -> str:
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"

def shorten_for_box(name: str, max_chars: int = TOPBOX_NAME_MAX) -> str:
    s = prettify_station_name(name)
    if "(" in s and ")" in s:
        base = s.split("(", 1)[0].strip()
        if base:
            s = base
    s = s.replace("«", "").replace("»", "").replace('"', "").replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    return ellipsize(s, max_chars=max_chars)


# Shared palette (same as today.py)
TEMP_VMIN = -25.0
TEMP_VMAX = 45.0

# Attica EGSA settings (meters)
CRS_WGS84 = "EPSG:4326"
CRS_EGSA87 = "EPSG:2100"
WGS_TO_EGSA = Transformer.from_crs(CRS_WGS84, CRS_EGSA87, always_xy=True)
EGSA_TO_WGS = Transformer.from_crs(CRS_EGSA87, CRS_WGS84, always_xy=True)

# IDW / masks in meters for Attica
AT_IDW_K = 8
AT_IDW_POWER = 2
AT_MAX_DISTANCE_M = 120_000
AT_MIN_NEIGHBORS = 3
AT_DISTANCE_MASK_M = 170_000

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
        raise SystemExit("CURRENTWEATHER_URL is not set.")
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
                try:
                    preview = (r.text or "")[:300].replace("\n", "\\n")
                except Exception:
                    preview = ""
                if preview:
                    print("ℹ️ Body preview:", preview)

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
                preview = text[:200].replace("\n", "\\n")
                raise requests.exceptions.RequestException(
                    "Response did not look like tab-delimited weather data. Preview: " + preview
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

    zi_ok = np.where(den > 0, num / den, np.nan)
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


# =========================
# MASKS / CONTOURS / STAMP
# =========================
def stamp_text(athens_now: datetime) -> str:
    ts = athens_now.strftime("%Y-%m-%d %H:%M %Z")
    return "Δημιουργήθηκε για το e-kairos.gr\n" + ts


def pick_station_label_column(df: pd.DataFrame) -> str:
    """
    Picks the best column to display for station name/area.
    Preference: citygr -> CityGR -> station -> name -> webcode.
    """
    for c in ["citygr", "Citygr", "CityGR", "station", "name", "webcode"]:
        if c in df.columns:
            return c
    return "webcode"


def add_top5_box_greece(ax, tt0: pd.DataFrame, frost_text: str = "") -> None:
    """
    Transparent top-right info + map markers (1..5).
    - Cold 5: blue numbers
    - Hot 5: red numbers
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

    cold5 = tmp.nsmallest(5, "TNow").copy()
    hot5  = tmp.nlargest(5, "TNow").copy()

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
            lines.append("{0}. {1}: {2:.1f}°C".format(i, name, t))
            i += 1
        return "\n".join(lines)

    box_text = fmt_block(cold5, "Ψυχρότερες 5 περιοχές") + "\n\n" + fmt_block(hot5, "Θερμότερες 5 περιοχές")

    # ---- add frost line (only when provided)
    if frost_text:
        box_text = box_text + "\n\n" + frost_text

    # Transparent box (no background), keep readability with white stroke
    ax.text(
        0.99, 0.99, box_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        color="black",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.35"),
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")]
    )

    # ---- draw map markers 1..5 for cold/hot (with white outline)
    def draw_rank_markers(dfx: pd.DataFrame, color: str):
        rank = 1
        for _, r in dfx.iterrows():
            try:
                lon = float(r["Longitude"])
                lat = float(r["Latitude"])
            except Exception:
                continue

            # subtle ring so the point is visible
            ax.scatter([lon], [lat], s=90, facecolors="none", edgecolors=color,
                       linewidths=1.4, zorder=12)

            txt = ax.text(
                lon, lat, str(rank),
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=color, zorder=13
            )
            txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])
            rank += 1
            if rank > 5:
                break

    draw_rank_markers(cold5, color="#1d4ed8")  # blue-ish
    draw_rank_markers(hot5,  color="#dc2626")  # red-ish


def add_contours(ax, X, Y, field):
    levels = np.arange(-30, 46, 3, dtype=float)
    thin_levels = [lv for lv in levels if abs(lv) > 1e-9]

    cs_thin = None
    cs_zero = None

    # thin contours (every 3°C except 0)
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

    # thick 0°C contour
    try:
        cs_zero = ax.contour(
            X, Y, field,
            levels=[0.0],
            colors="black",
            linewidths=1.3,
            alpha=0.95
        )
    except Exception:
        cs_zero = None

    # ---- labels (small, unobtrusive, readable)
    # label every other contour to reduce clutter
    if cs_thin is not None:
        try:
            label_levels = cs_thin.levels[:]    # label every contour level
            texts = ax.clabel(
                cs_thin,
                levels=label_levels,
                inline=True,
                inline_spacing=2,
                fmt="%d",
                fontsize=7  # small
            )
            for t in texts:
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        except Exception:
            pass

    # label 0°C as well (even smaller, just "0°C")
    if cs_zero is not None:
        try:
            texts0 = ax.clabel(
                cs_zero,
                inline=True,
                inline_spacing=2,
                fmt="0",
                fontsize=7
            )
            for t in texts0:
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        except Exception:
            pass


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

    st_lapse = estimate_local_lapse_rates_wgs(st_lons, st_lats, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    t0_grid = idw_fast(st_lons, st_lats, st_t0, grid_x, grid_y, k=8, power=2,
                       max_distance=1.2, min_neighbors=3)
    lapse_grid = idw_fast(st_lons, st_lats, st_lapse, grid_x, grid_y, k=8, power=2,
                          max_distance=1.2, min_neighbors=3)

    grid_elev = sample_dem_lonlat(dem_path, grid_x.ravel(), grid_y.ravel()).reshape(grid_x.shape)
    t_grid = t0_grid + (lapse_grid * grid_elev)

    dist_mask = build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_dist=1.5)
    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x.shape, np.nan, dtype=float)
    out[final_mask] = t_grid[final_mask]

    # =========================
    # NEW: % of territory (modelled) with air frost (T <= 0°C)
    # Shown ONLY if the rounded-to-2-decimals value is > 0.00%
    # Text is forced to NOT be a single line (newline inserted).
    # =========================
    frost_text = ""
    try:
        denom = float(np.sum(final_mask))
        if denom > 0:
            frost_cells = np.sum(final_mask & (out <= 0.0))
            frost_pct = 100.0 * float(frost_cells) / denom
            if round(frost_pct, 2) > 0.0:
                frost_text = f"{frost_pct:.1f}% της επικράτειας\nμε παγετό αέρα"
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
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)

    add_top5_box_greece(ax, tt0, frost_text=frost_text)

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
        return f"{lon:.2f}"

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return f"{lat:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours(ax, grid_x_m, grid_y_m, out)

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
        return f"{lon:.2f}"

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return f"{lat:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours(ax, grid_x_m, grid_y_m, out)

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
        return f"{lon:.2f}"

    def fmt_lat(y, pos):
        _lon, lat = EGSA_TO_WGS.transform(x_ref_for_lat, y)
        return f"{lat:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.set_xlabel("Γεωγρ. μήκος (°)", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος (°)", fontsize=12)

    add_contours(ax, grid_x_m, grid_y_m, out)

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



def filter_fresh_rows(data: pd.DataFrame, athens_now: datetime, max_age_minutes: int = 60) -> pd.DataFrame:
    """
    Keep only rows with a valid Datetime and age <= max_age_minutes, using Athens timezone.
    Also drops rows that appear "too far in the future" (clock issues).
    """
    if "Datetime" not in data.columns:
        return data

    d = data.copy()
    d["Datetime"] = pd.to_datetime(d["Datetime"], errors="coerce")

    # Ensure Datetime is timezone-aware Athens
    if getattr(d["Datetime"].dt, "tz", None) is None:
        d["Datetime"] = d["Datetime"].dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="shift_forward")
    else:
        d["Datetime"] = d["Datetime"].dt.tz_convert("Europe/Athens")

    # Age in minutes
    delta = athens_now - d["Datetime"]
    age_min = delta.dt.total_seconds() / 60.0

    # Keep only: not NaT, not negative (future), and <= max_age_minutes
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

    for col in ["Latitude", "Longitude", "TNow"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data[(data["Latitude"].notna()) & (data["Longitude"].notna())]
    data = data[(data["Latitude"] != 0) & (data["Longitude"] != 0)]
    data = data[data["Longitude"] <= 30]

    # Parse Datetime (timezone handling is done inside filter_fresh_rows)
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")

    if data.empty:
        print("❌ No usable rows after cleaning.")
        return

    # Freshness filter: keep only stations with Datetime <= 60 minutes old (Athens time)

    athens_now = datetime.now(ZoneInfo("Europe/Athens"))
    data = filter_fresh_rows(data, athens_now, max_age_minutes=60)

    if data.empty:
        print("❌ No usable rows after freshness filter (older than 60 minutes).")
        return


    greece = gpd.read_file(GEOJSON_PATH)
    if greece.crs is None:
        greece = greece.set_crs(CRS_WGS84)

    # 1) Attica first
    att_main, att_ts = make_tnow_attica_egsa(data, greece, DEM_PATH, athens_now)
    
    # 2) NE Greece second
    ne_main, ne_ts = make_tnow_negreece_egsa(data, greece, DEM_PATH, athens_now)
    
    # 3) Crete third
    cr_main, cr_ts = make_tnow_crete_egsa(data, greece, DEM_PATH, athens_now)
    
    # 4) Greece last (leave it as-is)
    gr_main, gr_ts = make_tnow_greece_wgs(data, greece, DEM_PATH, athens_now)
    
    # Upload ONLY the stable filenames (keep timestamped copies local only)
    for p in [att_main, ne_main, cr_main, gr_main]:

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
