#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# today.py
#
# Produces national + regional maps for:
#   1) TodayRain
#   2) TMin
#   3) TMax
#
# Regions:
#   - Attica
#   - NE Greece
#   - SW Greece
#   - Crete
#   - Greece (national)
#
# Stable national filenames stay unchanged:
#   todayrain.png
#   tmin.png
#   tmax.png
#
# Regional stable filenames follow the tnow.py rationale:
#   todayrain_attica.png
#   todayrain_negreece.png
#   todayrain_swgreece.png
#   todayrain_crete.png
#   tmin_attica.png
#   tmin_negreece.png
#   tmin_swgreece.png
#   tmin_crete.png
#   tmax_attica.png
#   tmax_negreece.png
#   tmax_swgreece.png
#   tmax_crete.png
#
# Uploads only the stable filenames to FTP.

import os
import re
import time
import socket
import subprocess
import zipfile
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import numpy.ma as ma
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter, MaxNLocator

from scipy.spatial import cKDTree
import requests
from ftplib import FTP_TLS
import rasterio
from pyproj import Transformer
from common_abbrev import shorten_for_box
import json


# =========================
# CONFIG
# =========================
EXCLUDE_ALL_WEBCODES = {"pws_gebze"}
RAW_EXCLUSION_RULES = os.environ.get("STATION_EXCLUSION_RULES", "").strip()

BASE_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
DEM_PATH = os.path.join(BASE_DIR, "GRC_alt.vrt")

GEOJSON_ENC = os.path.join(BASE_DIR, "greece.geojson.enc")
ALT_ENC = os.path.join(BASE_DIR, "altitude.zip.enc")
ALT_ZIP = os.path.join(BASE_DIR, "altitude.zip")

DATA_URL = os.environ.get("CURRENTWEATHER_URL", "").strip()
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()
BRAND_NAME = os.environ.get("BRAND_NAME", "").strip()

# National grid
GRID_LON_MIN, GRID_LON_MAX = 19.0, 30.0
GRID_LAT_MIN, GRID_LAT_MAX = 34.5, 42.5
GRID_N = 300

# Fixed rasterized Greek land area for the 300 x 300 national grid.
# Calculated once from the same greece.geojson boundary and latitude-weighted
# grid-cell areas. Used as the denominator for national territorial percentages.
GREECE_RASTERIZED_LAND_AREA_KM2 = 131595.026512276

# Regional bboxes from tnow.py
AT_LON_MIN, AT_LON_MAX = 22.7, 25.0
AT_LAT_MIN, AT_LAT_MAX = 37.5, 38.7
AT_N = 300

CR_LON_MIN, CR_LON_MAX = 23.37, 26.4
CR_LAT_MIN, CR_LAT_MAX = 34.7, 35.78
CR_N = 300

NE_LON_MIN, NE_LON_MAX = 22.0, 26.6
NE_LAT_MIN, NE_LAT_MAX = 39.7, 41.8
NE_N = 300

SW_LON_MIN, SW_LON_MAX = 20.0, 24.0
SW_LAT_MIN, SW_LAT_MAX = 36.0, 39.0
SW_N = 300

REGIONS = [
    {
        "key": "attica",
        "title_rain": "Σωρευτικός υετός ημέρας Αττικής",
        "title_tmin": "Ελάχιστη θερμοκρασία Αττικής (προσαρμογή υψομέτρου)",
        "title_tmax": "Μέγιστη θερμοκρασία Αττικής (προσαρμογή υψομέτρου)",
        "lon_min": AT_LON_MIN, "lon_max": AT_LON_MAX,
        "lat_min": AT_LAT_MIN, "lat_max": AT_LAT_MAX,
        "n": AT_N,
    },
    {
        "key": "negreece",
        "title_rain": "Σωρευτικός υετός ημέρας ΒΑ Ελλάδας",
        "title_tmin": "Ελάχιστη θερμοκρασία ΒΑ Ελλάδας (προσαρμογή υψομέτρου)",
        "title_tmax": "Μέγιστη θερμοκρασία ΒΑ Ελλάδας (προσαρμογή υψομέτρου)",
        "lon_min": NE_LON_MIN, "lon_max": NE_LON_MAX,
        "lat_min": NE_LAT_MIN, "lat_max": NE_LAT_MAX,
        "n": NE_N,
    },
    {
        "key": "swgreece",
        "title_rain": "Σωρευτικός υετός ημέρας ΝΔ Ελλάδας",
        "title_tmin": "Ελάχιστη θερμοκρασία ΝΔ Ελλάδας (προσαρμογή υψομέτρου)",
        "title_tmax": "Μέγιστη θερμοκρασία ΝΔ Ελλάδας (προσαρμογή υψομέτρου)",
        "lon_min": SW_LON_MIN, "lon_max": SW_LON_MAX,
        "lat_min": SW_LAT_MIN, "lat_max": SW_LAT_MAX,
        "n": SW_N,
    },
    {
        "key": "crete",
        "title_rain": "Σωρευτικός υετός ημέρας Κρήτης",
        "title_tmin": "Ελάχιστη θερμοκρασία Κρήτης (προσαρμογή υψομέτρου)",
        "title_tmax": "Μέγιστη θερμοκρασία Κρήτης (προσαρμογή υψομέτρου)",
        "lon_min": CR_LON_MIN, "lon_max": CR_LON_MAX,
        "lat_min": CR_LAT_MIN, "lat_max": CR_LAT_MAX,
        "n": CR_N,
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/plain, text/*;q=0.9, */*;q=0.8",
    "Accept-Encoding": "identity",
}
MAX_RETRIES = 5
DELAY = 10
TIMEOUT = 20

SENTINEL_TEMP = -67.8

TOPBOX_NAME_MAX = 26
TOP_RAIN_N = 15

TEMP_HARD_MIN = -30.0
TEMP_HARD_MAX = 49.0

TEMP_VMIN = -25.0
TEMP_VMAX = 45.0

CRS_WGS84 = "EPSG:4326"
CRS_EGSA87 = "EPSG:2100"
WGS_TO_EGSA = Transformer.from_crs(CRS_WGS84, CRS_EGSA87, always_xy=True)
EGSA_TO_WGS = Transformer.from_crs(CRS_EGSA87, CRS_WGS84, always_xy=True)

AT_IDW_K = 8
AT_IDW_POWER = 2
AT_MAX_DISTANCE_M = 120_000
AT_MIN_NEIGHBORS = 3
AT_DISTANCE_MASK_M = 170_000

LAPSE_DEFAULT = -0.0065
LAPSE_MIN = -0.0150
LAPSE_MAX = 0.0050
LAPSE_K = 25
LAPSE_RADIUS_M = 150_000
LAPSE_MIN_NBR = 8
LAPSE_ALT_RANGE_MIN_M = 200

# =========================
# SHARED EXCLUSION RULES
# =========================
def load_exclusion_rules():
    if not RAW_EXCLUSION_RULES:
        return {
            "version": None,
            "propagation": {
                "rain_implies_precip": True,
                "today_precip_implies_rain": True
            },
            "hard_excludes": {
                "temperature": [],
                "precip": [],
                "rain": []
            },
            "hard_exclude_prefixes": {
                "temperature": [],
                "precip": [],
                "rain": []
            },
            "date_rules": []
        }

    try:
        rules = json.loads(RAW_EXCLUSION_RULES)
    except Exception as e:
        raise SystemExit("Failed to parse STATION_EXCLUSION_RULES JSON: {}".format(e))

    if not isinstance(rules, dict):
        raise SystemExit("STATION_EXCLUSION_RULES must decode to a JSON object.")

    rules.setdefault("propagation", {})
    rules.setdefault("hard_excludes", {})
    rules.setdefault("hard_exclude_prefixes", {})
    rules.setdefault("date_rules", [])

    for fam in ("temperature", "precip", "rain"):
        rules["hard_excludes"].setdefault(fam, [])
        rules["hard_exclude_prefixes"].setdefault(fam, [])

    return rules


EXCLUSION_RULES = load_exclusion_rules()


def _norm_webcode(x):
    if x is None:
        return ""
    return str(x).strip().casefold()


def _norm_family(x):
    return str(x).strip().casefold()


def _parse_rule_date(x):
    if x in (None, "", "null"):
        return None
    return pd.to_datetime(x, errors="coerce").date()


def is_excluded_for_family(webcode, family, on_date, rules=None):
    if rules is None:
        rules = EXCLUSION_RULES

    wc = _norm_webcode(webcode)
    fam = _norm_family(family)

    if not wc:
        return False

    hard = {_norm_webcode(x) for x in rules.get("hard_excludes", {}).get(fam, [])}
    if wc in hard:
        return True

    prefixes = [
        str(x).strip().casefold()
        for x in rules.get("hard_exclude_prefixes", {}).get(fam, [])
        if str(x).strip()
    ]
    for pref in prefixes:
        if wc.startswith(pref):
            return True

    for rule in rules.get("date_rules", []):
        if _norm_webcode(rule.get("webcode")) != wc:
            continue
        if _norm_family(rule.get("family")) != fam:
            continue

        start = _parse_rule_date(rule.get("start"))
        end = _parse_rule_date(rule.get("end"))

        if start is not None and on_date < start:
            continue
        if end is not None and on_date > end:
            continue

        return True

    return False


def apply_family_exclusions(df, family, on_date, rules=None):
    if rules is None:
        rules = EXCLUSION_RULES

    if "webcode" not in df.columns:
        return df.copy()

    out = df.copy()
    webcodes = (
        out["webcode"].astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
    )

    mask = webcodes.apply(lambda w: is_excluded_for_family(w, family, on_date, rules))
    return out.loc[~mask].copy()

# =========================
# SHARED TEMP PALETTE
# =========================
def build_shared_temp_cmap_norm():
    anchors = [
        (-25.0, "#0b1d5c"),
        (-18.0, "#123b8a"),
        (-12.0, "#1f63c6"),
        (-6.0,  "#2f8fe6"),
        (-2.0,  "#44b6ff"),
        (0.0,   "#2b7bff"),
        (3.0,   "#2fb8d6"),
        (7.0,   "#2fc4a0"),
        (12.0,  "#34c759"),
        (18.0,  "#b7dd2a"),
        (24.0,  "#ffe11a"),
        (30.0,  "#ff9a1a"),
        (35.0,  "#ff4d1a"),
        (40.0,  "#d1166f"),
        (45.0,  "#6a00a8"),
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
# ENCRYPTED ASSET HANDLING
# =========================
def _openssl_decrypt(enc_path: str, out_path: str, passphrase: str) -> None:
    if not passphrase:
        raise SystemExit("GEOJSON_PASS not set.")
    try:
        subprocess.check_call([
            "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
            "-in", enc_path, "-out", out_path,
            "-pass", "pass:" + passphrase
        ])
    except FileNotFoundError:
        raise SystemExit("OpenSSL not found on runner.")
    except subprocess.CalledProcessError as e:
        raise SystemExit("OpenSSL decryption failed for %s: %s" % (enc_path, e))


def ensure_geojson_present() -> None:
    if os.path.exists(GEOJSON_PATH):
        return
    if not os.path.exists(GEOJSON_ENC):
        raise SystemExit("Missing greece.geojson and greece.geojson.enc not found.")
    _openssl_decrypt(GEOJSON_ENC, GEOJSON_PATH, GEOJSON_PASS)
    if not os.path.exists(GEOJSON_PATH):
        raise SystemExit("Decryption finished but greece.geojson still missing.")


def ensure_dem_present() -> None:
    if os.path.exists(DEM_PATH):
        return
    if not os.path.exists(ALT_ENC):
        raise SystemExit("Missing DEM and altitude.zip.enc not found.")
    _openssl_decrypt(ALT_ENC, ALT_ZIP, GEOJSON_PASS)
    with zipfile.ZipFile(ALT_ZIP, "r") as zf:
        zf.extractall(BASE_DIR)
    if not os.path.exists(DEM_PATH):
        raise SystemExit("Decrypted altitude bundle did not produce GRC_alt.vrt.")
    try:
        os.remove(ALT_ZIP)
    except Exception:
        pass


# =========================
# TEXT HELPERS
# =========================

def safe_name_from_row(r, prefer_col: str = "citygr") -> str:
    v = None
    if prefer_col in r and pd.notna(r[prefer_col]):
        v = str(r[prefer_col]).strip()
    if not v or v.lower() == "nan":
        if "webcode" in r and pd.notna(r["webcode"]):
            v = str(r["webcode"]).strip()
    if not v or v.lower() == "nan":
        v = "–"
    return v




def stamp_text(athens_now: datetime) -> str:
    ts = athens_now.strftime("%Y-%m-%d %H:%M %Z")
    if not BRAND_NAME:
        raise SystemExit("BRAND_NAME is not set.")
    return f"Δημιουργήθηκε για το {BRAND_NAME}\n" + ts


# =========================
# IO
# =========================
def fetch_weathernow_text(url: str) -> str:
    if not url:
        raise SystemExit("CURRENTWEATHER_URL is not set.")
    last_exc = None

    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 415:
                ct = r.headers.get("Content-Type", "")
                print(f"🌧️ 415 Unsupported Media Type (Content-Type={ct})")
                print("🌧️ First 200 bytes of response:", r.text[:200].replace("\n", " "))
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"🌧️ Attempt {i+1} failed: {e}")
            time.sleep(DELAY)

    raise SystemExit(last_exc)


def read_tabbed_df(text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text), sep="\t")
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
# NUMERIC / GEO HELPERS
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

    zi_ok = np.full_like(num, np.nan, dtype=float)
    good = den > 0
    zi_ok[good] = num[good] / den[good]
    zi[ok_pts] = zi_ok[ok_pts]
    return zi.reshape(xi.shape)


def sample_dem_robust(lons, lats, dem_path: str) -> np.ndarray:
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found at: {dem_path}")

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    with rasterio.open(dem_path) as src:
        nodata = src.nodata

        def _sample_once(xs, ys):
            samples = list(src.sample(zip(xs, ys)))
            elev = np.array([s[0] for s in samples], dtype=float)
            if nodata is not None:
                elev = np.where(elev == nodata, np.nan, elev)
            elev = np.where(elev < -100, np.nan, elev)
            return elev

        elev = _sample_once(lons, lats)

        jit = np.array([0.0, 0.001, -0.001, 0.002, -0.002], dtype=float)
        need = ~np.isfinite(elev)
        if np.any(need):
            for dx in jit:
                for dy in jit:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    if not np.any(need):
                        break
                    elev_try = _sample_once(lons[need] + dx, lats[need] + dy)
                    ok = np.isfinite(elev_try)
                    elev_idx = np.where(need)[0]
                    elev[elev_idx[ok]] = elev_try[ok]
                    need = ~np.isfinite(elev)

        elev = np.where(np.isfinite(elev), elev, 0.0)
        return elev


def build_geo_mask(grid_x, grid_y, greece_gdf) -> np.ndarray:
    if hasattr(greece_gdf.geometry, "union_all"):
        boundary = greece_gdf.geometry.union_all()
    else:
        boundary = greece_gdf.unary_union

    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(grid_x.ravel(), grid_y.ravel()),
        crs=greece_gdf.crs
    )
    return points.geometry.within(boundary).values.reshape(grid_x.shape)


def build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_deg=1.5) -> np.ndarray:
    tree = cKDTree(np.c_[st_lons, st_lats])
    distances, _ = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()])
    return distances.reshape(grid_x.shape) <= max_deg


def build_latitude_weighted_cell_area_km2(grid_y) -> np.ndarray:
    """Approximate surface area of each lon/lat grid cell in square kilometres."""
    earth_radius_km = 6371.0088
    dlon_rad = np.deg2rad((GRID_LON_MAX - GRID_LON_MIN) / max(GRID_N - 1, 1))
    dlat_rad = np.deg2rad((GRID_LAT_MAX - GRID_LAT_MIN) / max(GRID_N - 1, 1))

    return (
        (earth_radius_km ** 2)
        * dlon_rad
        * dlat_rad
        * np.cos(np.deg2rad(grid_y))
    )


def estimate_local_lapse_rates(st_lons, st_lats, st_temp, st_elev,
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


# =========================
# PLOTTING HELPERS
# =========================
def save_stable(fig, out_dir: str, out_name: str):
    os.makedirs(out_dir, exist_ok=True)

    main_path = os.path.join(out_dir, out_name)
    fig.savefig(main_path, dpi=300, bbox_inches="tight")
    return main_path
    
def ftps_connect_with_retries(host, user, passwd, attempts=6, base_sleep=5, timeout=60):
    """
    Retries FTPS connect/login with exponential backoff.
    Returns a logged-in FTP_TLS session in passive mode with PROT P.
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

def upload_to_ftp(local_file: str, remote_name: str = None, timeout: int = 60) -> bool:
    """
    Uploads a single file in its own FTPS session.
    Retries the CONNECTION (not the STOR), to match rainintensityall.py behavior.
    """
    if not (FTP_HOST and FTP_USER and FTP_PASS):
        return False
    if not local_file or not os.path.exists(local_file):
        print(f"⚠️ Skip upload (missing): {local_file}")
        return False
    if remote_name is None:
        remote_name = os.path.basename(local_file)
    ftps = None
    try:
        ftps = ftps_connect_with_retries(
            FTP_HOST, FTP_USER, FTP_PASS,
            attempts=6, base_sleep=5, timeout=timeout
        )
        with open(local_file, "rb") as f:
            ftps.storbinary("STOR " + remote_name, f)
        print(f"📤 Uploaded: {remote_name}")
        return True
    except Exception as e:
        print(f"⚠️ Upload failed for {remote_name}: {e}")
        return False
    finally:
        try:
            if ftps is not None:
                ftps.quit()
        except Exception:
            try:
                if ftps is not None:
                    ftps.close()
            except Exception:
                pass

def upload_all_to_ftp(files_to_upload):
    """
    Uploads each file in its OWN session, with per-file try/except,
    so one stuck transfer cannot kill the others.
    Adds a small pause between files so the FTPS server can reset.
    """
    if not (FTP_HOST and FTP_USER and FTP_PASS):
        return
    for i, (local_file, remote_name) in enumerate(files_to_upload):
        if i > 0:
            time.sleep(2)
        try:
            upload_to_ftp(local_file, remote_name)
        except Exception as e:
            print(f"⚠️ FTP upload failed for {remote_name}: {e}")

def add_top5_box(ax, title: str, lines: list, x0=0.99, y0=0.98, font_size=11, two_col_font_size=10.5, force_one_col=False, title_font_size=12):
    header = ax.text(
        x0, y0, title,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=title_font_size, color="black",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.35"),
        zorder=10
    )
    try:
        header.set_underline(True)
    except Exception:
        pass

    if not lines:
        return

    max_len = max(len(s) for s in lines)
    use_two_cols = (max_len > 44) and (not force_one_col)

    if not use_two_cols:
        ax.text(
            x0, y0 - 0.05, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=font_size, color="black",
            bbox=dict(facecolor=(1, 1, 1, 0.0), edgecolor="none", boxstyle="round,pad=0.25"),
            zorder=10
        )
    else:
        left = "\n".join(lines[:3])
        right = "\n".join(lines[3:])
        ax.text(
            0.60, y0 - 0.05, left,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=two_col_font_size, color="black",
            bbox=dict(facecolor=(1, 1, 1, 0.0), edgecolor="none", boxstyle="round,pad=0.25"),
            zorder=10
        )
        ax.text(
            x0, y0 - 0.05, right,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=two_col_font_size, color="black",
            bbox=dict(facecolor=(1, 1, 1, 0.0), edgecolor="none", boxstyle="round,pad=0.25"),
            zorder=10
        )


def draw_rank_markers(ax, df5: pd.DataFrame, lon_col="Longitude", lat_col="Latitude"):
    for rank, (_, r) in enumerate(df5.iterrows(), start=1):
        try:
            lon = float(r[lon_col])
            lat = float(r[lat_col])

            ax.scatter([lon], [lat], s=70, facecolors="none", edgecolors="black",
                       linewidths=1.2, zorder=12)

            t = ax.text(lon, lat, str(rank), ha="center", va="center",
                        fontsize=9, color="black", zorder=13)
            t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        except Exception:
            continue


def add_temp_contours_wgs(ax, grid_x, grid_y, field, special_levels=None):
    """
    Draw ordinary 3°C contours without labels.
    Draw selected prominent contours more strongly and label only those.
    """
    if special_levels is None:
        special_levels = []

    levels = np.arange(-30, 49, 3, dtype=float)
    thin_levels = [
        lv for lv in levels
        if not any(np.isclose(lv, special) for special in special_levels)
    ]

    try:
        ax.contour(
            grid_x, grid_y, field,
            levels=thin_levels,
            colors="black",
            linewidths=0.6,
            alpha=0.70
        )
    except Exception:
        pass

    cs_special = None

    try:
        if special_levels:
            cs_special = ax.contour(
                grid_x, grid_y, field,
                levels=special_levels,
                colors="black",
                linewidths=1.3,
                alpha=0.95
            )
    except Exception:
        cs_special = None

    if cs_special is not None:
        try:
            texts = ax.clabel(
                cs_special,
                levels=cs_special.levels[:],
                inline=True,
                inline_spacing=2,
                fmt="%d",
                fontsize=5
            )

            for t in texts:
                t.set_rotation(0)
                t.set_rotation_mode("anchor")
                t.set_path_effects([
                    pe.withStroke(linewidth=2.0, foreground="white")
                ])
        except Exception:
            pass


def add_temp_contours_egsa(ax, grid_x, grid_y, field):
    levels = np.arange(-30, 46, 3, dtype=float)
    thin_levels = [lv for lv in levels if abs(lv) > 1e-9]

    cs_thin = None
    cs_zero = None

    try:
        cs_thin = ax.contour(
            grid_x, grid_y, field,
            levels=thin_levels,
            colors="black",
            linewidths=0.6,
            alpha=0.70
        )
    except Exception:
        cs_thin = None

    try:
        cs_zero = ax.contour(
            grid_x, grid_y, field,
            levels=[0.0],
            colors="black",
            linewidths=1.3,
            alpha=0.95
        )
    except Exception:
        cs_zero = None

    if cs_thin is not None:
        try:
            texts = ax.clabel(
                cs_thin,
                levels=cs_thin.levels[:],
                inline=True,
                inline_spacing=2,
                fmt="%d",
                fontsize=7
            )
            for t in texts:
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        except Exception:
            pass

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


def temp_colorbar_national(ax, img):
    ticks = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    cbar = plt.colorbar(img, ax=ax, orientation="vertical", extend="both")
    cbar.set_ticks(ticks)
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)
    return cbar


def temp_colorbar_regional(fig, ax, img):
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", extend="both",
                        fraction=0.035, pad=0.02)
    cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)
    return cbar


def rain_cmap_norm():
    cmap = ListedColormap([
        "#ffffff", "#e3f2fd", "#90caf9", "#64b5f6", "#42a5f5",
        "#1e88e5", "#6a1b9a", "#b71c1c", "#d32f2f", "#fb8c00", "#fdd835"
    ])
    bounds = [0, 0.1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 1000]
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    return cmap, norm, bounds


# =========================
# DATA PREP
# =========================
def prepare_base_data(data: pd.DataFrame) -> pd.DataFrame:
    for col in ["Latitude", "Longitude", "TodayRain", "TMin", "TMax", "TNow",
                "RHNow", "Baronow", "WindDirNow", "WindSpeedNow", "RainIntensity"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data[(data["Latitude"].notna()) & (data["Longitude"].notna())]
    data = data[(data["Latitude"] != 0) & (data["Longitude"] != 0)]
    data = data[data["Longitude"] <= 30]

    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    if getattr(data["Datetime"].dt, "tz", None) is None:
        data["Datetime"] = data["Datetime"].dt.tz_localize("Europe/Athens", nonexistent="shift_forward")
    else:
        data["Datetime"] = data["Datetime"].dt.tz_convert("Europe/Athens")

    return data


def prepare_today_data(data: pd.DataFrame, athens_now: datetime) -> pd.DataFrame:
    today_start = athens_now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_data = data[data["Datetime"] >= today_start].copy()

    if "webcode" in today_data.columns:
        wc = (
            today_data["webcode"].astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("ï»¿", "", regex=False)
            .str.strip()
            .str.lower()
        )

        exclude_all = {w.strip().lower() for w in EXCLUDE_ALL_WEBCODES}

        mask = (
            ~wc.isin(exclude_all)
        )
        today_data = today_data[mask].copy()

    return today_data

def prepare_rain_data(today_data: pd.DataFrame, on_date) -> pd.DataFrame:
    rr = apply_family_exclusions(today_data, "precip", on_date, EXCLUSION_RULES)

    if "webcode" in today_data.columns:
        src = (
            today_data["webcode"].astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("ï»¿", "", regex=False)
            .str.strip()
        )
        kept = (
            rr["webcode"].astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("ï»¿", "", regex=False)
            .str.strip()
        )
        removed = sorted(set(src.str.casefold()) - set(kept.str.casefold()))
        if removed:
            print("🌧️ Excluding from TodayRain:", removed)

    return rr


def prepare_temp_data(today_data: pd.DataFrame, on_date) -> pd.DataFrame:
    tt0 = apply_family_exclusions(today_data, "temperature", on_date, EXCLUSION_RULES)

    if "webcode" in today_data.columns:
        src = (
            today_data["webcode"].astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("ï»¿", "", regex=False)
            .str.strip()
        )
        kept = (
            tt0["webcode"].astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("ï»¿", "", regex=False)
            .str.strip()
        )
        removed = sorted(set(src.str.casefold()) - set(kept.str.casefold()))
        if removed:
            print("🌡️ Excluding from temperature maps:", removed)

    return tt0

# =========================
# NATIONAL MAPS
# =========================
def make_todayrain_map_national(df, greece_gdf, grid_x, grid_y, geo_mask,
                                cell_area_km2, out_dir, athens_now):
    if "TodayRain" not in df.columns:
        print("❌ TodayRain missing.")
        return (None, None)

    rr = df.copy()
    rr["TodayRain"] = pd.to_numeric(rr["TodayRain"], errors="coerce")
    rr.dropna(subset=["TodayRain", "Latitude", "Longitude"], inplace=True)

    rr_map = rr.copy()
    rr_pos = rr_map[rr_map["TodayRain"] > 0].copy()

    if rr_map.empty:
        print("No valid TodayRain data.")
        return (None, None)

    st_lats = rr_map["Latitude"].to_numpy(dtype=float)
    st_lons = rr_map["Longitude"].to_numpy(dtype=float)
    vals = rr_map["TodayRain"].to_numpy(dtype=float)

    grid_val = idw_fast(st_lons, st_lats, vals, grid_x, grid_y,
                        k=8, power=2, max_distance=1.0, min_neighbors=3)

    dist_mask = build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_deg=1.5)
    final_mask = geo_mask & dist_mask

    out = np.full(grid_x.shape, np.nan)
    out[final_mask] = grid_val[final_mask]

    mapped_mask = final_mask & np.isfinite(out)
    mapped_area_km2 = float(np.sum(cell_area_km2[mapped_mask]))
    coverage_pct = 100.0 * mapped_area_km2 / GREECE_RASTERIZED_LAND_AREA_KM2

    rain_area_mask = mapped_mask & (out >= 0.1)
    rain_area_km2 = float(np.sum(cell_area_km2[rain_area_mask]))
    rain_area_pct = 100.0 * rain_area_km2 / GREECE_RASTERIZED_LAND_AREA_KM2

    print(f"ℹ️ National rain-map coverage: {mapped_area_km2:,.0f} km² ({coverage_pct:.1f}% of Greece)")
    print(f"ℹ️ Area with precipitation >=0.1 mm: {rain_area_pct:.1f}% of Greece")

    cmap, norm, bounds = rain_cmap_norm()

    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX),
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=0.9
    )

    greece_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)

    try:
        ax.contour(grid_x, grid_y, out,
                   levels=[0.2, 5, 10, 20, 30, 50, 75, 100, 150, 200],
                   colors="black", linewidths=1)
    except Exception:
        pass

    cbar = plt.colorbar(img, ax=ax, orientation="vertical", boundaries=bounds, extend="max")
    cbar.set_ticks([0, 0.1, 0.2, 5, 10, 20, 30, 50, 75, 100, 150, 200])
    cbar.set_label("Σωρευτικός υετός (mm)", fontsize=12)

    ax.set_title("Υπολογισμ. σωρευτικός υετός (από τα μεσάνυχτα)", fontsize=16)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)

    rain_text = (
        "Ποσοστό έκτασης επικράτειας\n"
        "με καταγεγραμμένο υετό ≥0,1 mm:\n"
        f"{rain_area_pct:.1f}%"
    )
    ax.text(
        0.01, 0.985, rain_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.2,
        color="black",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.2"),
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")]
    )

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=10, color="black", ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    if rr_pos.empty:
        add_top5_box(
            ax,
            "Υετός σήμερα",
            ["Δεν υπάρχει καταγεγραμμένος υετός σήμερα."],
            x0=0.99,
            y0=0.98
        )
    else:
        rr_pos["__name"] = rr_pos.apply(lambda r: safe_name_from_row(r, "citygr"), axis=1)
        wet = rr_pos.sort_values("TodayRain", ascending=False).head(TOP_RAIN_N)

        lines = []
        for rank, (_, r) in enumerate(wet.iterrows(), start=1):
            nm = shorten_for_box(r["__name"], max_chars=TOPBOX_NAME_MAX)
            val_txt = f"{float(r['TodayRain']):.1f}".replace(".", ",")
            lines.append(f"{rank}. {nm}: {val_txt} mm")

        add_top5_box(
            ax,
            f"Υψηλότερες {len(wet)} τιμές υετού",
            lines,
            x0=0.99,
            y0=0.98,
            font_size=8.2,
            two_col_font_size=8.2,
            force_one_col=True,
            title_font_size=10
        )
        ### draw_rank_markers(ax, wet, lon_col="Longitude", lat_col="Latitude") ### δείχνει τα τοπ 15 πάνω στο χάρτη

    main_path = save_stable(fig, out_dir, "todayrain.png")
    plt.close(fig)

    print(f"✅ Saved: {main_path}")
    return main_path, None


def make_temp_map_national(df, greece_gdf, grid_x, grid_y, geo_mask,
                           grid_elev, cell_area_km2, out_dir, athens_now, dem_path,
                           var_col, stable_name, title, box_title, sort_ascending,
                           extra_without_topbox_name=None):
    if var_col not in df.columns:
        print(f"❌ {var_col} missing.")
        return (None, None)

    tt0 = df.copy()
    tt0[var_col] = pd.to_numeric(tt0[var_col], errors="coerce")
    tt0.dropna(subset=[var_col, "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0[var_col].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]
    tt0 = tt0[(tt0[var_col] <= TEMP_HARD_MAX)]
    tt0 = tt0[(tt0[var_col] >= TEMP_HARD_MIN)]

    if tt0.empty:
        print(f"No valid {var_col} data after hard cap.")
        return (None, None)

    tt_rank = tt0.copy()

    st_lats = tt0["Latitude"].to_numpy(dtype=float)
    st_lons = tt0["Longitude"].to_numpy(dtype=float)
    st_temp = tt0[var_col].to_numpy(dtype=float)

    st_elev = sample_dem_robust(st_lons, st_lats, dem_path)

    ok = np.isfinite(st_temp) & np.isfinite(st_lons) & np.isfinite(st_lats) & np.isfinite(st_elev)
    st_lats = st_lats[ok]
    st_lons = st_lons[ok]
    st_temp = st_temp[ok]
    st_elev = st_elev[ok]

    if len(st_temp) < 5:
        print(f"❌ Too few stations with valid {var_col} for interpolation.")
        return (None, None)

    st_lapse = estimate_local_lapse_rates(
        st_lons, st_lats, st_temp, st_elev,
        k=12, max_deg=1.2,
        default_lapse=LAPSE_DEFAULT,
        clip_min=LAPSE_MIN, clip_max=LAPSE_MAX
    )

    st_t0 = st_temp - (st_lapse * st_elev)

    t0_grid = idw_fast(st_lons, st_lats, st_t0, grid_x, grid_y, k=8, power=2,
                       max_distance=1.2, min_neighbors=3)
    lapse_grid = idw_fast(st_lons, st_lats, st_lapse, grid_x, grid_y, k=8, power=2,
                          max_distance=1.2, min_neighbors=3)

    temp_grid = t0_grid + (lapse_grid * grid_elev)

    dist_mask = build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_deg=1.5)
    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x.shape, np.nan)
    out[final_mask] = temp_grid[final_mask]

    interp_min = None
    interp_max = None
    try:
        if np.any(np.isfinite(out)):
            interp_min = float(np.nanmin(out))
            interp_max = float(np.nanmax(out))
    except Exception:
        interp_min, interp_max = None, None

    mapped_mask = final_mask & np.isfinite(out)
    mapped_area_km2 = float(np.sum(cell_area_km2[mapped_mask]))
    coverage_pct = 100.0 * mapped_area_km2 / GREECE_RASTERIZED_LAND_AREA_KM2

    def pct_area_above(threshold_c: float) -> float:
        threshold_mask = mapped_mask & (out > threshold_c)
        area_km2 = float(np.sum(cell_area_km2[threshold_mask]))
        return 100.0 * area_km2 / GREECE_RASTERIZED_LAND_AREA_KM2

    def pct_area_below(threshold_c: float) -> float:
        threshold_mask = mapped_mask & (out < threshold_c)
        area_km2 = float(np.sum(cell_area_km2[threshold_mask]))
        return 100.0 * area_km2 / GREECE_RASTERIZED_LAND_AREA_KM2

    if var_col == "TMin":
        pct_below_0 = pct_area_below(0.0)

        # Prominent contour lines for the Tmin map.
        special_levels = [0.0, 10.0, 20.0, 25.0, 30.0]

        stats_text = (
            "Ποσοστό έκτασης επικράτειας βάσει παρεμβολής:\n"
            f"<0°C: {pct_below_0:.1f}%".replace(".", ",")
        )

        print(f"ℹ️ {var_col} area <0°C: {pct_below_0:.1f}% of Greece")

    else:
        pct_above_30 = pct_area_above(30.0)
        pct_above_37 = pct_area_above(37.0)
        pct_above_40 = pct_area_above(40.0)

        # Prominent contour lines for the Tmax map.
        special_levels = [0.0, 10.0, 20.0, 25.0, 30.0, 37.0, 40.0]

        def format_pct_with_observed_floor(pct_value: float, threshold_c: float) -> str:
            """
            Show <0,1% when:
              - the interpolated percentage is positive but below 0.1%, or
              - interpolation gives 0.0%, but at least one valid station
                has actually exceeded the threshold.
            """
            observed_exceedance = bool((tt_rank[var_col] > threshold_c).any())

            if pct_value < 0.1 and (pct_value > 0.0 or observed_exceedance):
                return "<0,1%"

            return f"{pct_value:.1f}%".replace(".", ",")

        pct_above_30_text = format_pct_with_observed_floor(pct_above_30, 30.0)
        pct_above_37_text = format_pct_with_observed_floor(pct_above_37, 37.0)
        pct_above_40_text = format_pct_with_observed_floor(pct_above_40, 40.0)

        stats_text = (
            "Ποσοστό έκτασης επικράτειας βάσει παρεμβολής:\n"
            f">30°C: {pct_above_30_text}\n"
            f">37°C: {pct_above_37_text}\n"
            f">40°C: {pct_above_40_text}"
        )

        print(f"ℹ️ {var_col} area >30°C: {pct_above_30:.1f}% of Greece")
        print(f"ℹ️ {var_col} area >37°C: {pct_above_37:.1f}% of Greece")
        print(f"ℹ️ {var_col} area >40°C: {pct_above_40:.1f}% of Greece")

    print(f"ℹ️ {var_col} interpolation coverage: {mapped_area_km2:,.0f} km² ({coverage_pct:.1f}% of Greece)")

    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(GRID_LON_MIN, GRID_LON_MAX, GRID_LAT_MIN, GRID_LAT_MAX),
        origin="lower",
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        alpha=0.95
    )

    greece_gdf.boundary.plot(ax=ax, color="black", linewidth=0.6)
    add_temp_contours_wgs(ax, grid_x, grid_y, out, special_levels=special_levels)
    temp_colorbar_national(ax, img)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)

    if interp_min is not None and interp_max is not None:
        display_min = interp_min
        display_max = interp_max

        # For Tmax, show the highest actual valid station reading
        # when it exceeds the highest interpolated land-grid value.
        if var_col == "TMax" and not tt_rank.empty:
            actual_station_max = float(tt_rank[var_col].max())

            if np.isfinite(actual_station_max):
                display_max = max(display_max, actual_station_max)

        left_text = (
            "Εύρος θερμοκρασιών στην ξηρά:\n"
            f"{display_min:.1f} έως {display_max:.1f}°C\n\n"
            + stats_text
        ).replace(".", ",")
        ax.text(
            0.01, 0.985, left_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8.2,
            color="black",
            bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.2"),
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")]
        )

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=10, color="black", ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    # Optionally save an additional copy before adding the top-15 box.
    # Used for the national Tmax map only.
    extra_without_topbox_path = None

    if extra_without_topbox_name:
        extra_without_topbox_path = save_stable(
            fig,
            out_dir,
            extra_without_topbox_name
        )

        print(f"✅ Saved without top-15 box: {extra_without_topbox_path}")
        
    tt_rank["__name"] = tt_rank.apply(lambda r: safe_name_from_row(r, "citygr"), axis=1)
    rank_df = tt_rank.sort_values(var_col, ascending=sort_ascending).head(15)

    lines = []
    for rank, (_, r) in enumerate(rank_df.iterrows(), start=1):
        nm = shorten_for_box(r["__name"], max_chars=TOPBOX_NAME_MAX)
        val_txt = f"{float(r[var_col]):.1f}".replace(".", ",")
        lines.append(f"{rank}. {nm}: {val_txt}°C")

    add_top5_box(ax, box_title, lines, x0=0.99, y0=0.98, font_size=8.2, two_col_font_size=8.2, force_one_col=True)
    ### draw_rank_markers(ax, rank_df, lon_col="Longitude", lat_col="Latitude") ### δείχνει τα τοπ 15 πάνω στο χάρτη

    main_path = save_stable(fig, out_dir, stable_name)
    plt.close(fig)

    print(f"✅ Saved: {main_path}")
    return main_path, extra_without_topbox_path


# =========================
# REGIONAL MAPS
# =========================
def region_bbox_to_egsa(region):
    corners_lon = [region["lon_min"], region["lon_min"], region["lon_max"], region["lon_max"]]
    corners_lat = [region["lat_min"], region["lat_max"], region["lat_min"], region["lat_max"]]
    cx, cy = WGS_TO_EGSA.transform(corners_lon, corners_lat)
    x_min, x_max = float(np.min(cx)), float(np.max(cx))
    y_min, y_max = float(np.min(cy)), float(np.max(cy))
    return x_min, x_max, y_min, y_max

def prepare_region_contexts(greece_gdf_wgs):
    greece_egsa = greece_gdf_wgs.to_crs(CRS_EGSA87)
    contexts = {}

    for region in REGIONS:
        x_min, x_max, y_min, y_max = region_bbox_to_egsa(region)

        grid_x_m, grid_y_m = np.meshgrid(
            np.linspace(x_min, x_max, region["n"]),
            np.linspace(y_min, y_max, region["n"])
        )

        greece_clip = greece_egsa.cx[x_min:x_max, y_min:y_max].copy()

        if hasattr(greece_clip.geometry, "union_all"):
            boundary = greece_clip.geometry.union_all()
        else:
            boundary = greece_clip.geometry.unary_union

        grid_pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(grid_x_m.ravel(), grid_y_m.ravel()),
            crs=CRS_EGSA87
        )
        geo_mask = grid_pts.geometry.within(boundary).values.reshape(grid_x_m.shape)

        contexts[region["key"]] = {
            "region": region,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "grid_x_m": grid_x_m,
            "grid_y_m": grid_y_m,
            "greece_clip": greece_clip,
            "geo_mask": geo_mask,
        }

    return contexts

def setup_regional_axes(ax, x_min, x_max, y_min, y_max):
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


def make_todayrain_region_egsa(df, ctx, out_dir, athens_now):
    region = ctx["region"]

    if "TodayRain" not in df.columns:
        print("❌ TodayRain missing.")
        return (None, None)

    rr = df.copy()
    rr["TodayRain"] = pd.to_numeric(rr["TodayRain"], errors="coerce")
    rr.dropna(subset=["TodayRain", "Latitude", "Longitude"], inplace=True)

    if rr.empty:
        print(f"❌ No valid TodayRain data for {region['key']}.")
        return (None, None)
    x_min = ctx["x_min"]
    x_max = ctx["x_max"]
    y_min = ctx["y_min"]
    y_max = ctx["y_max"]
    grid_x_m = ctx["grid_x_m"]
    grid_y_m = ctx["grid_y_m"]
    greece_clip = ctx["greece_clip"]
    geo_mask = ctx["geo_mask"]

    st_lon = rr["Longitude"].to_numpy(dtype=float)
    st_lat = rr["Latitude"].to_numpy(dtype=float)
    st_val = rr["TodayRain"].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    buf = 200_000.0
    near = (st_x >= (x_min - buf)) & (st_x <= (x_max + buf)) & (st_y >= (y_min - buf)) & (st_y <= (y_max + buf))
    st_x = st_x[near]
    st_y = st_y[near]
    st_val = st_val[near]

    if len(st_val) < 5:
        print(f"❌ Too few nearby rain stations for {region['key']}.")
        return (None, None)

    grid_val = idw_fast(
        st_x, st_y, st_val, grid_x_m, grid_y_m,
        k=AT_IDW_K, power=AT_IDW_POWER,
        max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS
    )

    tree = cKDTree(np.c_[st_x, st_y])
    d, _ = tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    dist_mask = d.reshape(grid_x_m.shape) <= AT_DISTANCE_MASK_M

    final_mask = geo_mask & dist_mask

    out = np.full(grid_x_m.shape, np.nan)
    out[final_mask] = grid_val[final_mask]

    cmap, norm, bounds = rain_cmap_norm()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    img = ax.imshow(
        ma.masked_invalid(out),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=0.9
    )

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)
    setup_regional_axes(ax, x_min, x_max, y_min, y_max)

    try:
        ax.contour(
            grid_x_m, grid_y_m, out,
            levels=[0.2, 5, 10, 20, 30, 50, 75, 100, 150, 200],
            colors="black", linewidths=1.0
        )
    except Exception:
        pass

    cbar = fig.colorbar(img, ax=ax, orientation="vertical", boundaries=bounds, extend="max",
                        fraction=0.035, pad=0.02)
    cbar.set_ticks([0, 0.1, 0.2, 5, 10, 20, 30, 50, 75, 100, 150, 200])
    cbar.set_label("Σωρευτικός υετός (mm)", fontsize=12)

    ax.set_title(region["title_rain"], fontsize=16, pad=10)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
    )

    stable_name = f"todayrain_{region['key']}.png"
    main_path = save_stable(fig, out_dir, stable_name)
    plt.close(fig)

    print(f"✅ Saved: {main_path}")
    return main_path, None


def make_temp_region_egsa(df, ctx, out_dir, athens_now, dem_path,
                          var_col, stable_name, title):
    region = ctx["region"]

    if var_col not in df.columns:
        print(f"❌ {var_col} missing.")
        return (None, None)

    tt0 = df.copy()
    tt0[var_col] = pd.to_numeric(tt0[var_col], errors="coerce")
    tt0.dropna(subset=[var_col, "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0[var_col].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]
    tt0 = tt0[(tt0[var_col] <= TEMP_HARD_MAX)]
    tt0 = tt0[(tt0[var_col] >= TEMP_HARD_MIN)]

    if tt0.empty:
        print(f"❌ No valid {var_col} data for {region['key']}.")
        return (None, None)
    x_min = ctx["x_min"]
    x_max = ctx["x_max"]
    y_min = ctx["y_min"]
    y_max = ctx["y_max"]
    grid_x_m = ctx["grid_x_m"]
    grid_y_m = ctx["grid_y_m"]
    greece_clip = ctx["greece_clip"]
    geo_mask = ctx["geo_mask"]

    st_lon = tt0["Longitude"].to_numpy(dtype=float)
    st_lat = tt0["Latitude"].to_numpy(dtype=float)
    st_t = tt0[var_col].to_numpy(dtype=float)

    st_x, st_y = WGS_TO_EGSA.transform(st_lon.tolist(), st_lat.tolist())
    st_x = np.asarray(st_x, dtype=float)
    st_y = np.asarray(st_y, dtype=float)

    buf = 200_000.0
    near = (st_x >= (x_min - buf)) & (st_x <= (x_max + buf)) & (st_y >= (y_min - buf)) & (st_y <= (y_max + buf))
    st_lon = st_lon[near]
    st_lat = st_lat[near]
    st_t = st_t[near]
    st_x = st_x[near]
    st_y = st_y[near]

    if len(st_t) < 8:
        print(f"❌ Too few nearby stations for {var_col} in {region['key']}.")
        return (None, None)

    st_elev = sample_dem_robust(st_lon, st_lat, dem_path)

    ok = np.isfinite(st_t) & np.isfinite(st_x) & np.isfinite(st_y) & np.isfinite(st_elev)
    st_t = st_t[ok]
    st_x = st_x[ok]
    st_y = st_y[ok]
    st_elev = st_elev[ok]

    if len(st_t) < 8:
        print(f"❌ Too few valid stations (after DEM) for {var_col} in {region['key']}.")
        return (None, None)

    st_lapse = estimate_local_lapse_rates_egsa(st_x, st_y, st_t, st_elev)
    st_t0 = st_t - (st_lapse * st_elev)

    t0_grid = idw_fast(
        st_x, st_y, st_t0, grid_x_m, grid_y_m,
        k=AT_IDW_K, power=AT_IDW_POWER,
        max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS
    )

    lapse_grid = idw_fast(
        st_x, st_y, st_lapse, grid_x_m, grid_y_m,
        k=AT_IDW_K, power=AT_IDW_POWER,
        max_distance=AT_MAX_DISTANCE_M, min_neighbors=AT_MIN_NEIGHBORS
    )

    glon, glat = EGSA_TO_WGS.transform(grid_x_m.ravel().tolist(), grid_y_m.ravel().tolist())
    grid_elev = sample_dem_robust(np.array(glon, dtype=float), np.array(glat, dtype=float), dem_path).reshape(grid_x_m.shape)

    t_grid = t0_grid + (lapse_grid * grid_elev)

    tree = cKDTree(np.c_[st_x, st_y])
    d, _ = tree.query(np.c_[grid_x_m.ravel(), grid_y_m.ravel()])
    dist_mask = d.reshape(grid_x_m.shape) <= AT_DISTANCE_MASK_M

    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x_m.shape, np.nan)
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

    greece_clip.boundary.plot(ax=ax, color="black", linewidth=0.6)

    setup_regional_axes(ax, x_min, x_max, y_min, y_max)
    add_temp_contours_egsa(ax, grid_x_m, grid_y_m, out)
    temp_colorbar_regional(fig, ax, img)

    ax.set_title(title, fontsize=16, pad=10)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=9, color="black",
        ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", boxstyle="round,pad=0.3"),
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
    )

    main_path = save_stable(fig, out_dir, stable_name)
    plt.close(fig)

    print(f"✅ Saved: {main_path}")
    return main_path, None


# =========================
# MAIN
# =========================
def main():
    print("✅ RUNNING FILE:", os.path.abspath(__file__))
    print("✅ FTP enabled:", bool(FTP_HOST and FTP_USER and FTP_PASS))
    print("✅ Shared exclusion rules loaded:", EXCLUSION_RULES.get("version"))

    ensure_geojson_present()
    ensure_dem_present()

    text = fetch_weathernow_text(DATA_URL)
    data = read_tabbed_df(text)

    if "Datetime" not in data.columns:
        print("❌ Datetime column missing. Parsed columns:")
        print(list(data.columns))
        raise SystemExit(1)

    data = prepare_base_data(data)

    athens_now = datetime.now(ZoneInfo("Europe/Athens"))
    today_data = prepare_today_data(data, athens_now)

    if today_data.empty:
        print("❌ No data after midnight filter.")
        return

    greece = gpd.read_file(GEOJSON_PATH)
    if greece.crs is None:
        greece = greece.set_crs(CRS_WGS84)

    grid_x, grid_y = np.meshgrid(
        np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_N),
        np.linspace(GRID_LAT_MIN, GRID_LAT_MAX, GRID_N)
    )
    geo_mask = build_geo_mask(grid_x, grid_y, greece)

    # National static grids, calculated once and reused by all three maps.
    grid_elev = sample_dem_robust(
        grid_x.ravel(),
        grid_y.ravel(),
        DEM_PATH
    ).reshape(grid_x.shape)
    cell_area_km2 = build_latitude_weighted_cell_area_km2(grid_y)

    # -------- Rain --------
    rain_input = prepare_rain_data(today_data, athens_now.date())
    rain_dir = os.path.join(BASE_DIR, "TodayRainMaps")
    rain_main, _ = make_todayrain_map_national(
        rain_input, greece, grid_x, grid_y, geo_mask,
        cell_area_km2, rain_dir, athens_now
    )

    # -------- Temperature-family input (for both Tmin and Tmax) --------
    temp_input = prepare_temp_data(today_data, athens_now.date())

    # -------- Tmin --------
    tmin_dir = os.path.join(BASE_DIR, "TminMaps")
    tmin_main, _ = make_temp_map_national(
        temp_input, greece, grid_x, grid_y, geo_mask,
        grid_elev, cell_area_km2, tmin_dir, athens_now, DEM_PATH,
        var_col="TMin",
        stable_name="tmin.png",
        title="Ελάχιστη θερμοκρασία (προσαρμογή υψομέτρου)",
        box_title="Ψυχρότερες 15 περιοχές",
        sort_ascending=True
    )


    # -------- Tmax --------
    tmax_input = temp_input
    tmax_dir = os.path.join(BASE_DIR, "TmaxMaps")

    tmax_with_top15, tmax_main = make_temp_map_national(
        tmax_input, greece, grid_x, grid_y, geo_mask,
        grid_elev, cell_area_km2, tmax_dir, athens_now, DEM_PATH,
        var_col="TMax",
        stable_name="tmax_with_top15.png",
        title="Μέγιστη θερμοκρασία (προσαρμογή υψομέτρου)",
        box_title="Θερμότερες 15 περιοχές",
        sort_ascending=False,
        extra_without_topbox_name="tmax.png"
    )


    # -------- Upload stable filenames only --------
    uploads = [
        (rain_main, "todayrain.png"),
        (tmin_main, "tmin.png"),
        (tmax_main, "tmax.png"),
        (tmax_with_top15, "tmax_with_top15.png"),
    ]

    try:
        upload_all_to_ftp(uploads)
    except Exception as e:
        print(f"⚠️ Batch FTP upload failed: {e}")


if __name__ == "__main__":
    main()
