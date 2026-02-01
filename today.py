#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# today.py  (GitHub-ready, no secrets)
#
# Produces and uploads (same outputs as before):
#   1) todayrain.png  (TodayRain accumulated since midnight)
#   2) tmin.png       (Altitude-aware Tmin map with spatially varying lapse rate)
#   3) tmax.png       (Altitude-aware Tmax map with spatially varying lapse rate)
#
# Requirements:
#   pip install numpy pandas geopandas matplotlib scipy requests rasterio
#
# Secrets/config are provided via environment variables (GitHub Actions Secrets/Variables):
#   CURRENTWEATHER_URL   -> URL to weathernow.txt (tab-separated)
#   FTP_HOST
#   FTP_USER
#   FTP_PASS
#   GEOJSON_PASS         -> passphrase to decrypt greece.geojson.enc and altitude.zip.enc
#
# Encrypted assets expected in repo root:
#   greece.geojson.enc   -> decrypts to greece.geojson
#   altitude.zip.enc     -> decrypts to altitude.zip -> extracts DEM files incl. GRC_alt.vrt (and its sidecars)

import os
import re
import time
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

from scipy.spatial import cKDTree
import requests
from ftplib import FTP_TLS
import rasterio


# =========================
# CONFIG (no secrets here)
# =========================

EXCLUDE_TMAX_WEBCODES = {"hua_ilion", "hua_argyroupoli"}

BASE_DIR = os.path.abspath(os.path.dirname(__file__) or ".")

# Public repo paths (decrypted/unzipped into repo root)
GEOJSON_PATH = os.path.join(BASE_DIR, "greece.geojson")
DEM_PATH = os.path.join(BASE_DIR, "GRC_alt.vrt")

# Encrypted bundles (repo root)
GEOJSON_ENC = os.path.join(BASE_DIR, "greece.geojson.enc")
ALT_ENC = os.path.join(BASE_DIR, "altitude.zip.enc")
ALT_ZIP = os.path.join(BASE_DIR, "altitude.zip")

# Secrets injected via env
DATA_URL = os.environ.get("CURRENTWEATHER_URL", "").strip()
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()
BRAND_NAME = os.environ.get("BRAND_NAME", "").strip()

GRID_LON_MIN, GRID_LON_MAX = 19.0, 30.0
GRID_LAT_MIN, GRID_LAT_MAX = 34.5, 42.5
GRID_N = 300

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/plain, text/*;q=0.9, */*;q=0.8",
    "Accept-Encoding": "identity",
}
MAX_RETRIES = 5
DELAY = 10
TIMEOUT = 20

SENTINEL_TEMP = -67.8

# How aggressive you want the top-right box to be
TOPBOX_NAME_MAX = 26
TOP_RAIN_N = 10

TEMP_HARD_MIN = -30.0   # optional, keep or remove
TEMP_HARD_MAX = 49.0    # your requirement

# =========================
# SHARED TEMP PALETTE (TMIN + TMAX)  (UNCHANGED)
# =========================
TEMP_VMIN = -25.0
TEMP_VMAX = 45.0


def build_shared_temp_cmap_norm():
    # Goal:
    # - Blue for <= 0°C (0°C is clearly blue, not white)
    # - Positive temps move into blue-green/green, then yellow/orange/red
    # - Very hot temps (>= ~40°C) shift into purple
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
# ENCRYPTED ASSET HANDLING
# =========================
def _openssl_decrypt(enc_path: str, out_path: str, passphrase: str) -> None:
    if not passphrase:
        raise SystemExit("GEOJSON_PASS not set (cannot decrypt encrypted assets).")
    try:
        subprocess.check_call([
            "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
            "-in", enc_path, "-out", out_path,
            "-pass", "pass:" + passphrase
        ])
    except FileNotFoundError:
        raise SystemExit("OpenSSL not found on runner. Install it or decrypt files in CI step.")
    except subprocess.CalledProcessError as e:
        raise SystemExit("OpenSSL decryption failed for %s: %s" % (enc_path, e))


def ensure_geojson_present() -> None:
    if os.path.exists(GEOJSON_PATH):
        return
    if not os.path.exists(GEOJSON_ENC):
        raise SystemExit("Missing greece.geojson and greece.geojson.enc not found in repo root.")
    _openssl_decrypt(GEOJSON_ENC, GEOJSON_PATH, GEOJSON_PASS)
    if not os.path.exists(GEOJSON_PATH):
        raise SystemExit("Decryption finished but greece.geojson still missing. Check paths.")


def ensure_dem_present() -> None:
    # If VRT is already present at repo root, nothing to do
    if os.path.exists(DEM_PATH):
        return

    # Try to decrypt and unzip if encrypted bundle exists
    if not os.path.exists(ALT_ENC):
        raise SystemExit("Missing DEM: %s and altitude.zip.enc not found in repo root." % DEM_PATH)

    _openssl_decrypt(ALT_ENC, ALT_ZIP, GEOJSON_PASS)

    # Unzip into repo root so GRC_alt.vrt and its sidecars land next to this script
    with zipfile.ZipFile(ALT_ZIP, "r") as zf:
        zf.extractall(BASE_DIR)

    # Verify VRT exists
    if not os.path.exists(DEM_PATH):
        raise SystemExit("Decrypted altitude bundle did not produce GRC_alt.vrt at repo root.")

    # Remove plaintext zip
    try:
        os.remove(ALT_ZIP)
    except Exception:
        pass


# =========================
# TEXT HELPERS (TOP BOX)
# =========================
def prettify_station_name(s: str) -> str:
    if s is None:
        return "–"
    s = str(s).strip()
    s = s.replace("Ιερά Μονή", "Ι.Μ.")
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


def shorten_for_box(name: str, max_chars: int = TOPBOX_NAME_MAX) -> str:
    s = prettify_station_name(name)
    if "(" in s and ")" in s:
        base = s.split("(", 1)[0].strip()
        if base:
            s = base
    s = s.replace("«", "").replace("»", "").replace('"', "").replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()
    return ellipsize(s, max_chars=max_chars)


# =========================
# HELPERS
# =========================
def fetch_weathernow_text(url: str) -> str:
    if not url:
        raise SystemExit("CURRENTWEATHER_URL is not set.")
    last_exc = None

    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 415:
                # Show what the server says it wants (often hints at required Accept/content type)
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

    print("❌ All attempts to fetch data failed.")
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


def sample_dem_robust(lons, lats, dem_path: str) -> np.ndarray:
    """
    Robust DEM sampling:
    - sample exact point
    - if NaN/nodata, sample a few tiny jitters around it (helps coastal/border pixels)
    - if still NaN, fallback to 0 m
    """
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


def upload_to_ftp(local_file: str, remote_name: str):
    # Upload only if all credentials are present
    if not (FTP_HOST and FTP_USER and FTP_PASS):
        return

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


def build_geo_mask(grid_x, grid_y, greece_gdf) -> np.ndarray:
    # Keep same behavior, but avoid deprecation if available
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
    return (distances.reshape(grid_x.shape) <= max_deg)


def estimate_local_lapse_rates(st_lons, st_lats, st_temp, st_elev,
                              k=12, max_deg=1.2,
                              default_lapse=-0.0065,
                              clip_min=-0.015, clip_max=0.005) -> np.ndarray:
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


def stamp_text(athens_now: datetime) -> str:
    ts = athens_now.strftime("%Y-%m-%d %H:%M %Z")

    if not BRAND_NAME:
        raise SystemExit("BRAND_NAME is not set (required for stamp text).")

    return f"Δημιουργήθηκε για το {BRAND_NAME}\n" + ts

def add_top5_box(ax, title: str, lines: list, x0=0.99, y0=0.98):
    header = ax.text(
        x0, y0, title,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, color="black",
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
    use_two_cols = max_len > 44

    if not use_two_cols:
        ax.text(
            x0, y0 - 0.05, "\n".join(lines),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="black",
            bbox=dict(facecolor=(1, 1, 1, 0.0), edgecolor="none", boxstyle="round,pad=0.25"),
            zorder=10
        )
    else:
        left = "\n".join(lines[:3])
        right = "\n".join(lines[3:])
        ax.text(
            0.60, y0 - 0.05, left,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, color="black",
            bbox=dict(facecolor=(1, 1, 1, 0.0), edgecolor="none", boxstyle="round,pad=0.25"),
            zorder=10
        )
        ax.text(
            x0, y0 - 0.05, right,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10.5, color="black",
            bbox=dict(facecolor=(1, 1, 1, 0.0), edgecolor="none", boxstyle="round,pad=0.25"),
            zorder=10
        )


def draw_rank_markers(ax, df5: pd.DataFrame, lon_col="Longitude", lat_col="Latitude"):
    for rank, (_, r) in enumerate(df5.iterrows(), start=1):
        try:
            lon = float(r[lon_col])
            lat = float(r[lat_col])

            ax.scatter([lon], [lat], s=90, facecolors="none", edgecolors="black",
                       linewidths=1.2, zorder=12)

            t = ax.text(lon, lat, str(rank), ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black", zorder=13)
            t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
        except Exception:
            continue


# =========================
# MAPS (UNCHANGED OUTPUTS)
# =========================
def make_todayrain_map(df, greece_gdf, grid_x, grid_y, geo_mask, out_dir, athens_now):
    if "TodayRain" not in df.columns:
        print("❌ TodayRain missing.")
        return None

    rr = df.copy()
    rr["TodayRain"] = pd.to_numeric(rr["TodayRain"], errors="coerce")
    rr.dropna(subset=["TodayRain", "Latitude", "Longitude"], inplace=True)

    # Keep a copy for mapping (we want the map even if all stations are 0 mm)
    rr_map = rr.copy()

    # For the Top box/markers, keep only stations with rain > 0
    rr_pos = rr_map[rr_map["TodayRain"] > 0].copy()

    if rr_map.empty:
        print("No valid TodayRain data.")
        return None

    st_lats = rr_map["Latitude"].to_numpy(dtype=float)
    st_lons = rr_map["Longitude"].to_numpy(dtype=float)
    vals = rr_map["TodayRain"].to_numpy(dtype=float)

    grid_val = idw_fast(st_lons, st_lats, vals, grid_x, grid_y,
                        k=8, power=2, max_distance=1.0, min_neighbors=3)

    dist_mask = build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_deg=1.5)
    final_mask = geo_mask & dist_mask

    out = np.full(grid_x.shape, np.nan)
    out[final_mask] = grid_val[final_mask]

    cmap = ListedColormap([
        "#ffffff", "#e3f2fd", "#90caf9", "#64b5f6", "#42a5f5",
        "#1e88e5", "#6a1b9a", "#b71c1c", "#d32f2f", "#fb8c00", "#fdd835"
    ])
    bounds = [0, 0.1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 1000]
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

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

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=10, color="black", ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    # ---------- TOP (rainy stations only) ----------
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
            lines.append(f"{rank}. {nm}: {float(r['TodayRain']):.1f} mm")

        add_top5_box(ax, f"Υψηλότερες {len(wet)} τιμές υετού", lines, x0=0.99, y0=0.98)
        draw_rank_markers(ax, wet, lon_col="Longitude", lat_col="Latitude")

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "todayrain.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved locally: {out_file}")
    return out_file


def _temp_colorbar(ax, img):
    ticks = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    cbar = plt.colorbar(img, ax=ax, orientation="vertical", extend="both")
    cbar.set_ticks(ticks)
    cbar.set_label("Θερμοκρασία (°C)", fontsize=12)
    return cbar


def make_tmin_map(df, greece_gdf, grid_x, grid_y, geo_mask, out_dir, athens_now, dem_path):
    if "TMin" not in df.columns:
        print("❌ TMin missing.")
        return None

    # ---------- 1) numeric cleaning (NO DEM HERE) ----------
    tt0 = df.copy()
    tt0["TMin"] = pd.to_numeric(tt0["TMin"], errors="coerce")
    tt0.dropna(subset=["TMin", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TMin"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]

    # ---- HARD CAP (applies to TOP-5 + interpolation + map) ----
    tt0 = tt0[(tt0["TMin"] <= TEMP_HARD_MAX)]
    tt0 = tt0[(tt0["TMin"] >= TEMP_HARD_MIN)]

    if tt0.empty:
        print("No valid Tmin data after hard cap.")
        return None

    # This is the dataset used for TOP-5: only needs value + location
    tt_rank = tt0.copy()

    # ---------- 2) interpolation dataset (uses DEM) ----------
    st_lats = tt0["Latitude"].to_numpy(dtype=float)
    st_lons = tt0["Longitude"].to_numpy(dtype=float)
    st_tmin = tt0["TMin"].to_numpy(dtype=float)

    st_elev = sample_dem_robust(st_lons, st_lats, dem_path)

    ok = np.isfinite(st_tmin) & np.isfinite(st_lons) & np.isfinite(st_lats) & np.isfinite(st_elev)
    st_lats = st_lats[ok]
    st_lons = st_lons[ok]
    st_tmin = st_tmin[ok]
    st_elev = st_elev[ok]

    if len(st_tmin) < 5:
        print("❌ Too few stations with valid Tmin for interpolation.")
        return None

    st_lapse = estimate_local_lapse_rates(
        st_lons, st_lats, st_tmin, st_elev,
        k=12, max_deg=1.2,
        default_lapse=-0.0065,
        clip_min=-0.015, clip_max=0.005
    )

    st_t0 = st_tmin - (st_lapse * st_elev)

    t0_grid = idw_fast(st_lons, st_lats, st_t0, grid_x, grid_y, k=8, power=2,
                       max_distance=1.2, min_neighbors=3)
    lapse_grid = idw_fast(st_lons, st_lats, st_lapse, grid_x, grid_y, k=8, power=2,
                          max_distance=1.2, min_neighbors=3)

    grid_elev = sample_dem_robust(grid_x.ravel(), grid_y.ravel(), dem_path).reshape(grid_x.shape)
    tmin_grid = t0_grid + (lapse_grid * grid_elev)

    dist_mask = build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_deg=1.5)
    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x.shape, np.nan)
    out[final_mask] = tmin_grid[final_mask]

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

    try:
        ax.contour(grid_x, grid_y, out, levels=[0.0], colors="black", linewidths=1.2)
    except Exception:
        pass

    _temp_colorbar(ax, img)

    ax.set_title("Ελάχιστη θερμοκρασία (προσαρμογή υψομέτρου)", fontsize=16)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=10, color="black", ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    # ---------- TOP-5 from tt_rank (NO DEM INVOLVEMENT) ----------
    tt_rank["__name"] = tt_rank.apply(lambda r: safe_name_from_row(r, "citygr"), axis=1)
    cold = tt_rank.sort_values("TMin", ascending=True).head(5)

    lines = []
    for rank, (_, r) in enumerate(cold.iterrows(), start=1):
        nm = shorten_for_box(r["__name"], max_chars=TOPBOX_NAME_MAX)
        lines.append(f"{rank}. {nm}: {float(r['TMin']):.1f}°C")

    add_top5_box(ax, "Ψυχρότερες 5 περιοχές", lines, x0=0.99, y0=0.98)
    draw_rank_markers(ax, cold, lon_col="Longitude", lat_col="Latitude")

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "tmin.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved locally: {out_file}")
    return out_file


def make_tmax_map(df, greece_gdf, grid_x, grid_y, geo_mask, out_dir, athens_now, dem_path):
    if "TMax" not in df.columns:
        print("❌ TMax missing.")
        return None

    # ---------- 1) exclusion + numeric cleaning (NO DEM HERE) ----------
    tt0 = df.copy()

    if "webcode" in tt0.columns:
        exclude = {str(w).strip().lower() for w in EXCLUDE_TMAX_WEBCODES}
        tt0["webcode_norm"] = (
            tt0["webcode"].astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("ï»¿", "", regex=False)
            .str.strip()
            .str.lower()
        )
        present = sorted(set(tt0.loc[tt0["webcode_norm"].isin(exclude), "webcode_norm"].unique()))
        if present:
            print("🔥 Excluding from Tmax:", present)
        tt0 = tt0[~tt0["webcode_norm"].isin(exclude)].copy()

    tt0["TMax"] = pd.to_numeric(tt0["TMax"], errors="coerce")
    tt0.dropna(subset=["TMax", "Latitude", "Longitude"], inplace=True)
    tt0 = tt0[~np.isclose(tt0["TMax"].to_numpy(dtype=float), SENTINEL_TEMP, atol=1e-6)]

    # ---- HARD CAP (applies to TOP-5 + interpolation + map) ----
    tt0 = tt0[(tt0["TMax"] <= TEMP_HARD_MAX)]
    tt0 = tt0[(tt0["TMax"] >= TEMP_HARD_MIN)]

    if tt0.empty:
        print("No valid Tmax data after hard cap.")
        return None

    # This is the dataset used for TOP-5: only needs value + location
    tt_rank = tt0.copy()

    # ---------- 2) interpolation dataset (uses DEM) ----------
    st_lats = tt0["Latitude"].to_numpy(dtype=float)
    st_lons = tt0["Longitude"].to_numpy(dtype=float)
    st_tmax = tt0["TMax"].to_numpy(dtype=float)

    st_elev = sample_dem_robust(st_lons, st_lats, dem_path)

    ok = np.isfinite(st_tmax) & np.isfinite(st_lons) & np.isfinite(st_lats) & np.isfinite(st_elev)
    st_lats = st_lats[ok]
    st_lons = st_lons[ok]
    st_tmax = st_tmax[ok]
    st_elev = st_elev[ok]

    if len(st_tmax) < 5:
        print("❌ Too few stations with valid Tmax for interpolation.")
        return None

    st_lapse = estimate_local_lapse_rates(
        st_lons, st_lats, st_tmax, st_elev,
        k=12, max_deg=1.2,
        default_lapse=-0.0065,
        clip_min=-0.015, clip_max=0.005
    )

    st_t0 = st_tmax - (st_lapse * st_elev)

    t0_grid = idw_fast(st_lons, st_lats, st_t0, grid_x, grid_y, k=8, power=2,
                       max_distance=1.2, min_neighbors=3)
    lapse_grid = idw_fast(st_lons, st_lats, st_lapse, grid_x, grid_y, k=8, power=2,
                          max_distance=1.2, min_neighbors=3)

    grid_elev = sample_dem_robust(grid_x.ravel(), grid_y.ravel(), dem_path).reshape(grid_x.shape)
    tmax_grid = t0_grid + (lapse_grid * grid_elev)

    dist_mask = build_distance_mask(grid_x, grid_y, st_lons, st_lats, max_deg=1.5)
    final_mask = geo_mask & dist_mask & np.isfinite(grid_elev)

    out = np.full(grid_x.shape, np.nan)
    out[final_mask] = tmax_grid[final_mask]

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

    try:
        ax.contour(grid_x, grid_y, out, levels=[0.0], colors="black", linewidths=1.2)
    except Exception:
        pass

    _temp_colorbar(ax, img)

    ax.set_title("Μέγιστη θερμοκρασία (προσαρμογή υψομέτρου)", fontsize=16)
    ax.set_xlabel("Γεωγρ. μήκος", fontsize=12)
    ax.set_ylabel("Γεωγρ. πλάτος", fontsize=12)

    ax.text(
        0.01, 0.01, stamp_text(athens_now),
        transform=ax.transAxes, fontsize=10, color="black", ha="left", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.3")
    )

    # ---------- TOP-5 from tt_rank (NO DEM INVOLVEMENT) ----------
    tt_rank["__name"] = tt_rank.apply(lambda r: safe_name_from_row(r, "citygr"), axis=1)
    hot = tt_rank.sort_values("TMax", ascending=False).head(5)

    lines = []
    for rank, (_, r) in enumerate(hot.iterrows(), start=1):
        nm = shorten_for_box(r["__name"], max_chars=TOPBOX_NAME_MAX)
        lines.append(f"{rank}. {nm}: {float(r['TMax']):.1f}°C")

    add_top5_box(ax, "Θερμότερες 5 περιοχές", lines, x0=0.99, y0=0.98)
    draw_rank_markers(ax, hot, lon_col="Longitude", lat_col="Latitude")

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "tmax.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved locally: {out_file}")
    return out_file


# =========================
# MAIN (logic unchanged, only secure inputs/paths)
# =========================
def main():
    print("✅ RUNNING FILE:", os.path.abspath(__file__))
    print("✅ EXCLUDE_TMAX_WEBCODES:", EXCLUDE_TMAX_WEBCODES)
    print("✅ FTP enabled:", bool(FTP_HOST and FTP_USER and FTP_PASS))

    ensure_geojson_present()
    ensure_dem_present()

    text = fetch_weathernow_text(DATA_URL)
    data = read_tabbed_df(text)

    if "Datetime" not in data.columns:
        print("❌ Datetime column missing. Parsed columns:")
        print(list(data.columns))
        raise SystemExit(1)

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

    athens_now = datetime.now(ZoneInfo("Europe/Athens"))
    today_start = athens_now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_data = data[data["Datetime"] >= today_start].copy()

    if "webcode" in today_data.columns:
        wc = today_data["webcode"].astype(str)
        mask = (
            ~wc.str.match(r"(?i)^wu_lefkaditi$", na=False) &
            ~wc.str.match(r"(?i)^age_klimamilou$", na=False) &
            ~wc.str.match(r"(?i)^uoi_", na=False)
        )
        today_data = today_data[mask].copy()

    if today_data.empty:
        print("No data after midnight filter.")
        return

    greece = gpd.read_file(GEOJSON_PATH)

    grid_x, grid_y = np.meshgrid(
        np.linspace(GRID_LON_MIN, GRID_LON_MAX, GRID_N),
        np.linspace(GRID_LAT_MIN, GRID_LAT_MAX, GRID_N)
    )
    geo_mask = build_geo_mask(grid_x, grid_y, greece)

    rain_out = make_todayrain_map(
        today_data, greece, grid_x, grid_y, geo_mask,
        out_dir=os.path.join(BASE_DIR, "TodayRainMaps"),
        athens_now=athens_now
    )

    tmin_out = make_tmin_map(
        today_data, greece, grid_x, grid_y, geo_mask,
        out_dir=os.path.join(BASE_DIR, "TminMaps"),
        athens_now=athens_now,
        dem_path=DEM_PATH
    )

    tmax_out = make_tmax_map(
        today_data, greece, grid_x, grid_y, geo_mask,
        out_dir=os.path.join(BASE_DIR, "TmaxMaps"),
        athens_now=athens_now,
        dem_path=DEM_PATH
    )

    for local_path, remote_name in [
        (rain_out, "todayrain.png"),
        (tmin_out, "tmin.png"),
        (tmax_out, "tmax.png"),
    ]:
        if not local_path:
            continue
        try:
            upload_to_ftp(local_path, remote_name)
        except Exception as e:
            print(f"⚠️ FTP upload failed for {remote_name}: {e}")


if __name__ == "__main__":
    main()
