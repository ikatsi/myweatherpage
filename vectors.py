#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build gridded mosquito favourability indices from station TSV + DEM,
write GeoTIFFs for all species, then create and upload PNG maps for:

  - Culex pipiens
  - Aedes albopictus
  - Aedes aegypti
  - Anopheles sacharovi

Designed for GitHub Actions.

Input TSV is downloaded from:

  VECTOR_TSV_BASE_URL/vector_indices_YYYYMMDD.tsv

where VECTOR_TSV_BASE_URL is a GitHub secret, for example:
  https://www.e-kairos.gr/vectors

PNG maps are uploaded to the FTP root unless VECTOR_FTP_TARGET_DIR is set.
"""

import os
import io
import time
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from ftplib import FTP_TLS

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

ATHENS_NOW = datetime.now(ZoneInfo("Europe/Athens"))
RUN_DATE = ATHENS_NOW.strftime("%Y%m%d")

VECTOR_TSV_BASE_URL = os.environ.get("VECTOR_TSV_BASE_URL", "").strip()
if not VECTOR_TSV_BASE_URL:
    raise RuntimeError("VECTOR_TSV_BASE_URL environment variable is not set.")

TSV_NAME = "vector_indices_{}.tsv".format(RUN_DATE)
TSV_URL = "{}/{}?nocache={}".format(
    VECTOR_TSV_BASE_URL.rstrip("/"),
    TSV_NAME,
    ATHENS_NOW.strftime("%Y%m%d%H%M%S")
)

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "vectors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEM_ENV = os.environ.get("VECTOR_DEM_PATH", "").strip()
GEOJSON_ENV = os.environ.get("VECTOR_GEOJSON_PATH", "").strip()

DEM_PATH = Path(DEM_ENV).expanduser() if DEM_ENV else OUTPUT_DIR / "GRC_alt_filled.tif"
GREECE_GEOJSON = Path(GEOJSON_ENV).expanduser() if GEOJSON_ENV else OUTPUT_DIR / "greece.geojson"

FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()
GEOJSON_PASS = os.environ.get("GEOJSON_PASS", "").strip()

FTP_TARGET_DIR = os.environ.get("VECTOR_FTP_TARGET_DIR", "").strip()

GREEK_GRID_CRS = CRS.from_epsg(2100)
WGS84_CRS = CRS.from_epsg(4326)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

SPECIES_LIST = [
    "culex_pipiens",
    "aedes_albopictus",
    "aedes_aegypti",
    "anopheles_sacharovi",
]

SPECIES_LABELS = {
    "culex_pipiens": "Culex pipiens",
    "aedes_albopictus": "Aedes albopictus",
    "aedes_aegypti": "Aedes aegypti",
    "anopheles_sacharovi": "Anopheles sacharovi",
}

METRICS = {
    "dev_index_14": {
        "window_days": 14,
        "output_suffix": "dev_index_14",
        "scale_to_days": True,
    },
    "act_index_7": {
        "window_days": 7,
        "output_suffix": "act_index_7",
        "scale_to_days": True,
    },
}


# -------------------------------------------------------------------
# STARTUP CHECKS
# -------------------------------------------------------------------

def require_env():
    missing = []

    if not FTP_HOST:
        missing.append("FTP_HOST")
    if not FTP_USER:
        missing.append("FTP_USER")
    if not FTP_PASS:
        missing.append("FTP_PASS")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: {}".format(
                ", ".join(missing)
            )
        )


def decrypt_file(in_path, out_path, password):
    if not password:
        raise RuntimeError(
            "GEOJSON_PASS is not set, cannot decrypt {}".format(in_path)
        )

    subprocess.check_call([
        "openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
        "-in", str(in_path),
        "-out", str(out_path),
        "-pass", "pass:" + password
    ])


def ensure_greece_geojson():
    global GREECE_GEOJSON

    if GREECE_GEOJSON.exists():
        print("Using Greece GeoJSON:", GREECE_GEOJSON)
        return

    fallback_plain = BASE_DIR / "greece.geojson"
    if fallback_plain.exists():
        GREECE_GEOJSON = fallback_plain
        print("Using fallback Greece GeoJSON:", GREECE_GEOJSON)
        return

    candidates = [
        OUTPUT_DIR / "greece.geojson.enc",
        BASE_DIR / "greece.geojson.enc",
    ]

    for enc_path in candidates:
        if enc_path.exists():
            out_path = enc_path.with_suffix("")
            print("Decrypting Greece GeoJSON:", enc_path)
            decrypt_file(enc_path, out_path, GEOJSON_PASS)
            GREECE_GEOJSON = out_path
            print("Using decrypted Greece GeoJSON:", GREECE_GEOJSON)
            return

    raise RuntimeError(
        "Greece GeoJSON not found. Expected vectors/greece.geojson, "
        "greece.geojson, vectors/greece.geojson.enc, or greece.geojson.enc."
    )


def ensure_dem():
    global DEM_PATH

    if DEM_PATH.exists():
        print("Using DEM:", DEM_PATH)
        return

    fallback_plain = BASE_DIR / "GRC_alt_filled.tif"
    if fallback_plain.exists():
        DEM_PATH = fallback_plain
        print("Using fallback DEM:", DEM_PATH)
        return

    alt_enc_candidates = [
        OUTPUT_DIR / "altitude.zip.enc",
        BASE_DIR / "altitude.zip.enc",
    ]

    for alt_enc in alt_enc_candidates:
        if alt_enc.exists():
            alt_zip = alt_enc.with_suffix("")
            print("Decrypting altitude bundle:", alt_enc)
            decrypt_file(alt_enc, alt_zip, GEOJSON_PASS)

            extract_dir = alt_enc.parent
            print("Extracting altitude bundle to:", extract_dir)

            with zipfile.ZipFile(str(alt_zip), "r") as zf:
                zf.extractall(str(extract_dir))

            try:
                alt_zip.unlink()
            except Exception:
                pass

            candidates = [
                OUTPUT_DIR / "GRC_alt_filled.tif",
                BASE_DIR / "GRC_alt_filled.tif",
                OUTPUT_DIR / "GRC_alt.vrt",
                BASE_DIR / "GRC_alt.vrt",
            ]

            for cand in candidates:
                if cand.exists():
                    DEM_PATH = cand
                    print("Using extracted DEM:", DEM_PATH)
                    return

    raise RuntimeError(
        "DEM not found. Expected vectors/GRC_alt_filled.tif, "
        "GRC_alt_filled.tif, or an altitude.zip.enc bundle containing the DEM."
    )


# -------------------------------------------------------------------
# DATA DOWNLOAD
# -------------------------------------------------------------------

def download_tsv():
    print("Downloading TSV:", TSV_NAME)

    resp = requests.get(TSV_URL, headers=HEADERS, timeout=45)
    resp.raise_for_status()

    text = resp.text.strip()
    if not text:
        raise RuntimeError("Downloaded TSV is empty.")

    df = pd.read_csv(io.StringIO(text), sep="\t")
    print("Loaded TSV with {} rows from HTTP".format(len(df)))

    return df


# -------------------------------------------------------------------
# RASTER HELPERS
# -------------------------------------------------------------------

def load_dem_and_mask():
    print("Opening DEM:", DEM_PATH)

    with rasterio.open(str(DEM_PATH)) as dem:
        dem_data = dem.read(1).astype(np.float32)
        profile = dem.profile
        transform = dem.transform
        crs = dem.crs
        nodata = dem.nodata

    height, width = dem_data.shape

    if nodata is not None:
        dem_valid = dem_data != nodata
    else:
        dem_valid = np.isfinite(dem_data)

    print("Reading Greece GeoJSON:", GREECE_GEOJSON)
    gdf = gpd.read_file(str(GREECE_GEOJSON))

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    if crs is not None and gdf.crs != crs:
        print("Reprojecting Greece polygons from {} to {}".format(gdf.crs, crs))
        gdf = gdf.to_crs(crs)

    shapes = list(gdf.geometry)
    print("Number of polygon parts:", len(shapes))

    print("Building geometry mask...")
    poly_mask = geometry_mask(
        shapes,
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )

    valid_mask = dem_valid & poly_mask

    return dem_data, profile, transform, crs, nodata, valid_mask


def build_grid_design_matrix(dem_data, transform, valid_mask):
    print("Building design matrix for grid...")

    rows, cols = np.where(valid_mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    elev = dem_data[rows, cols].astype(np.float32)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    X_grid = np.column_stack([
        np.ones_like(elev, dtype=np.float32),
        elev,
        ys,
        xs,
    ])

    print("Grid design matrix shape:", X_grid.shape)

    return X_grid, rows, cols


def sample_dem_at_points(dem_data, transform, nodata, xs, ys):
    src_height, src_width = dem_data.shape
    elev = np.full(xs.shape, np.nan, dtype=np.float32)

    for i, (x, y) in enumerate(zip(xs, ys)):
        col, row = ~transform * (x, y)
        row_i = int(round(row))
        col_i = int(round(col))

        if 0 <= row_i < src_height and 0 <= col_i < src_width:
            val = dem_data[row_i, col_i]
            if nodata is None or val != nodata:
                elev[i] = val

    return elev


def fit_linear_regression(X, y):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    if X_clean.shape[0] < 5:
        print(
            "Too few valid points for regression, n = {}. Skipping.".format(
                X_clean.shape[0]
            )
        )
        return None

    beta, _, _, _ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
    return beta


def write_raster(path, data, profile, nodata):
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="lzw"
    )

    with rasterio.open(str(path), "w", **out_profile) as dst:
        dst.write(data.astype(np.float32), 1)

    print("Wrote", path)


def raster_extent(transform, width, height):
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + transform.a * width
    ymin = ymax + transform.e * height
    return xmin, xmax, ymin, ymax


def reproject_to_greek_grid(src_array, src_transform, src_crs, src_nodata):
    if src_crs is None:
        raise ValueError("Source CRS is None, cannot reproject.")

    transform_dst, width_dst, height_dst = calculate_default_transform(
        src_crs,
        GREEK_GRID_CRS,
        src_array.shape[1],
        src_array.shape[0],
        *rasterio.transform.array_bounds(
            src_array.shape[0],
            src_array.shape[1],
            src_transform
        )
    )

    dst_array = np.empty((height_dst, width_dst), dtype=np.float32)
    dst_array.fill(src_nodata)

    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=transform_dst,
        dst_crs=GREEK_GRID_CRS,
        dst_nodata=src_nodata,
        resampling=Resampling.bilinear,
    )

    return dst_array, transform_dst, src_nodata


# -------------------------------------------------------------------
# MAP CREATION
# -------------------------------------------------------------------

def create_species_maps(species, label, ref_date):
    print("\nCreating PNG maps for {} ({})...".format(label, species))

    dev_tif = OUTPUT_DIR / "{}_dev_good_days_14_{}.tif".format(species, ref_date)
    act_tif = OUTPUT_DIR / "{}_act_good_days_7_{}.tif".format(species, ref_date)

    if not dev_tif.exists() or not act_tif.exists():
        print("Required TIFFs for {} not found, skipping PNG creation.".format(species))
        return []

    with rasterio.open(str(dev_tif)) as src_dev:
        dev_src = src_dev.read(1)
        src_crs = src_dev.crs
        src_transform = src_dev.transform
        src_nodata = src_dev.nodata

    dev_gg, transform_gg, nodata_gg = reproject_to_greek_grid(
        dev_src,
        src_transform,
        src_crs,
        src_nodata
    )

    height_gg, width_gg = dev_gg.shape
    extent_gg = raster_extent(transform_gg, width_gg, height_gg)

    with rasterio.open(str(act_tif)) as src_act:
        act_src = src_act.read(1)
        act_nodata = src_act.nodata
        act_src_transform = src_act.transform
        act_src_crs = src_act.crs

    act_gg = np.empty_like(dev_gg, dtype=np.float32)
    act_gg.fill(act_nodata)

    reproject(
        source=act_src,
        destination=act_gg,
        src_transform=act_src_transform,
        src_crs=act_src_crs,
        src_nodata=act_nodata,
        dst_transform=transform_gg,
        dst_crs=GREEK_GRID_CRS,
        dst_nodata=act_nodata,
        resampling=Resampling.bilinear,
    )

    gdf = gpd.read_file(str(GREECE_GEOJSON))

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf_gg = gdf.to_crs(GREEK_GRID_CRS)

    transformer_to_lonlat = Transformer.from_crs(
        GREEK_GRID_CRS,
        WGS84_CRS,
        always_xy=True
    )

    def setup_axes(title):
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.set_xlim(extent_gg[0], extent_gg[1])
        ax.set_ylim(extent_gg[2], extent_gg[3])
        ax.set_aspect("equal", adjustable="box")

        xticks = np.linspace(extent_gg[0], extent_gg[1], 7)
        yticks = np.linspace(extent_gg[2], extent_gg[3], 7)
        x_c = 0.5 * (extent_gg[0] + extent_gg[1])
        y_c = 0.5 * (extent_gg[2] + extent_gg[3])

        lons, _ = transformer_to_lonlat.transform(
            xticks,
            np.full_like(xticks, y_c)
        )
        _, lats = transformer_to_lonlat.transform(
            np.full_like(yticks, x_c),
            yticks
        )

        ax.set_xticks(xticks)
        ax.set_xticklabels(["{:.1f}".format(l) for l in lons])
        ax.set_yticks(yticks)
        ax.set_yticklabels(["{:.1f}".format(l) for l in lats])

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(title)

        gdf_gg.boundary.plot(ax=ax, linewidth=0.5, color="lightblue")

        return fig, ax

    def add_full_height_colorbar(fig, ax, mappable, label_text=""):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = fig.colorbar(mappable, cax=cax)

        if label_text:
            cbar.set_label(label_text)

        return cbar

    png_paths = []

    dev_png = OUTPUT_DIR / "{}_dev_good_days_14_{}.png".format(
        species,
        ref_date
    )
    dev_masked = np.where(dev_gg == nodata_gg, np.nan, dev_gg)

    fig, ax = setup_axes(
        "Ανάπτυξη {}\nΕυνοϊκές ημέρες τελευταίων 14 ημερών ({})".format(
            label,
            ref_date
        )
    )

    im = ax.imshow(
        dev_masked,
        origin="upper",
        extent=extent_gg,
        vmin=0,
        vmax=14
    )

    add_full_height_colorbar(
        fig,
        ax,
        im,
        label_text="Good development days (last 14 days)"
    )

    fig.tight_layout(rect=[0.06, 0.06, 0.92, 0.94])
    fig.savefig(str(dev_png), dpi=150)
    plt.close(fig)

    print("Wrote", dev_png)
    png_paths.append(dev_png)

    act_png = OUTPUT_DIR / "{}_act_good_days_7_{}.png".format(
        species,
        ref_date
    )
    act_masked = np.where(act_gg == act_nodata, np.nan, act_gg)

    fig, ax = setup_axes(
        "Δραστηριότητα {}\nΕυνοϊκές ημέρες τελευταίων 7 ημερών ({})".format(
            label,
            ref_date
        )
    )

    im = ax.imshow(
        act_masked,
        origin="upper",
        extent=extent_gg,
        vmin=0,
        vmax=7
    )

    add_full_height_colorbar(
        fig,
        ax,
        im,
        label_text="Good activity days (last 7 days)"
    )

    fig.tight_layout(rect=[0.06, 0.06, 0.92, 0.94])
    fig.savefig(str(act_png), dpi=150)
    plt.close(fig)

    print("Wrote", act_png)
    png_paths.append(act_png)

    combined = np.full(dev_gg.shape, -1, dtype=np.int16)
    nodata_mask = (dev_gg == nodata_gg) | (act_gg == act_nodata)

    dev_vals = dev_gg.astype(float)
    act_vals = act_gg.astype(float)

    combined[~nodata_mask] = 0

    mask_dev_only = (~nodata_mask) & (dev_vals >= 6.0) & (act_vals < 4.0)
    mask_act_only = (~nodata_mask) & (dev_vals < 6.0) & (act_vals >= 4.0)
    mask_both = (~nodata_mask) & (dev_vals >= 6.0) & (act_vals >= 4.0)

    combined[mask_dev_only] = 1
    combined[mask_act_only] = 2
    combined[mask_both] = 3

    combined_plot = np.where(combined < 0, np.nan, combined)

    class_colors = [
        "#f0f0f0",
        "#1f77b4",
        "#2ca02c",
        "#d62728",
    ]

    cmap = ListedColormap(class_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    combined_png = OUTPUT_DIR / "{}_combined_{}.png".format(species, ref_date)

    fig, ax = setup_axes(
        "Κλιματική καταλληλότητα για {} ({})".format(label, ref_date)
    )

    im = ax.imshow(
        combined_plot,
        origin="upper",
        extent=extent_gg,
        cmap=cmap,
        norm=norm
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])

    cbar.ax.set_yticklabels([
        "Unfavourable",
        "Dev only",
        "Activity only",
        "Dev + activity"
    ])

    fig.tight_layout(rect=[0.06, 0.06, 0.92, 0.94])
    fig.savefig(str(combined_png), dpi=150)
    plt.close(fig)

    print("Wrote", combined_png)
    png_paths.append(combined_png)

    return png_paths


# -------------------------------------------------------------------
# FTP
# -------------------------------------------------------------------

def ftp_connect(max_attempts=8, sleep_seconds=30):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            print("Opening FTPS connection, attempt {}/{}...".format(
                attempt,
                max_attempts
            ))

            ftps = FTP_TLS()
            ftps.connect(FTP_HOST, 21, timeout=60)
            ftps.login(user=FTP_USER, passwd=FTP_PASS)
            ftps.prot_p()

            if FTP_TARGET_DIR not in ("", "/"):
                ftps.cwd(FTP_TARGET_DIR)

            return ftps

        except Exception as e:
            last_error = e
            print("FTPS connection attempt {} failed: {}".format(attempt, e))

            try:
                ftps.close()
            except Exception:
                pass

            if attempt < max_attempts:
                print("Waiting {} seconds before retry...".format(sleep_seconds))
                time.sleep(sleep_seconds)

    raise RuntimeError(
        "Could not open FTPS connection after {} attempts. Last error: {}".format(
            max_attempts,
            last_error
        )
    )


def upload_files_via_ftp(file_paths):
    if not file_paths:
        print("No files to upload via FTP.")
        return

    print("\nUploading PNG maps via FTPS...")

    ftps = None

    try:
        ftps = ftp_connect(max_attempts=8, sleep_seconds=30)
        print("FTPS session opened.")

        for path in file_paths:
            filename = Path(path).name
            print("Uploading", filename)

            with open(str(path), "rb") as f:
                ftps.storbinary("STOR " + filename, f)

            print("Uploaded:", filename)

    finally:
        if ftps is not None:
            try:
                ftps.quit()
            except Exception:
                try:
                    ftps.close()
                except Exception:
                    pass

    print("FTPS upload complete.")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    require_env()
    ensure_greece_geojson()
    ensure_dem()

    df = download_tsv()

    required_cols = {
        "webcode",
        "ref_date",
        "species",
        "dev_index_14",
        "act_index_7",
        "dev_good_days_14",
        "act_good_days_7",
        "lat",
        "lon",
    }

    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "TSV is missing columns: {}".format(", ".join(sorted(missing)))
        )

    print("Species found in TSV:", sorted(df["species"].dropna().unique()))
    print("Rows per species:")
    print(df["species"].value_counts(dropna=False))

    ref_dates = df["ref_date"].dropna().unique()

    if len(ref_dates) == 0:
        raise ValueError("No ref_date values in TSV.")

    ref_date = str(ref_dates[0])
    print("Reference date from TSV:", ref_date)

    today_ref = ATHENS_NOW.strftime("%Y-%m-%d")

    if ref_date != today_ref:
        print(
            "WARNING: TSV ref_date is {}, but Athens today is {}.".format(
                ref_date,
                today_ref
            )
        )

    for col in [
        "dev_index_14",
        "act_index_7",
        "dev_good_days_14",
        "act_good_days_7",
        "lat",
        "lon",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dem_data, profile, transform, crs, nodata, valid_mask = load_dem_and_mask()

    if nodata is None:
        nodata = -9999.0

    X_grid, grid_rows, grid_cols = build_grid_design_matrix(
        dem_data,
        transform,
        valid_mask
    )

    for species in SPECIES_LIST:
        df_sp = df[df["species"] == species].copy()

        if df_sp.empty:
            print("\n=== Species: {} has no rows, skipping ===".format(species))
            continue

        print("\n=== Species: {} ===".format(species))

        lats = df_sp["lat"].values.astype(np.float32)
        lons = df_sp["lon"].values.astype(np.float32)

        coord_mask = np.isfinite(lats) & np.isfinite(lons)

        if coord_mask.sum() < 5:
            print(
                "Too few stations with coordinates for {}. Skipping.".format(
                    species
                )
            )
            continue

        lats = lats[coord_mask]
        lons = lons[coord_mask]
        df_sp = df_sp.loc[coord_mask].reset_index(drop=True)

        elev_st = sample_dem_at_points(
            dem_data,
            transform,
            nodata,
            lons,
            lats
        )

        X_st_base = np.column_stack([
            np.ones_like(elev_st, dtype=np.float32),
            elev_st,
            lats,
            lons,
        ])

        for metric_name, meta in METRICS.items():
            window_days = meta["window_days"]
            metric_suffix = meta["output_suffix"]

            print("Metric:", metric_name)

            y = df_sp[metric_name].values.astype(np.float32)

            beta = fit_linear_regression(X_st_base, y)

            if beta is None:
                print(
                    "Skipping metric {} for species {} due to too few points".format(
                        metric_name,
                        species
                    )
                )
                continue

            print("Coefficients:", beta)

            y_grid_flat = np.matmul(X_grid, beta)
            y_grid_flat = np.clip(y_grid_flat, 0.0, 1.0)

            grid_out = np.full(dem_data.shape, nodata, dtype=np.float32)
            grid_out[grid_rows, grid_cols] = y_grid_flat

            out_name_index = "{}_{}_{}.tif".format(
                species,
                metric_suffix,
                ref_date
            )
            out_path_index = OUTPUT_DIR / out_name_index
            write_raster(out_path_index, grid_out, profile, nodata)

            if meta.get("scale_to_days", False):
                days_grid = np.where(
                    grid_out == nodata,
                    nodata,
                    grid_out * float(window_days)
                )

                if "dev" in metric_name:
                    gd_suffix = "dev_good_days_{}".format(window_days)
                else:
                    gd_suffix = "act_good_days_{}".format(window_days)

                out_name_days = "{}_{}_{}.tif".format(
                    species,
                    gd_suffix,
                    ref_date
                )
                out_path_days = OUTPUT_DIR / out_name_days
                write_raster(out_path_days, days_grid, profile, nodata)

    print("\nGeoTIFF grids written to:", OUTPUT_DIR)

    png_files = []

    for species in SPECIES_LIST:
        label = SPECIES_LABELS.get(species, species)
        png_files.extend(create_species_maps(species, label, ref_date))

    upload_files_via_ftp(png_files)

    print("\nAll done.")


if __name__ == "__main__":
    main()
