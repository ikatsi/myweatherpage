#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download each HUA Meteoclima station latest_month.nc, compute DAILY stats in Europe/Athens
(local day boundaries with DST handled), write harokopeio.json, and optionally upload it via FTPS.

Outputs per station (per local date YYYY-MM-DD):
- tmin: min tempC
- tavg: mean tempC
- tmax: max tempC
- ppn:  sum of precipitation increments (mm) derived from cumulative rainYear
- wspeed: mean windSpeed converted to kph (source is m/s)
- hisp: max windGust converted to kph (source is m/s)
"""

import os
import json
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path
from ftplib import FTP_TLS

import numpy as np
import pandas as pd
import requests
import xarray as xr
from zoneinfo import ZoneInfo


# ======================
# CONFIG
# ======================
TZ_LOCAL = ZoneInfo("Europe/Athens")

# All sensitive values are injected via environment variables by the CI runner.
# You created these in GitHub → Settings → Secrets and variables → Actions.
DATA_URL = os.environ.get("CURRENTWEATHER_URL", "").strip()  # your secret name
FTP_HOST = os.environ.get("FTP_HOST", "").strip()
FTP_USER = os.environ.get("FTP_USER", "").strip()
FTP_PASS = os.environ.get("FTP_PASS", "").strip()  # empty disables uploads

STATIONS = {
    # webcode: station_folder
    "hua_alimos": "Alimos",
    "hua_argyroupoli": "Argyroupolis",
    "hua_peristeri": "Peristeri",
    "hua_perama": "Perama",
    "hua_elefsina": "Elefsis",
    "hua_nikaia": "Nikaia",
    "hua_rafina": "Rafina",
    "hua_ilion": "Tritsi",  # dashboard folder is Tritsi, webcode you want is hua_ilion
}

# Fallback base if DATA_URL is not set (keeps local runs working)
DEFAULT_BASE = "http://meteoclima.hua.gr/stations"
NC_NAME = "latest_month.nc"

OUT_JSON = "harokopeio.json"

HTTP_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (compatible; harokopeio-json-builder/1.0)"


# ======================
# HELPERS
# ======================
def station_nc_url(folder: str) -> str:
    """
    If DATA_URL is provided, use it as the base prefix.
    Otherwise use DEFAULT_BASE.

    Expected:
    - DATA_URL like "http://meteoclima.hua.gr/stations"
      final: "{DATA_URL}/{folder}/latest_month.nc"
    """
    base = DATA_URL if DATA_URL else DEFAULT_BASE
    return f"{base.rstrip('/')}/{folder}/{NC_NAME}"


def download_nc(url: str) -> str:
    """Download to a temp file and return the file path."""
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT, stream=True)
    r.raise_for_status()

    fd, tmp_path = tempfile.mkstemp(suffix=".nc", prefix="hua_")
    os.close(fd)

    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    return tmp_path


def to_local_times_athens(time_values: np.ndarray) -> pd.DatetimeIndex:
    """
    Treat netCDF 'time' as UTC timestamps, convert to Europe/Athens.
    Returns tz-aware DatetimeIndex in Europe/Athens.
    """
    t = pd.to_datetime(time_values)  # naive datetime64 -> Timestamp, no tz
    t = t.tz_localize("UTC").tz_convert(TZ_LOCAL)
    return t


def safe_series(ds: xr.Dataset, varname: str) -> np.ndarray:
    """Return ds[varname] as a 1D numpy float array (or raise KeyError)."""
    arr = ds[varname].values
    arr = np.asarray(arr).reshape(-1)
    return arr.astype("float64", copy=False)


def precipitation_from_cumulative(cum_mm: np.ndarray) -> np.ndarray:
    """
    Convert cumulative rainfall (mm) to per-observation increments (mm).
    Handles resets by clipping negative diffs to 0.
    """
    cum = np.asarray(cum_mm, dtype="float64")
    if cum.size == 0:
        return cum

    inc = np.diff(cum, prepend=cum[0])
    inc[~np.isfinite(inc)] = 0.0
    inc = np.where(inc < 0, 0.0, inc)
    return inc


def ms_to_kph(x: np.ndarray) -> np.ndarray:
    """m/s to km/h"""
    return np.asarray(x, dtype="float64") * 3.6


def _round_or_none(x: Optional[float], nd: int) -> Optional[float]:
    if x is None:
        return None
    if not np.isfinite(x):
        return None
    return round(float(x), nd)


def build_daily_stats(ds: xr.Dataset) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Given an opened station dataset, compute per-local-day stats.
    Returns: { "YYYY-MM-DD": {tmin,tavg,tmax,ppn,wspeed,hisp} }
    """
    needed = ["time", "tempC", "windSpeed", "windGust", "rainYear"]
    for v in needed:
        if v not in ds:
            raise KeyError(f"Missing variable '{v}' in nc file. Found: {list(ds.variables)}")

    t_local = to_local_times_athens(ds["time"].values)
    if len(t_local) == 0:
        return {}

    tempC = safe_series(ds, "tempC")
    windSpeed_ms = safe_series(ds, "windSpeed")
    windGust_ms = safe_series(ds, "windGust")
    rainYear = safe_series(ds, "rainYear")

    # Convert wind to kph
    windSpeed_kph = ms_to_kph(windSpeed_ms)
    windGust_kph = ms_to_kph(windGust_ms)

    # Precip increments from cumulative rainYear (mm)
    rain_inc = precipitation_from_cumulative(rainYear)

    # Align lengths defensively (sometimes variables can mismatch in weird files)
    n = min(len(t_local), len(tempC), len(windSpeed_kph), len(windGust_kph), len(rain_inc))
    t_local = t_local[:n]
    tempC = tempC[:n]
    windSpeed_kph = windSpeed_kph[:n]
    windGust_kph = windGust_kph[:n]
    rain_inc = rain_inc[:n]

    df = pd.DataFrame(
        {
            "tempC": tempC,
            "windSpeed_kph": windSpeed_kph,
            "windGust_kph": windGust_kph,
            "rain_inc_mm": rain_inc,
        },
        index=t_local,
    )

    df = df[~df.index.isna()].copy()
    if df.empty:
        return {}

    g = df.groupby(df.index.date)

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for d, part in g:
        day_key = d.isoformat()

        if part["tempC"].notna().any():
            tmin = float(np.nanmin(part["tempC"].values))
            tmax = float(np.nanmax(part["tempC"].values))
            tavg = float(np.nanmean(part["tempC"].values))
        else:
            tmin = tmax = tavg = None

        # Precip sum: if no valid values, default 0.0
        if part["rain_inc_mm"].notna().any():
            ppn = float(np.nansum(part["rain_inc_mm"].values))
        else:
            ppn = 0.0

        if part["windSpeed_kph"].notna().any():
            wspeed = float(np.nanmean(part["windSpeed_kph"].values))
        else:
            wspeed = None

        if part["windGust_kph"].notna().any():
            hisp = float(np.nanmax(part["windGust_kph"].values))
        else:
            hisp = None

        out[day_key] = {
            "tmin": _round_or_none(tmin, 1),
            "tavg": _round_or_none(tavg, 1),
            "tmax": _round_or_none(tmax, 1),
            "ppn": _round_or_none(ppn, 1),
            "wspeed": _round_or_none(wspeed, 1),
            "hisp": _round_or_none(hisp, 1),
        }

    return out


def ftp_upload_file(local_path: str, remote_filename: str) -> None:
    """
    Upload local_path to FTP server root as remote_filename using FTP over TLS.
    If FTP_PASS is empty, uploads are disabled.
    """
    if not FTP_PASS:
        print("FTP_PASS empty -> uploads disabled; skipping FTP upload.")
        return
    if not FTP_HOST or not FTP_USER:
        raise RuntimeError("FTP upload requested but FTP_HOST/FTP_USER not set.")

    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Local file missing: {local_path}")

    with FTP_TLS(FTP_HOST, timeout=30) as ftp:
        ftp.login(FTP_USER, FTP_PASS)
        ftp.prot_p()  # secure data channel
        with p.open("rb") as f:
            ftp.storbinary(f"STOR {remote_filename}", f)


# ======================
# MAIN
# ======================
def main() -> None:
    all_data: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for webcode, folder in STATIONS.items():
        url = station_nc_url(folder)
        tmp_path = None

        try:
            tmp_path = download_nc(url)

            # Load dataset (disable CF decode guessing if you ever see time weirdness)
            ds = xr.open_dataset(tmp_path)

            daily = build_daily_stats(ds)

            all_data[webcode] = {
                "source_nc": url,
                "tz": "Europe/Athens",
                "days": daily,  # {YYYY-MM-DD: {...}}
            }

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            errors[webcode] = err
            all_data[webcode] = {
                "source_nc": url,
                "tz": "Europe/Athens",
                "days": {},
                "error": err,
            }

        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    payload = {
        "generated_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "stations": all_data,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK: wrote {OUT_JSON}")

    # Optional FTP upload to server root
    try:
        ftp_upload_file(OUT_JSON, os.path.basename(OUT_JSON))
        if FTP_PASS:
            print(f"OK: uploaded {OUT_JSON} to FTP root as {os.path.basename(OUT_JSON)}")
    except Exception as e:
        print(f"WARNING: FTP upload failed: {type(e).__name__}: {e}")

    if errors:
        print("\nSome stations had errors:")
        for k, v in errors.items():
            print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
