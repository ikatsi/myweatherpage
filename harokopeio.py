#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_harokopeio_json.py

Download each HUA Meteoclima station `latest_month.nc`, compute DAILY stats in Europe/Athens
(local day boundaries with DST handled), and write `harokopeio.json`.

Per station, per local date (YYYY-MM-DD), outputs:
- tmin:   minimum temperature (°C)
- tavg:   mean temperature (°C)
- tmax:   maximum temperature (°C)
- ppn:    daily precipitation sum (mm), derived from cumulative `rainYear` increments
- wspeed: mean wind speed (km/h), from `windSpeed` (m/s) * 3.6
- hisp:   maximum wind gust (km/h), from `windGust` (m/s) * 3.6

Notes
- NetCDF time is treated as UTC and converted to Europe/Athens.
- If `rainYear` resets, negative diffs are clipped to 0.
- Day aggregation is done by local calendar day (Athens), not UTC day.

Example:
  python3 build_harokopeio_json.py --out harokopeio.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr
from zoneinfo import ZoneInfo

LOG = logging.getLogger("harokopeio")


# ======================
# DEFAULT CONFIG
# ======================
TZ_LOCAL = ZoneInfo("Europe/Athens")

STATIONS: Dict[str, str] = {
    # webcode: station_folder
    "hua_alimos": "Alimos",
    "hua_argyroupoli": "Argyroupolis",
    "hua_peristeri": "Peristeri",
    "hua_perama": "Perama",
    "hua_elefsina": "Elefsis",
    "hua_nikaia": "Nikaia",
    "hua_rafina": "Rafina",
    "hua_ilion": "Tritsi",  # dashboard folder is Tritsi, webcode requested is hua_ilion
}

BASE = "http://meteoclima.hua.gr/stations"
NC_NAME = "latest_month.nc"

DEFAULT_OUT = "harokopeio.json"
DEFAULT_TIMEOUT = 30

USER_AGENT = "harokopeio-json-builder/1.0 (+https://github.com/your-org/your-repo)"


# ======================
# HELPERS
# ======================
def download_nc(url: str, timeout: int) -> str:
    """
    Download a URL to a temporary .nc file and return the file path.
    Caller is responsible for deleting the returned path.
    """
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout, stream=True)
    r.raise_for_status()

    fd, tmp_path = tempfile.mkstemp(suffix=".nc", prefix="hua_")
    os.close(fd)

    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    return tmp_path


def to_local_time_index(time_values: np.ndarray) -> pd.DatetimeIndex:
    """
    Interpret netCDF 'time' values as UTC, convert to Europe/Athens, return tz-aware index.
    """
    t = pd.to_datetime(time_values)              # naive timestamps
    t = t.tz_localize("UTC").tz_convert(TZ_LOCAL)
    return t


def safe_1d_float(ds: xr.Dataset, varname: str) -> np.ndarray:
    """
    Return ds[varname] flattened to 1D float64.
    """
    arr = np.asarray(ds[varname].values).reshape(-1)
    return arr.astype("float64", copy=False)


def ms_to_kph(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype="float64") * 3.6


def precipitation_from_cumulative(cum_mm: np.ndarray) -> np.ndarray:
    """
    Convert a cumulative rainfall series (mm) to per-sample increments (mm).
    Handles resets by clipping negative diffs to 0.
    """
    cum = np.asarray(cum_mm, dtype="float64")
    inc = np.diff(cum, prepend=cum[0])  # first increment becomes 0 by construction
    inc[~np.isfinite(inc)] = 0.0
    inc = np.where(inc < 0, 0.0, inc)
    return inc


def r1(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 1)


def build_daily_stats(ds: xr.Dataset) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute per-local-day stats from a station dataset.

    Returns:
      { "YYYY-MM-DD": {"tmin":..., "tavg":..., "tmax":..., "ppn":..., "wspeed":..., "hisp":...} }
    """
    required = ["time", "tempC", "windSpeed", "windGust", "rainYear"]
    missing = [v for v in required if v not in ds]
    if missing:
        raise KeyError(f"Missing required variables: {', '.join(missing)}")

    t_local = to_local_time_index(ds["time"].values)
    if len(t_local) == 0:
        return {}

    tempC = safe_1d_float(ds, "tempC")
    windSpeed_kph = ms_to_kph(safe_1d_float(ds, "windSpeed"))
    windGust_kph = ms_to_kph(safe_1d_float(ds, "windGust"))
    rain_inc_mm = precipitation_from_cumulative(safe_1d_float(ds, "rainYear"))

    df = pd.DataFrame(
        {
            "tempC": tempC,
            "windSpeed_kph": windSpeed_kph,
            "windGust_kph": windGust_kph,
            "rain_inc_mm": rain_inc_mm,
        },
        index=t_local,
    )

    df = df[~df.index.isna()].copy()

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for d, part in df.groupby(df.index.date):
        day_key = d.isoformat()

        temp = part["tempC"].to_numpy()
        ws = part["windSpeed_kph"].to_numpy()
        wg = part["windGust_kph"].to_numpy()
        rain = part["rain_inc_mm"].to_numpy()

        tmin = float(np.nanmin(temp)) if np.isfinite(temp).any() else None
        tmax = float(np.nanmax(temp)) if np.isfinite(temp).any() else None
        tavg = float(np.nanmean(temp)) if np.isfinite(temp).any() else None

        wspeed = float(np.nanmean(ws)) if np.isfinite(ws).any() else None
        hisp = float(np.nanmax(wg)) if np.isfinite(wg).any() else None

        ppn = float(np.nansum(rain)) if np.isfinite(rain).any() else 0.0

        out[day_key] = {
            "tmin": r1(tmin),
            "tavg": r1(tavg),
            "tmax": r1(tmax),
            "ppn": r1(ppn),
            "wspeed": r1(wspeed),
            "hisp": r1(hisp),
        }

    return out


def build_payload(timeout: int) -> Dict[str, Any]:
    """
    Build the full JSON payload across all stations.
    """
    stations_out: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for webcode, folder in STATIONS.items():
        url = f"{BASE}/{folder}/{NC_NAME}"
        tmp_path: Optional[str] = None

        try:
            LOG.info("Downloading %s (%s)", webcode, url)
            tmp_path = download_nc(url, timeout=timeout)

            # engine="netcdf4" may be needed on some setups; xarray will choose best available.
            ds = xr.open_dataset(tmp_path)
            daily = build_daily_stats(ds)

            stations_out[webcode] = {
                "source_nc": url,
                "tz": "Europe/Athens",
                "days": daily,
            }

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            errors[webcode] = msg
            LOG.exception("Failed for %s: %s", webcode, msg)
            stations_out[webcode] = {
                "source_nc": url,
                "tz": "Europe/Athens",
                "days": {},
                "error": msg,
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    return {
        "generated_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "stations": stations_out,
        "errors": errors,  # convenience top-level
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build harokopeio.json from HUA Meteoclima netCDF files.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output JSON path (default: harokopeio.json)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds (default: 30)")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING...)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    payload = build_payload(timeout=args.timeout)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    LOG.info("Wrote %s", args.out)

    if payload.get("errors"):
        LOG.warning("Some stations had errors (%d). See `errors` in JSON.", len(payload["errors"]))


if __name__ == "__main__":
    main()
