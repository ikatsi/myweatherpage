import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from ftplib import FTP
from pathlib import Path
from zoneinfo import ZoneInfo

import netCDF4
import numpy as np
import requests


ATHENS = ZoneInfo("Europe/Athens")

SOURCE_DIRECTORY = Path("source_files")
OUTPUT_FILE = Path("data_output.json")

FTP_HOST = os.environ.get("CLIMATE_FTP_HOST")
FTP_USER = os.environ.get("CLIMATE_FTP_USER")
FTP_PASS = os.environ.get("CLIMATE_FTP_PASS")


def clean_number(value, minimum=None, maximum=None):
    if value is None or np.ma.is_masked(value):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    if minimum is not None and number < minimum:
        return None

    if maximum is not None and number > maximum:
        return None

    return number


def rounded(value, digits=2):
    if value is None:
        return None

    return round(float(value), digits)


def variable_value(dataset, variable_name, index):
    if variable_name not in dataset.variables:
        return None

    variable = dataset.variables[variable_name]

    try:
        return variable[index]
    except (IndexError, TypeError, ValueError):
        return None


def first_available(dataset, variable_names, index):
    for variable_name in variable_names:
        value = variable_value(
            dataset,
            variable_name,
            index,
        )

        if value is not None and not np.ma.is_masked(value):
            return value

    return None


def dew_point(temperature, humidity):
    if (
        temperature is None
        or humidity is None
        or humidity <= 0
    ):
        return None

    a = 17.625
    b = 243.04

    gamma = (
        math.log(humidity / 100.0)
        + (
            (a * temperature)
            / (b + temperature)
        )
    )

    return (b * gamma) / (a - gamma)


def compass_direction(degrees):
    if degrees is None:
        return None

    labels = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]

    normalized = degrees % 360.0
    index = int(
        (normalized + 11.25) // 22.5
    ) % 16

    return labels[index]


def vector_mean_direction(rows):
    vector_x = 0.0
    vector_y = 0.0
    total_weight = 0.0

    for row in rows:
        direction = row["wdir"]

        if direction is None:
            continue

        weight = row["wind_ms"]

        if weight is None or weight <= 0:
            weight = 1.0

        radians = math.radians(direction)

        vector_x += weight * math.sin(radians)
        vector_y += weight * math.cos(radians)
        total_weight += weight

    if total_weight == 0:
        return None

    if (
        abs(vector_x) < 0.000001
        and abs(vector_y) < 0.000001
    ):
        return None

    degrees = math.degrees(
        math.atan2(vector_x, vector_y)
    )

    if degrees < 0:
        degrees += 360.0

    return degrees


def summarize_day(rows):
    temperatures = [
        row["temp"]
        for row in rows
        if row["temp"] is not None
    ]

    humidities = [
        row["rh"]
        for row in rows
        if row["rh"] is not None
    ]

    wind_speeds = [
        row["wind_ms"]
        for row in rows
        if row["wind_ms"] is not None
    ]

    gusts = [
        row["gust_ms"]
        for row in rows
        if row["gust_ms"] is not None
    ]

    rain_values = [
        row["rain_day"]
        for row in rows
        if row["rain_day"] is not None
    ]

    maximum_gust_row = None

    for row in rows:
        if row["gust_ms"] is None:
            continue

        if (
            maximum_gust_row is None
            or row["gust_ms"]
            > maximum_gust_row["gust_ms"]
        ):
            maximum_gust_row = row

    mean_direction = vector_mean_direction(rows)

    return {
        "tmin": (
            rounded(min(temperatures), 2)
            if temperatures
            else None
        ),
        "tavg": (
            rounded(
                sum(temperatures)
                / len(temperatures),
                2,
            )
            if temperatures
            else None
        ),
        "tmax": (
            rounded(max(temperatures), 2)
            if temperatures
            else None
        ),
        "ppn": (
            rounded(max(rain_values), 1)
            if rain_values
            else None
        ),
        "avgsp": (
            rounded(
                (
                    sum(wind_speeds)
                    / len(wind_speeds)
                )
                * 3.6,
                2,
            )
            if wind_speeds
            else None
        ),
        "hisp": (
            rounded(max(gusts) * 3.6, 2)
            if gusts
            else None
        ),
        "minrh": (
            int(round(min(humidities)))
            if humidities
            else None
        ),
        "maxrh": (
            int(round(max(humidities)))
            if humidities
            else None
        ),
        "domwdir": compass_direction(
            mean_direction
        ),
        "hiwinddir": (
            rounded(
                maximum_gust_row["wdir"],
                1,
            )
            if (
                maximum_gust_row is not None
                and maximum_gust_row["wdir"]
                is not None
            )
            else None
        ),
    }


def read_station_file(file_path):
    dataset = netCDF4.Dataset(
        file_path,
        mode="r",
    )

    try:
        if "time" not in dataset.variables:
            raise RuntimeError(
                "NetCDF file has no time variable"
            )

        time_values = dataset.variables["time"][:]
        rows = []

        for index, raw_time in enumerate(
            time_values
        ):
            timestamp = clean_number(
                raw_time,
                946684800,
                4102444800,
            )

            if timestamp is None:
                continue

            observation_time = (
                datetime.fromtimestamp(
                    timestamp,
                    tz=timezone.utc,
                ).astimezone(ATHENS)
            )

            temperature = clean_number(
                variable_value(
                    dataset,
                    "tempC",
                    index,
                ),
                -50,
                60,
            )

            if temperature is None:
                continue

            humidity = clean_number(
                variable_value(
                    dataset,
                    "relHumidity",
                    index,
                ),
                0,
                100,
            )

            pressure_pa = clean_number(
                first_available(
                    dataset,
                    [
                        "baromRelPa",
                        "baromAbsPa",
                    ],
                    index,
                ),
                80000,
                110000,
            )

            rain_day = clean_number(
                variable_value(
                    dataset,
                    "rainDay",
                    index,
                ),
                0,
                2000,
            )

            rain_event = clean_number(
                variable_value(
                    dataset,
                    "rainEvent",
                    index,
                ),
                0,
                2000,
            )

            rain_rate = clean_number(
                variable_value(
                    dataset,
                    "rainRate",
                    index,
                ),
                0,
                1000,
            )

            wind_speed = clean_number(
                variable_value(
                    dataset,
                    "windSpeed",
                    index,
                ),
                0,
                100,
            )

            wind_gust = clean_number(
                variable_value(
                    dataset,
                    "windGust",
                    index,
                ),
                0,
                150,
            )

            wind_direction = clean_number(
                first_available(
                    dataset,
                    [
                        "windDirAvg10m",
                        "windDir",
                    ],
                    index,
                ),
                0,
                360,
            )

            rows.append(
                {
                    "datetime": observation_time,
                    "month_key": (
                        observation_time.strftime(
                            "%Y-%m"
                        )
                    ),
                    "date_key": (
                        observation_time.strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    "temp": temperature,
                    "rh": humidity,
                    "pressure_hpa": (
                        pressure_pa / 100.0
                        if pressure_pa is not None
                        else None
                    ),
                    "rain_day": rain_day,
                    "rain_event": rain_event,
                    "rain_rate": rain_rate,
                    "wind_ms": wind_speed,
                    "gust_ms": wind_gust,
                    "wdir": wind_direction,
                }
            )

        return rows
    finally:
        dataset.close()


def download_station_file(
    station_id,
    source_url,
):
    if not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        station_id,
    ):
        raise RuntimeError(
            "Invalid station identifier"
        )

    destination = (
        SOURCE_DIRECTORY
        / f"{station_id}.nc"
    )

    response = requests.get(
        source_url,
        timeout=90,
        headers={
            "User-Agent": (
                "weather-data-sync/1.0"
            ),
        },
    )

    response.raise_for_status()

    if not response.content.startswith(
        b"\x89HDF\r\n\x1a\n"
    ):
        raise RuntimeError(
            "Downloaded response is not HDF5"
        )

    temporary_file = destination.with_suffix(
        ".nc.downloading"
    )

    temporary_file.write_bytes(
        response.content
    )

    temporary_file.replace(destination)

    return destination


def process_station(configuration):
    station_id = configuration["id"]
    source_url = configuration["url"]

    only_tmax = bool(
        configuration.get(
            "only_tmax",
            False,
        )
    )

    downloaded_file = download_station_file(
        station_id,
        source_url,
    )

    rows = read_station_file(
        downloaded_file
    )

    if not rows:
        raise RuntimeError(
            "No valid observations found"
        )

    month_counts = Counter(
        row["month_key"]
        for row in rows
    )

    selected_month = (
        month_counts.most_common(1)[0][0]
    )

    rows = [
        row
        for row in rows
        if row["month_key"]
        == selected_month
    ]

    rows.sort(
        key=lambda row: row["datetime"]
    )

    grouped_rows = defaultdict(list)

    for row in rows:
        grouped_rows[
            row["date_key"]
        ].append(row)

    daily_output = []
    summaries = {}

    for date_key in sorted(grouped_rows):
        summary = summarize_day(
            grouped_rows[date_key]
        )

        summaries[date_key] = summary

        year, month, day = [
            int(part)
            for part in date_key.split("-")
        ]

        if only_tmax:
            daily_output.append(
                {
                    "webcode": station_id,
                    "day": day,
                    "month": month,
                    "year": year,
                    "date": date_key,
                    "tmax": summary["tmax"],
                }
            )
        else:
            daily_output.append(
                {
                    "webcode": station_id,
                    "day": day,
                    "month": month,
                    "year": year,
                    "date": date_key,
                    "tmin": summary["tmin"],
                    "tavg": summary["tavg"],
                    "tmax": summary["tmax"],
                    "ppn": summary["ppn"],
                    "avgsp": summary["avgsp"],
                    "hisp": summary["hisp"],
                    "minrh": summary["minrh"],
                    "maxrh": summary["maxrh"],
                    "domwdir": summary["domwdir"],
                }
            )

    latest = rows[-1]
    latest_summary = summaries[
        latest["date_key"]
    ]

    latest_datetime = (
        latest["datetime"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    if only_tmax:
        latest_output = {
            "webcode": station_id,
            "datetime": latest_datetime,
            "tmax": latest_summary["tmax"],
        }
    else:
        latest_output = {
            "webcode": station_id,
            "datetime": latest_datetime,
            "tnow": rounded(
                latest["temp"],
                1,
            ),
            "tdnow": rounded(
                dew_point(
                    latest["temp"],
                    latest["rh"],
                ),
                1,
            ),
            "rhnow": rounded(
                latest["rh"],
                1,
            ),
            "baronow": rounded(
                latest["pressure_hpa"],
                1,
            ),
            "rainnow": rounded(
                latest["rain_event"],
                1,
            ),
            "rainintensity": rounded(
                latest["rain_rate"],
                1,
            ),
            "windspeednow": (
                rounded(
                    latest["wind_ms"] * 3.6,
                    1,
                )
                if latest["wind_ms"]
                is not None
                else None
            ),
            "wdirnow": rounded(
                latest["wdir"],
                1,
            ),
            "todayrain": (
                latest_summary["ppn"]
            ),
            "tmax": rounded(
                latest_summary["tmax"],
                1,
            ),
            "tmin": rounded(
                latest_summary["tmin"],
                1,
            ),
            "hiwindspeed": rounded(
                latest_summary["hisp"],
                1,
            ),
            "hiwinddir": (
                latest_summary["hiwinddir"]
            ),
        }

    return {
        "station": station_id,
        "selected_month": selected_month,
        "downloaded_file": (
            downloaded_file.name
        ),
        "valid_observations": len(rows),
        "latestvalues": latest_output,
        "alldata": daily_output,
    }


def upload_file(
    ftp,
    local_path,
    remote_name,
):
    temporary_name = (
        f"{remote_name}.uploading"
    )

    with local_path.open("rb") as handle:
        ftp.storbinary(
            f"STOR {temporary_name}",
            handle,
            blocksize=1024 * 1024,
        )

    try:
        ftp.delete(remote_name)
    except Exception:
        pass

    ftp.rename(
        temporary_name,
        remote_name,
    )


def upload_results():
    if not FTP_HOST:
        raise RuntimeError(
            "CLIMATE_FTP_HOST is missing"
        )

    if not FTP_USER:
        raise RuntimeError(
            "CLIMATE_FTP_USER is missing"
        )

    if not FTP_PASS:
        raise RuntimeError(
            "CLIMATE_FTP_PASS is missing"
        )

    with FTP(
        FTP_HOST,
        timeout=90,
    ) as ftp:
        ftp.login(
            user=FTP_USER,
            passwd=FTP_PASS,
        )

        ftp.set_pasv(True)

        downloaded_files = sorted(
            SOURCE_DIRECTORY.glob("*.nc")
        )

        for local_path in downloaded_files:
            upload_file(
                ftp,
                local_path,
                local_path.name,
            )

        # Upload JSON last. The PHP importer will
        # therefore never see the new JSON before
        # the corresponding downloads are complete.
        upload_file(
            ftp,
            OUTPUT_FILE,
            OUTPUT_FILE.name,
        )

        print(
            "FTP upload completed: "
            f"{len(downloaded_files)} "
            "source files and one JSON file"
        )


def main():
    source_config = os.environ.get(
        "STATION_SOURCES"
    )

    if not source_config:
        raise RuntimeError(
            "STATION_SOURCES is missing"
        )

    configurations = json.loads(
        source_config
    )

    if (
        not isinstance(configurations, list)
        or not configurations
    ):
        raise RuntimeError(
            "STATION_SOURCES must contain "
            "a non-empty JSON list"
        )

    SOURCE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    errors = []

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):
        try:
            result = process_station(
                configuration
            )

            results.append(result)

            print(
                f"Processed source "
                f"{index} of "
                f"{len(configurations)}"
            )
        except Exception as error:
            errors.append(
                {
                    "station": (
                        configuration.get(
                            "id",
                            "unknown",
                        )
                    ),
                    "error": str(error),
                }
            )

            print(
                f"Source {index} failed: "
                f"{error}"
            )

    output = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),
        "timezone": "Europe/Athens",
        "stations": results,
        "errors": errors,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Output written to "
        f"{OUTPUT_FILE.name}"
    )

    if not results:
        raise RuntimeError(
            "Every configured source failed"
        )

    upload_results()


if __name__ == "__main__":
    main()
