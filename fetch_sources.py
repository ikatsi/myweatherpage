#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from ftplib import FTP

import requests
import urllib3


# -------------------------------------------------------------------
# LOG MASKING
# -------------------------------------------------------------------

def mask_value(value):
    """
    Ask GitHub Actions to mask a value in logs.

    This helps prevent accidental leakage if a value is ever printed.
    """
    if value is None:
        return

    value = str(value)

    if value:
        print("::add-mask::{}".format(value))


def mask_config(cfg):
    """
    Mask all sensitive values contained in the config.

    This includes:
      - FTP credentials
      - processing URL and secret
      - fetch headers
      - source filenames
      - source URLs
    """

    ftp_cfg = cfg.get("ftp", {})
    processing_cfg = cfg.get("processing", {})
    fetch_cfg = cfg.get("fetch", {})
    fetch_headers = fetch_cfg.get("headers", {})
    sources = cfg.get("sources", [])

    for key in ["host", "user", "pass"]:
        mask_value(ftp_cfg.get(key, ""))

    for key in ["url", "secret"]:
        mask_value(processing_cfg.get(key, ""))

    for _, value in fetch_headers.items():
        mask_value(value)

    for item in sources:
        mask_value(item.get("filename", ""))
        mask_value(item.get("url", ""))


# -------------------------------------------------------------------
# CONFIG LOADING
# -------------------------------------------------------------------

def load_config():
    raw = os.environ.get("SOURCE_CONFIG_JSON")

    if not raw:
        raise RuntimeError("Missing SOURCE_CONFIG_JSON secret.")

    mask_value(raw)

    try:
        cfg = json.loads(raw)
    except Exception:
        raise RuntimeError("SOURCE_CONFIG_JSON is not valid JSON.")

    mask_config(cfg)

    if "ftp" not in cfg:
        raise RuntimeError("Config is missing ftp section.")

    if "sources" not in cfg:
        raise RuntimeError("Config is missing sources section.")

    if not isinstance(cfg["sources"], list) or not cfg["sources"]:
        raise RuntimeError("Config sources section is empty or invalid.")

    for i, item in enumerate(cfg["sources"], start=1):
        if "filename" not in item:
            raise RuntimeError("Source item #{0} is missing filename.".format(i))

        if "url" not in item:
            raise RuntimeError("Source item #{0} is missing url.".format(i))

        filename = item["filename"]

        if "/" in filename or "\\" in filename:
            raise RuntimeError("Source item #{0} has an unsafe filename.".format(i))

        if not filename.lower().endswith(".json"):
            raise RuntimeError("Source item #{0} filename must end with .json.".format(i))

    return cfg


# -------------------------------------------------------------------
# FILE HELPERS
# -------------------------------------------------------------------

def ensure_output_dir(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def save_file(output_dir, filename, content):
    path = os.path.join(output_dir, filename)

    with open(path, "wb") as f:
        f.write(content)

    return path


# -------------------------------------------------------------------
# FETCHING
# -------------------------------------------------------------------

def configure_ssl_warning_behavior(verify_ssl):
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_one(url, headers, verify_ssl, timeout_connect, timeout_read):
    response = requests.get(
        url,
        headers=headers,
        verify=verify_ssl,
        timeout=(timeout_connect, timeout_read)
    )

    response.raise_for_status()

    return response.content


# -------------------------------------------------------------------
# FTP UPLOAD
# -------------------------------------------------------------------

def upload_files_to_ftp(file_paths, ftp_cfg):
    """
    Upload ONLY JSON files created by this script.

    Upload strategy:
      1) Try passive FTP first.
      2) If passive mode uploads zero files, try active FTP.
      3) Reconnect cleanly for the second attempt.

    Important:
      - No remote folder is selected.
      - No ftp.cwd() is used.
      - Files are uploaded directly into the FTP login directory.
      - Filenames are not printed.
      - Non-JSON files are skipped.
      - FTP errors are printed without filenames.
    """

    host = ftp_cfg.get("host", "")
    user = ftp_cfg.get("user", "")
    password = ftp_cfg.get("pass", "")
    timeout = int(ftp_cfg.get("timeout", 60))

    if not host:
        raise RuntimeError("FTP host is missing.")

    if not user:
        raise RuntimeError("FTP user is missing.")

    if not password:
        raise RuntimeError("FTP password is missing.")

    clean_paths = []

    for path in file_paths:
        if not path:
            continue

        if not os.path.exists(path) or not os.path.isfile(path):
            continue

        if not path.lower().endswith(".json"):
            continue

        clean_paths.append(path)

    if not clean_paths:
        return 0, len(file_paths)

    def try_upload_with_mode(passive_mode):
        uploaded = 0
        failed = 0

        ftp = FTP()

        mode_name = "passive" if passive_mode else "active"

        try:
            try:
                ftp.connect(host, 21, timeout=timeout)
                ftp.login(user, password)
                ftp.set_pasv(passive_mode)
            except Exception as e:
                raise RuntimeError(
                    "FTP connection/login failed in {0} mode: {1}".format(
                        mode_name,
                        repr(e)
                    )
                )

            try:
                ftp.pwd()
                print("FTP login successful in {0} mode. Current remote directory is available.".format(
                    mode_name
                ))
            except Exception:
                print("FTP login successful in {0} mode. Could not read current remote directory.".format(
                    mode_name
                ))

            for i, local_path in enumerate(clean_paths, start=1):
                remote_name = os.path.basename(local_path)

                try:
                    with open(local_path, "rb") as f:
                        ftp.storbinary("STOR {0}".format(remote_name), f)

                    uploaded += 1

                except Exception as e:
                    failed += 1

                    if failed <= 5:
                        print(
                            "FTP upload failed in {0} mode for file #{1}/{2}: {3}".format(
                                mode_name,
                                i,
                                len(clean_paths),
                                repr(e)
                            )
                        )

                    # If the first upload times out, the connection is usually broken.
                    # Stop this mode and let the caller try the next mode.
                    if uploaded == 0 and failed == 1:
                        break

            if failed > 5:
                print("Additional FTP upload failures suppressed in {0} mode: {1}".format(
                    mode_name,
                    failed - 5
                ))

        finally:
            try:
                ftp.quit()
            except Exception:
                try:
                    ftp.close()
                except Exception:
                    pass

        return uploaded, failed

    # First try passive mode, the usual/default mode for GitHub-hosted runners.
    uploaded, failed = try_upload_with_mode(True)

    if uploaded > 0:
        # Count all files not uploaded as failures.
        total_failed = len(clean_paths) - uploaded
        return uploaded, total_failed

    print("Passive FTP uploaded 0 files. Trying active FTP mode...")

    uploaded2, failed2 = try_upload_with_mode(False)

    if uploaded2 > 0:
        total_failed = len(clean_paths) - uploaded2
        return uploaded2, total_failed

    # Both modes uploaded zero files.
    return 0, len(clean_paths)


# -------------------------------------------------------------------
# OPTIONAL PROCESSING TRIGGER
# -------------------------------------------------------------------

def trigger_processing(processing_cfg):
    """
    Optionally call a server-side processing endpoint.

    This is disabled unless processing.enabled is true in SOURCE_CONFIG_JSON.
    No sensitive URL or response body is printed.
    """

    enabled = bool(processing_cfg.get("enabled", False))

    if not enabled:
        return True

    url = processing_cfg.get("url", "")
    secret = processing_cfg.get("secret", "")

    if not url:
        raise RuntimeError("Processing is enabled but processing URL is missing.")

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    if secret:
        headers["X-Auth-Secret"] = secret

    try:
        response = requests.post(
            url,
            params={"action": "process"},
            headers=headers,
            timeout=120,
        )

        if response.status_code >= 400:
            print("Processing trigger failed with HTTP {0}.".format(response.status_code))
            return False

        print("Processing trigger completed with HTTP {0}.".format(response.status_code))
        return True

    except Exception:
        print("Processing trigger failed.")
        return False


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    cfg = load_config()

    ftp_cfg = cfg["ftp"]
    sources = cfg["sources"]

    fetch_cfg = cfg.get("fetch", {})
    fetch_headers = fetch_cfg.get("headers", {})

    verify_ssl = bool(fetch_cfg.get("verify_ssl", True))
    timeout_connect = int(fetch_cfg.get("timeout_connect", 10))
    timeout_read = int(fetch_cfg.get("timeout_read", 30))

    processing_cfg = cfg.get("processing", {})

    configure_ssl_warning_behavior(verify_ssl)

    output_dir = "downloaded_json"
    ensure_output_dir(output_dir)

    created_files = []
    failed_fetches = 0

    total_sources = len(sources)

    print("Starting fetch for {0} configured sources.".format(total_sources))

    for item in sources:
        filename = item["filename"]
        url = item["url"]

        try:
            content = fetch_one(
                url=url,
                headers=fetch_headers,
                verify_ssl=verify_ssl,
                timeout_connect=timeout_connect,
                timeout_read=timeout_read
            )

            path = save_file(
                output_dir=output_dir,
                filename=filename,
                content=content
            )

            created_files.append(path)

        except Exception:
            failed_fetches += 1

        time.sleep(0.1)

    print("Fetch complete. Created {0}/{1} files.".format(
        len(created_files),
        total_sources
    ))

    if not created_files:
        raise RuntimeError("No files were created. Not uploading.")

    uploaded, failed_uploads = upload_files_to_ftp(
        file_paths=created_files,
        ftp_cfg=ftp_cfg
    )

    print("FTP upload complete. Uploaded {0}/{1} files.".format(
        uploaded,
        len(created_files)
    ))

    processing_ok = True

    if uploaded > 0:
        processing_ok = trigger_processing(processing_cfg)

    if failed_fetches:
        print("Fetch failures: {0}".format(failed_fetches))

    if failed_uploads:
        print("FTP upload failures: {0}".format(failed_uploads))

    if not processing_ok:
        print("Processing trigger did not complete successfully.")

    if uploaded <= 0 or failed_uploads or not processing_ok:
        raise RuntimeError(
            "Run completed with problems: uploaded={0}, upload failure(s)={1}, processing_ok={2}.".format(
                uploaded,
                failed_uploads,
                processing_ok
            )
        )

    if failed_fetches:
        print(
            "Warning: {0} fetch failure(s), but all successfully fetched files were uploaded and processing completed.".format(
                failed_fetches
            )
        )

    print("All done.")


if __name__ == "__main__":
    main()
