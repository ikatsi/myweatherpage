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
    """
    if value is None:
        return

    value = str(value)

    if value:
        print("::add-mask::{}".format(value))


def mask_config(cfg):
    """
    Mask all sensitive values contained in the config.
    """

    ftp_cfg = cfg.get("ftp", {})
    fetch_cfg = cfg.get("fetch", {})
    fetch_headers = fetch_cfg.get("headers", {})
    sources = cfg.get("sources", [])

    for key in ["host", "user", "pass", "remote_dir"]:
        mask_value(ftp_cfg.get(key, ""))

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


def fetch_one(url, headers, verify_ssl, timeout_connect, timeout_read, api_token):
    request_headers = dict(headers)

    if api_token:
        request_headers["Authorization"] = "Bearer {0}".format(api_token)

    response = requests.get(
        url,
        headers=request_headers,
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
    Upload ONLY JSON files created by this script by FTP.

    Files are uploaded to the FTP login directory unless ftp.remote_dir is set
    in SOURCE_CONFIG_JSON.
    """

    host = ftp_cfg.get("host", "")
    user = ftp_cfg.get("user", "")
    password = ftp_cfg.get("pass", "")
    timeout = int(ftp_cfg.get("timeout", 60))
    remote_dir = ftp_cfg.get("remote_dir", "")

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

    uploaded = 0
    failed = 0

    ftp = FTP()

    try:
        ftp.connect(host, 21, timeout=timeout)
        ftp.login(user, password)
        ftp.set_pasv(True)

        if remote_dir:
            ftp.cwd(remote_dir)

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
                        "FTP upload failed for file #{0}/{1}: {2}".format(
                            i,
                            len(clean_paths),
                            repr(e)
                        )
                    )

        if failed > 5:
            print("Additional FTP upload failures suppressed: {0}".format(failed - 5))

    finally:
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass

    return uploaded, failed


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    cfg = load_config()

    api_token = os.environ.get("API_TOKEN", "")

    if api_token:
        mask_value(api_token)

    ftp_cfg = cfg["ftp"]
    sources = cfg["sources"]

    fetch_cfg = cfg.get("fetch", {})
    fetch_headers = fetch_cfg.get("headers", {})

    verify_ssl = bool(fetch_cfg.get("verify_ssl", True))
    timeout_connect = int(fetch_cfg.get("timeout_connect", 10))
    timeout_read = int(fetch_cfg.get("timeout_read", 30))

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
                timeout_read=timeout_read,
                api_token=api_token
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

    if failed_fetches:
        print("Fetch failures: {0}".format(failed_fetches))

    if failed_uploads:
        print("FTP upload failures: {0}".format(failed_uploads))

    if uploaded <= 0 or failed_uploads:
        raise RuntimeError(
            "Run completed with problems: uploaded={0}, upload failure(s)={1}.".format(
                uploaded,
                failed_uploads
            )
        )

    if failed_fetches:
        print(
            "Warning: {0} fetch failure(s), but all successfully fetched files were uploaded.".format(
                failed_fetches
            )
        )

    print("All done.")


if __name__ == "__main__":
    main()
