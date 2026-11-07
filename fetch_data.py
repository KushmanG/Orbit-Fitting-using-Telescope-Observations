#!/usr/bin/env python3
"""
!! THIS FILE NEEDS INTERNET ACCESS TO RUN SINCE IT FETCHES DATA FROM URLs ON THE INTERNET

- This file downloads the raw telescopic data for the Eros orbit-fitting project.
- Everything is stored in ./data/raw/
- No need to pip install anything
- Run it using: python3 fetch_data.py

What we download and why (both files come from the Minor Planet Center, MPC):

  MPC_Observations_433Eros.json     Every published telescopic observation of 433 Eros, from
                                    the MPC get-obs dataset.
                                    API: https://data.minorplanetcenter.net/api/get-obs

  MPC_Observatory_Codes.json        MPC list of observatory codes with their parallax constants
                                    This tells us the position of each telescope used for observation
                                    in this dataset
                                    File: https://minorplanetcenter.net/Extended_Files/obscodes_extended.json.gz
"""
import gzip
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

# API URLS and file names:
mpc_api = "https://data.minorplanetcenter.net/api/get-obs"

mpc_path = "MPC_Observations_433Eros.json"
obscodes_path = "MPC_Observatory_Codes.json"

# Gets the list of trusted certificate authorities from the ceritifi library
# Using it, python checks if the https:/ site is secure
def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

CTX = _ssl_context()

# Setting directory Aliases
here = os.path.dirname(os.path.abspath(__file__)) # Current Directory
raw = os.path.join(here, "data", "raw")           # Storage Directory


# Timeout and Tries are totally Arbitrary values
'''
    This Internet Fetch function is written by Claude Opus 5.5
    It's task is to request the required data from the required API URLs
'''
def fetch(url, json_body=None, timeout=300, tries=3):
    """GET a URL (optionally with a JSON body, which the MPC API wants) and return bytes."""
    data = None
    headers = {"User-Agent": "orbitfit-student-project (IIIT Bangalore, AIT 203)"}
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="GET")
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.read()
        except urllib.error.URLError as e:
            last = e
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                sys.exit(
                    "\nSSL certificate error. On a Mac with python.org Python, run\n"
                    "  /Applications/Python\\ 3.*/Install\\ Certificates.command\n"
                    "(or: pip3 install certifi) and then run this script again.\n")
            print(f"    attempt {k + 1} failed: {e}; retrying...")
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"could not download {url}: {last}")

# Save some continuous data "blob" in data/raw/file
def save(name, blob):
    path = os.path.join(raw, name)
    with open(path, "wb") as f:
        f.write(blob)
    print(f"    Saved {name}  ({len(blob) / 1e6:.2f} MB)")
    return path

# 1. MPC observations -------------------------------------------------------
def mpc_obs():
    blob = fetch(mpc_api, {"desigs": ["433"], "output_format": ["OBS80"]})
    save(mpc_path, blob)
    reply = json.loads(blob)
    obs80 = reply[0].get("OBS80", "") if isinstance(reply, list) else reply.get("OBS80", "")
    print(f"    {sum(len(l) >= 80 for l in obs80.splitlines())} observation lines in OBS80 format")

# 2. Observatory codes ------------------------------------------------------
def obscodes():
    from mpc_obscodes import mpc_obscodes
    blob = mpc_obscodes.read_bytes()
    save(obscodes_path, blob)
    print(f"    {len(json.loads(blob))} observatory codes")

def main():
    os.makedirs(raw, exist_ok=True)

    print("[1/2] MPC observations of 433 Eros (this one is big, give it a minute)")
    mpc_obs()

    print("[2/2] MPC observatory codes")
    obscodes()

    # Write Manifest: size + checksum of each file, so anyone can check they have the same data
    manifest = {}
    for name in (mpc_path, obscodes_path):
        path = os.path.join(raw, name)
        manifest[name] = {"Size": os.path.getsize(path),
                          "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    manifest["Download_Time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    # Dump the Manifest Dict to a json file
    with open(os.path.join(raw, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Finishing Log
    print(f"\nRaw Dataset is ready in data/raw/")


# Run the Driver
if __name__ == "__main__":
    main()
