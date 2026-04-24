"""Probe public mirrors of the NEU dataset (no auth)."""

import ssl
import urllib.request

ctx = ssl.create_default_context()
URLS = [
    "https://huggingface.co/datasets/keremberke/neu-metal-surface-defects/resolve/main/data/train.zip",
    "https://github.com/abin24/Surface-Inspection-defect-detection-dataset/raw/master/NEU-CLS.zip",
    "https://huggingface.co/datasets/Francesco/neu-defects/resolve/main/data/train-00000-of-00001.parquet",
]

for url in URLS:
    try:
        req = urllib.request.Request(url, method="HEAD")
        r = urllib.request.urlopen(req, timeout=15, context=ctx)
        size = r.headers.get("content-length") or "?"
        print(f"OK   {r.status} size={size}  {url}")
    except Exception as e:
        print(f"FAIL {type(e).__name__:20s}        {url}  ({e})")
