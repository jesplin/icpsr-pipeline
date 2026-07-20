"""
Pre-workshop environment test for the ICPSR Social Science Data Pipeline course.

Run this from the repo root:

    python setup/test_environment.py

It checks that (1) every package the labs need imports, (2) your OpenRouter
key works, with a one-token call that costs a fraction of a cent, and
(3) every cached dataset loads. Screenshot the output and email it to the
instructor before Day 1. If anything prints FAIL, see the participant setup
guide, then email if you're stuck.
"""

import importlib
import os
import sys

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

failures = 0


def report(ok, label, detail=""):
    global failures
    print(f"  [{PASS if ok else FAIL}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures += 1


# ---------------------------------------------------------------------------
print("\n1. Package imports")
# ---------------------------------------------------------------------------

PACKAGES = [
    # (import name, pip name if different)
    ("pandas", None),
    ("numpy", None),
    ("requests", None),
    ("bs4", "beautifulsoup4"),
    ("openai", None),
    ("pyarrow", None),
    ("matplotlib", None),
    ("sklearn", "scikit-learn"),
    ("tqdm", None),
    ("dateparser", None),
    ("geopandas", None),
    ("rioxarray", None),
    ("shapely", None),
    ("plotly", None),
    ("PIL", "pillow"),
    ("transformers", None),
    ("torch", None),
    ("telethon", None),
    ("jupytext", None),
]

for mod, pip_name in PACKAGES:
    try:
        importlib.import_module(mod)
        report(True, mod)
    except ImportError:
        report(False, mod, f"pip install {pip_name or mod}")

# ---------------------------------------------------------------------------
print("\n2. OpenRouter API")
# ---------------------------------------------------------------------------

if "OPENROUTER_API_KEY" not in os.environ:
    report(False, "OPENROUTER_API_KEY environment variable",
           "set it with: export OPENROUTER_API_KEY=sk-or-...")
else:
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
        resp = client.chat.completions.create(
            model="qwen/qwen3-30b-a3b",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=1,
        )
        report(True, "OpenRouter call", f"model replied, {resp.usage.total_tokens} tokens")
    except Exception as e:
        report(False, "OpenRouter call", str(e)[:120])

# ---------------------------------------------------------------------------
print("\n3. Cached datasets")
# ---------------------------------------------------------------------------

# (path, loader) — one representative cache per lab; every lab must be
# able to start offline from these files.
import pandas as pd


def load_geojson(path):
    import geopandas as gpd
    return gpd.read_file(path)


def load_html(path):
    return open(path).read()


def load_raster(path):
    import rioxarray
    return rioxarray.open_rasterio(path)


CACHED = [
    ("data/cached/reliefweb_html/listing_page_0.html", load_html),       # Lab 1
    ("data/cached/telegram_infra.parquet", pd.read_parquet),             # Lab 2
    ("data/cached/telegram_channels.parquet", pd.read_parquet),          # Lab 9 (synthetic)
    ("data/cached/ucdp_ged_kharkiv_may2024.csv", pd.read_csv),           # Labs 2, 8
    ("data/cached/firms_fires.csv", pd.read_csv),                        # Lab 3
    ("data/cached/deepstate_control.geojson", load_geojson),             # Lab 3
    ("data/cached/sentinel2_kharkiv_clip.tif", load_raster),             # Lab 3
    ("data/cached/eurosat_index.csv", pd.read_csv),                      # Lab 4
    ("data/cached/ukraine_text_sample.parquet", pd.read_parquet),        # Lab 5
    ("data/cached/vlm_images/ground_truth.csv", pd.read_csv),            # Lab 6
    ("data/cached/ukraine_text_labeled.parquet", pd.read_parquet),       # Lab 7
    ("data/cached/lab7_llm_labels.parquet", pd.read_parquet),            # Lab 7
    ("data/cached/lab7_perturbations.parquet", pd.read_parquet),         # Lab 7
    ("data/cached/viina_kharkiv_may2024.csv", pd.read_csv),              # Lab 8
]

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for rel_path, loader in CACHED:
    path = os.path.join(root, rel_path)
    if not os.path.exists(path):
        report(False, rel_path, "file missing — did you clone the full repo?")
        continue
    try:
        obj = loader(path)
        n = len(obj)
        report(True, rel_path, f"{n} rows")
    except Exception as e:
        report(False, rel_path, str(e)[:120])

# ---------------------------------------------------------------------------
print()
if failures == 0:
    print(f"All checks passed. See you on Day 1!")
else:
    print(f"{failures} check(s) failed. See setup/participant_setup.qmd, "
          "then email the instructor with a screenshot.")
    sys.exit(1)
