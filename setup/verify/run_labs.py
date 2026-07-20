#!/usr/bin/env python3
"""Execute the course's jupytext .py files into .ipynb with inline output.

For each ``labs/*.py`` (or ``solutions/*.py``) this converts the jupytext
percent-format script to a notebook, runs every cell with nbclient's
``allow_errors=True`` (so a failing cell never stops the run -- its traceback
is embedded inline instead), writes the executed ``.ipynb`` next to the ``.py``,
and prints a per-file pass/error summary plus a JSON report.

It is the batch companion to ``setup/test_environment.py``: that checks the
environment can import things; this checks every lab actually runs end to end.

Usage
-----
    python setup/verify/run_labs.py                 # all labs
    python setup/verify/run_labs.py 05_llm_classification.py
    python setup/verify/run_labs.py --dir solutions # all solutions
    python setup/verify/run_labs.py --smoke         # cap live API cost (see below)

Environment
-----------
    KERNEL_NAME    Jupyter kernel to run against (default: python3). Point this
                   at the uv verification kernel, e.g. KERNEL_NAME=icpsr-uv.
    CELL_TIMEOUT   Per-cell timeout in seconds (default: 900).
    OPENROUTER_API_KEY   Read from the environment, or from ``openrouter.txt``
                   at the repo root if present (gitignored; never committed).

--smoke
-------
Sets ICPSR_SMOKE_TEST=1. Labs that honor it (currently Lab 5) make only a
couple of real API calls to prove the live path works, then load cached
results for the rest -- so an automated re-render doesn't pay to label a whole
corpus. Without the flag everything runs fully live.
"""
import os
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]

# Inject the OpenRouter key from openrouter.txt (gitignored) if not already set.
_key_file = REPO / "openrouter.txt"
if "OPENROUTER_API_KEY" not in os.environ and _key_file.exists():
    os.environ["OPENROUTER_API_KEY"] = _key_file.read_text().strip()

CELL_TIMEOUT = int(os.environ.get("CELL_TIMEOUT", "900"))
KERNEL = os.environ.get("KERNEL_NAME", "python3")


def run_one(py_path, work_dir):
    """Execute one .py, write the .ipynb beside it, return a result record."""
    py_path = Path(py_path)
    ipynb_path = py_path.with_suffix(".ipynb")
    rec = {"file": py_path.name, "ipynb": str(ipynb_path), "errors": [],
           "status": "ok", "elapsed": None, "n_cells": 0}
    nb = None
    t0 = time.time()
    try:
        nb = jupytext.read(py_path)
        # Run against a real kernel; cwd = the lab's own dir so each lab's
        # COURSE_DIR logic (which keys off the directory basename) resolves.
        nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": KERNEL,
                                     "language": "python"}
        NotebookClient(
            nb, timeout=CELL_TIMEOUT, allow_errors=True, kernel_name=KERNEL,
            resources={"metadata": {"path": str(work_dir)}},
        ).execute()

        rec["n_cells"] = sum(1 for c in nb.cells if c.cell_type == "code")
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    rec["errors"].append({
                        "ename": out.get("ename"),
                        "evalue": out.get("evalue"),
                        "source_head": "\n".join(
                            cell.get("source", "").splitlines()[:6]),
                    })
        if rec["errors"]:
            rec["status"] = "errors"
    except Exception as e:  # kernel death, timeout, etc.
        rec["status"] = "fatal"
        rec["fatal"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()
    finally:
        if nb is not None:  # keep partial output so you can see how far it got
            try:
                nbformat.write(nb, ipynb_path)
            except Exception:
                pass
        rec["elapsed"] = round(time.time() - t0, 1)
    return rec


def main():
    args = sys.argv[1:]
    # Mark this as an automated run: labs with deliberate participation gates
    # (Lab 5's codebook-drafting assert) skip the gate when this is set, so
    # unattended execution doesn't stop at a cell that expects a human.
    os.environ.setdefault("ICPSR_AUTOMATED", "1")
    if "--smoke" in args:
        args.remove("--smoke")
        os.environ["ICPSR_SMOKE_TEST"] = "1"
        print("SMOKE MODE: labs honoring ICPSR_SMOKE_TEST will cap live API calls",
              flush=True)

    subdir = "labs"
    if "--dir" in args:
        i = args.index("--dir")
        subdir = args[i + 1]
        del args[i:i + 2]
    work_dir = REPO / subdir

    if args:
        targets = [work_dir / a for a in args]
    else:
        targets = sorted(work_dir.glob("*.py"))

    if not targets:
        print(f"No .py files found in {work_dir}", flush=True)
        sys.exit(1)

    report = []
    for py in targets:
        print(f"\n{'=' * 70}\nEXECUTING {py.name}\n{'=' * 70}", flush=True)
        rec = run_one(py, work_dir)
        report.append(rec)
        print(f"  -> {rec['status']} ({rec['elapsed']}s, {rec['n_cells']} code "
              f"cells, {len(rec['errors'])} error cell(s))", flush=True)
        for err in rec["errors"]:
            print(f"     ! [{err['ename']}] {str(err['evalue'])[:110]}", flush=True)
        if rec["status"] == "fatal":
            print("     FATAL:", rec.get("fatal"), flush=True)

    out = Path(tempfile.gettempdir()) / "icpsr_lab_run_report.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'#' * 70}\nSUMMARY  (report: {out})\n{'#' * 70}", flush=True)
    for rec in report:
        print(f"{rec['file']:32} {rec['status']:8} {rec['elapsed']}s  "
              f"errors={len(rec['errors'])}", flush=True)

    # non-zero exit if anything failed, so CI / && chains notice
    if any(r["status"] != "ok" for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
