# Lab verification harness

Tools for the **clean-environment sweep**: build a fresh, reproducible
virtualenv and render every lab notebook end-to-end, so you know the labs run
as written before a workshop — not just in whatever env you happen to have.

This complements `setup/test_environment.py` (which checks that imports work).
Here we actually *run* every lab.

## Quick start

```bash
# One command: create the uv env, register a kernel, audit, render all labs.
setup/verify/verify.sh            # smoke mode (caps live API cost — see below)
setup/verify/verify.sh --full     # fully live (labs 5/6 hit the OpenRouter API)
```

Requires [`uv`](https://docs.astral.sh/uv/). For the API labs, put an
`OPENROUTER_API_KEY` in your environment or an `openrouter.txt` at the repo root
(gitignored — never commit it).

The executed notebooks are written next to each `.py` as `labs/*.ipynb`
(gitignored). Open them to review the inline output; a pass/error summary and a
JSON report are printed at the end.

## The pieces

| File | What it does |
|------|--------------|
| `verify.sh` | End-to-end wrapper: build venv → install → register kernel → audit → render. |
| `run_labs.py` | Executes each `labs/*.py` into an `.ipynb` with `allow_errors=True`, so a failing cell embeds its traceback inline instead of stopping the run. |
| `import_audit.py` | Static check: flags any package a lab **imports but never `!pip install`s** (the "Lab 1 forgot openai" bug), deterministically, regardless of the current env. |
| `verify_requirements.txt` | Union of every lab's declared installs + `ipykernel`, for the verification venv. Uses default PyPI indexes (resolves on macOS too), unlike the Linux-hub-tuned `../requirements.txt`. |

## Running pieces individually

```bash
# Point the runner at the uv kernel (created by verify.sh):
KERNEL_NAME=icpsr-uv python3 setup/verify/run_labs.py                 # all labs
KERNEL_NAME=icpsr-uv python3 setup/verify/run_labs.py 05_llm_classification.py
KERNEL_NAME=icpsr-uv python3 setup/verify/run_labs.py --dir solutions # solutions/

python3 setup/verify/import_audit.py            # static audit only, no env needed
```

`run_labs.py` reads two env vars: `KERNEL_NAME` (default `python3`) and
`CELL_TIMEOUT` (seconds, default 900).

## `--smoke` mode

Rendering the LLM labs live costs API calls and time (Lab 5 labels a 119-doc
corpus twice). `--smoke` sets `ICPSR_SMOKE_TEST=1`; labs that honor it make only
a couple of real calls to prove the live path works, then load cached results
for the rest. So a re-render stays cheap (~1 min for Lab 5) while still
exercising the real API and producing a complete-looking notebook.

Currently only **Lab 5** honors the flag (its cache fixture is
`data/cached/lab5_labels_cache.parquet`). Labs 1 and 6 make only a handful of
calls and always run live. Use `--full` (no smoke) to render everything live.

> **Note:** in smoke mode Lab 5's accuracy figures come from the cache, not
> fresh labeling. The numbers are real (from a prior full live run) and the
> live `else:` branch is the same code students run — but if you want the
> shipped notebook to show freshly-computed numbers, render with `--full`.
