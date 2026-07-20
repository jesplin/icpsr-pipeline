#!/usr/bin/env bash
# Build a clean uv virtualenv, register it as a Jupyter kernel, and render every
# lab notebook end-to-end. This is the clean-environment sweep in one command.
#
# Usage:
#   setup/verify/verify.sh              # build env + render all labs (smoke mode)
#   setup/verify/verify.sh --full       # render fully live (labs 5/6 hit the API)
#   setup/verify/verify.sh --dir solutions
#
# Requirements: uv (https://docs.astral.sh/uv/) and, for the API labs, an
# OPENROUTER_API_KEY in the environment or an openrouter.txt at the repo root.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="$REPO/.venv"
KERNEL="icpsr-uv"

SMOKE="--smoke"
DIR_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --full) SMOKE="" ;;
    *) DIR_ARGS+=("$arg") ;;
  esac
done

echo ">> Creating uv venv at $VENV (Python 3.11)"
uv venv "$VENV" --python 3.11 --seed

echo ">> Installing verification dependencies"
uv pip install --python "$VENV/bin/python" -r "$HERE/verify_requirements.txt"

echo ">> Registering Jupyter kernel '$KERNEL'"
"$VENV/bin/python" -m ipykernel install --user --name "$KERNEL" \
  --display-name "ICPSR (uv)"

echo ">> Static import-vs-install audit"
python3 "$HERE/import_audit.py" "${DIR_ARGS[@]}" || true

echo ">> Rendering notebooks (kernel=$KERNEL ${SMOKE:-full})"
KERNEL_NAME="$KERNEL" python3 "$HERE/run_labs.py" $SMOKE "${DIR_ARGS[@]}"
