#!/usr/bin/env bash
# EdgeGuard setup script (HP Edge AI SJSUHack 2026 deliverable).
#
# Creates a Python virtual environment and installs the exact dependencies
# EdgeGuard needs to train, score, test and demo -- no numpy, no compiled
# ML libraries, nothing that needs a build toolchain on ARM (the ZGX Nano
# is aarch64): pydantic, fastapi, uvicorn, httpx, pytest (see
# requirements.txt). The dashboard (dashboard/) is a separate Node project;
# see dashboard/README.md for its own setup.
#
# Usage:
#   bash setup.sh
#   source .venv/bin/activate
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "Using $($PYTHON --version)"
$PYTHON -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Then, e.g.:"
echo "  python -m pytest -q                                     # run the test suite (fake data only)"
echo "  python -m defender.run_training --max-false-alarm-rate 0.01 --data-dir <path-to-road>"
echo "  python -m integration.run_demo --model-version v2 --data-dir <path-to-road>"
