#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$(command -v python)}"
exec "$PYTHON" "$ROOT/production_submit.py" --input "$ROOT/configs/production.json"
