#!/usr/bin/env bash
# Uses the active environment; no Conda environment is activated automatically.
set -euo pipefail
RELEASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$RELEASE_DIR/verified_meniscus_and_cartilage_models_2026-11-11/run_combined_segmentation.sh" "$@"
