#!/usr/bin/env bash
# Shared interactive cleanup, configured for both cartilage and both menisci.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export POSTPROCESS_LABELS="1 2 3 4"
exec bash "$SCRIPT_DIR/../segmentation_postprocessing/postprocess_interactive.sh" "$@"
