#!/usr/bin/env bash
# Interactive visual review of cartilage connected components.
# Usage: bash postprocess_interactive.sh INPUT_MASK
set -euo pipefail

usage() {
    printf 'Usage: bash %s INPUT_MASK\n' "${0##*/}"
    printf 'Prepare review maps, wait for inspection, then ask which IDs to remove.\n'
    printf 'Outputs go into review/ beside the input mask; existing passes are preserved.\n'
    printf 'Requires the existing Python environment with NumPy, SciPy and nibabel.\n'
    printf 'Optional environment variable: PYTHON_BIN (default: python).\n'
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# != 1 )) || [[ -z ${1:-} ]]; then
    usage >&2
    exit 2
fi

CURRENT_MASK=$1
[[ "$CURRENT_MASK" == /* ]] || CURRENT_MASK="$PWD/$CURRENT_MASK"
WORK_DIR="$(dirname -- "$CURRENT_MASK")/review"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
PROCESSOR="$SCRIPT_DIR/postprocess_clusters.py"

if [[ ! -f "$CURRENT_MASK" || ! -r "$CURRENT_MASK" ]]; then
    printf 'Missing or unreadable mask: %s\n' "$CURRENT_MASK" >&2
    exit 1
fi
if [[ ! -f "$PROCESSOR" || ! -r "$PROCESSOR" ]]; then
    printf 'Missing or unreadable processor: %s\n' "$PROCESSOR" >&2
    exit 1
fi
if [[ -L "$WORK_DIR" || ( -e "$WORK_DIR" && ! -d "$WORK_DIR" ) ]]; then
    printf 'Expected a regular review directory beside the mask; conflicting path: %s\n' "$WORK_DIR" >&2
    exit 1
fi
command -v "$PYTHON_BIN" >/dev/null || {
    printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
}
mkdir -p -- "$WORK_DIR"

finish() {
    printf '\nStopped. Files already created are preserved in: %s\n' "$WORK_DIR"
    printf 'Latest completed mask (or original if no pass completed): %s\n' "$CURRENT_MASK"
    exit 0
}
trap 'printf "\nInterrupted; existing files are preserved in: %s\n" "$WORK_DIR"; exit 130' INT

# EOF never counts as permission to trim. q quits at any prompt.
ask() {
    local prompt=$1
    printf '%s' "$prompt"
    if ! IFS= read -r ANSWER; then
        finish
    fi
    if [[ ${ANSWER,,} == q || ${ANSWER,,} == quit ]]; then
        finish
    fi
}

read_ids() {
    local label=$1
    while true; do
        ask "Tissue $label: IDs to REMOVE (e.g. 2,5; Enter = none; q = quit): "
        ANSWER=${ANSWER//[[:space:]]/}
        if [[ -z "$ANSWER" || ${ANSWER,,} == none ]]; then
            IDS=none
            return
        fi
        if [[ "$ANSWER" =~ ^[1-7](,[1-7])*$ ]]; then
            IDS=$ANSWER
            return
        fi
        printf 'Use comma-separated IDs from 1 to 7, or press Enter for none.\n'
    done
}

PASS=1
while true; do
    # Reuse the review directory, but never reuse an occupied pass number.
    while [[ -e "$WORK_DIR/review_pass$PASS" || -L "$WORK_DIR/review_pass$PASS" ||
             -e "$WORK_DIR/cleaned_pass$PASS.nii.gz" || -L "$WORK_DIR/cleaned_pass$PASS.nii.gz" ||
             -e "$WORK_DIR/cleaned_pass$PASS.nii.gz.removals.json" || -L "$WORK_DIR/cleaned_pass$PASS.nii.gz.removals.json" ]]; do
        PASS=$((PASS + 1))
    done
    REVIEW_DIR="$WORK_DIR/review_pass$PASS"
    CLEANED_MASK="$WORK_DIR/cleaned_pass$PASS.nii.gz"
    printf '\n--- Preparing review pass %s ---\n' "$PASS"
    "$PYTHON_BIN" -u "$PROCESSOR" prepare \
        --input "$CURRENT_MASK" \
        --review-dir "$REVIEW_DIR"

    printf '\nOpen these review maps in your viewer over the MRI:\n'
    printf '  %s\n' "$REVIEW_DIR/label1_clusters.nii.gz" "$REVIEW_DIR/label2_clusters.nii.gz"
    printf 'Review IDs 1-6 are separate clusters, ranked largest first.\n'
    printf 'ID 7 groups the seventh-largest and ALL smaller clusters.\n'
    printf 'Selecting 7 removes that entire group. Keep it if you want to inspect it later.\n'
    printf 'Inspect the maps without editing or resaving them.\n'
    ask 'Press Enter when you have finished inspecting both maps (q = quit): '

    while true; do
        read_ids 1
        REMOVE1=$IDS
        read_ids 2
        REMOVE2=$IDS
        printf '\nProposed removal:\n  Tissue 1: %s\n  Tissue 2: %s\n' "$REMOVE1" "$REMOVE2"
        if [[ ",$REMOVE1," == *,7,* || ",$REMOVE2," == *,7,* ]]; then
            printf 'This selection includes ID 7: ALL components in that group will be removed.\n'
        fi
        printf 'New cleaned mask: %s\n' "$CLEANED_MASK"
        ask 'Apply these removals? [y/N] (q = quit; Enter = revise selections): '
        if [[ ${ANSWER,,} == y || ${ANSWER,,} == yes ]]; then
            break
        fi
        printf 'No changes applied. Enter your selections again.\n'
    done

    if ! "$PYTHON_BIN" -u "$PROCESSOR" apply \
        --input "$CURRENT_MASK" \
        --review-dir "$REVIEW_DIR" \
        --remove-label1 "$REMOVE1" \
        --remove-label2 "$REMOVE2" \
        --output "$CLEANED_MASK"; then
        printf '\nApply failed. Review files are preserved in: %s\n' "$REVIEW_DIR" >&2
        printf 'Use postprocess_clusters.py apply with that review directory to retry.\n' >&2
        exit 1
    fi
    CURRENT_MASK=$CLEANED_MASK
    printf '\nCompleted pass %s. Cleaned mask: %s\n' "$PASS" "$CURRENT_MASK"
    ask 'Prepare another review pass from this cleaned mask? [y/N] (q = quit): '
    if [[ ${ANSWER,,} != y && ${ANSWER,,} != yes ]]; then
        finish
    fi
    PASS=$((PASS + 1))
done
