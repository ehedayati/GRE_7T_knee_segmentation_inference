#!/usr/bin/env bash
# Run cartilage and joint models, correct menisci, merge, and repeat across echoes.
# Usage: bash run_combined_segmentation.sh INPUT_NIFTI OUTPUT_DIR [DEVICE] [MENISCUS_RADIUS_MM]
# Activate your segmentation environment first. All stages run without prompts.
set -euo pipefail

usage() {
    printf 'Usage: bash %s INPUT_NIFTI OUTPUT_DIR [DEVICE] [MENISCUS_RADIUS_MM]\n' "${0##*/}"
    printf 'Defaults to top3 UNet2 cartilage and five-loss joint inference; corrects meniscus residue, merges with overlaps = 5, then repeats across all echoes.\n'
    printf 'DEVICE defaults to cuda:0; MENISCUS_RADIUS_MM defaults to 1.0. PYTHON_BIN defaults to python.\n'
    printf 'Optional checkpoints: CARTILAGE_MODEL_PATH and MENISCUS_MODEL_PATH, with adjacent manifests.\n'
    printf 'Set ALL_ECHOES=0 to omit step 5; default ALL_ECHOES=1.\n'
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# < 2 || $# > 4 )) || [[ -z ${1:-} || -z ${2:-} ]]; then
    usage >&2
    exit 2
fi
INPUT=$1
OUT=$2
DEVICE=${3:-cuda:0}
PYTHON_BIN=${PYTHON_BIN:-python}
MENISCUS_RADIUS_MM=${4:-1.0}
ALL_ECHOES=${ALL_ECHOES:-1}
if [[ "$ALL_ECHOES" != 0 && "$ALL_ECHOES" != 1 ]]; then
    printf 'ALL_ECHOES must be 0 or 1.\n' >&2
    exit 2
fi
TOTAL_STEPS=$((4 + ALL_ECHOES))
if [[ ! "$MENISCUS_RADIUS_MM" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ || -z ${MENISCUS_RADIUS_MM//[0.]/} ]]; then
    printf 'MENISCUS_RADIUS_MM must be a positive decimal number.\n' >&2
    exit 2
fi
[[ "$INPUT" == /* ]] || INPUT="$PWD/$INPUT"
[[ "$OUT" == /* ]] || OUT="$PWD/$OUT"
JOINT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CART_DIR="$JOINT_DIR/../verified_cartilage_models_2026-11-10"
WEIGHTS_DIR="$JOINT_DIR/../weights"
CART_MODEL=${CARTILAGE_MODEL_PATH:-"$WEIGHTS_DIR/cartilage/top3_unet2_4echo.pt"}
JOINT_MODEL=${MENISCUS_MODEL_PATH:-"$WEIGHTS_DIR/joint/joint_unet_4echo_five_losses.pt"}
[[ "$CART_MODEL" == /* ]] || CART_MODEL="$PWD/$CART_MODEL"
[[ "$JOINT_MODEL" == /* ]] || JOINT_MODEL="$PWD/$JOINT_MODEL"
CART_MANIFEST="$(dirname -- "$CART_MODEL")/manifest.json"
JOINT_MANIFEST="$(dirname -- "$JOINT_MODEL")/manifest.json"
CART_OUTPUT="$OUT/cartilage_only.nii.gz"
MENISCUS_OUTPUT="$OUT/meniscus_only.nii.gz"
MERGED_OUTPUT="$OUT/merged_cartilage_meniscus.nii.gz"
REPORT_OUTPUT="$OUT/merged_cartilage_meniscus_merge.json"
ALL_ECHOES_OUTPUT="$OUT/merged_cartilage_meniscus_all_echoes.nii.gz"

# Preflight every dependency/output before either model runs.
EXTRA_DEPENDENCIES=()
EXTRA_OUTPUTS=()
if [[ "$ALL_ECHOES" == 1 ]]; then
    EXTRA_DEPENDENCIES+=("$JOINT_DIR/repeat_mask_across_echoes.py")
    EXTRA_OUTPUTS+=("$ALL_ECHOES_OUTPUT")
fi
for path in "$INPUT" "$CART_MODEL" "$JOINT_MODEL" \
    "$CART_MANIFEST" "$JOINT_MANIFEST" \
    "$CART_DIR/inference_single.py" "$JOINT_DIR/inference_single.py" \
    "$JOINT_DIR/correct_meniscus_residue.py" \
    "$JOINT_DIR/merge_cartilage_meniscus.py" \
    "${EXTRA_DEPENDENCIES[@]}"; do
    if [[ ! -f "$path" || ! -r "$path" ]]; then
        printf 'Missing or unreadable file: %s\n' "$path" >&2
        exit 1
    fi
done
for path in "$CART_OUTPUT" "$MENISCUS_OUTPUT" "$MERGED_OUTPUT" "$REPORT_OUTPUT" "${EXTRA_OUTPUTS[@]}"; do
    if [[ -e "$path" || -L "$path" ]]; then
        printf 'Output already exists: %s\nChoose a different output directory.\n' "$path" >&2
        exit 1
    fi
done
command -v "$PYTHON_BIN" >/dev/null || {
    printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
}
mkdir -p -- "$OUT"
# Only the joint network's raw output is temporary; preserve the final masks.
TEMP_DIR=$(mktemp -d "$OUT/.joint_inference.XXXXXX")
JOINT_OUTPUT="$TEMP_DIR/joint_seg.nii.gz"
cleanup() {
    rm -f -- "$JOINT_OUTPUT"
    rmdir -- "$TEMP_DIR"
}
trap cleanup EXIT

printf "\n[1/$TOTAL_STEPS] Cartilage segmentation, device %s\n" "$DEVICE"
printf 'Cartilage checkpoint: %s\n' "$CART_MODEL"
"$PYTHON_BIN" -u "$CART_DIR/inference_single.py" \
    --model-path "$CART_MODEL" \
    --input "$INPUT" \
    --output "$CART_OUTPUT" \
    --device "$DEVICE"

printf "\n[2/$TOTAL_STEPS] Joint/meniscus segmentation, device %s\n" "$DEVICE"
printf 'Meniscus checkpoint: %s\n' "$JOINT_MODEL"
"$PYTHON_BIN" -u "$JOINT_DIR/inference_single.py" \
    --model-path "$JOINT_MODEL" \
    --input "$INPUT" \
    --output "$JOINT_OUTPUT" \
    --device "$DEVICE"

printf "\n[3/$TOTAL_STEPS] Correct meniscus residue (radius %s mm, relabel residue; remove label-4 residue within 1 mm; restore clusters with two-sided label-3 contact)\n" "$MENISCUS_RADIUS_MM"
"$PYTHON_BIN" -u "$JOINT_DIR/correct_meniscus_residue.py" \
    --input "$JOINT_OUTPUT" \
    --radius-mm "$MENISCUS_RADIUS_MM" \
    --action relabel \
    --protect-label3 1 \
    --reference-label4 1 \
    --min-fraction 0.9 \
    --margin-mm 0.5 \
    --erase-label4-near-label3-mm 1.0 \
    --connectivity 26 \
    --output "$MENISCUS_OUTPUT" \
    --meniscus-only

printf "\n[4/$TOTAL_STEPS] Merge raw cartilage and corrected menisci; overlapping voxels become label 5\n"
"$PYTHON_BIN" -u "$JOINT_DIR/merge_cartilage_meniscus.py" \
    --meniscus-mask "$MENISCUS_OUTPUT" \
    --cartilage-mask "$CART_OUTPUT" \
    --output "$MERGED_OUTPUT" \
    --no-overlap-map

if [[ "$ALL_ECHOES" == 1 ]]; then
    printf '\n[5/5] Repeat merged segmentation across every echo of the input MRI\n'
    "$PYTHON_BIN" -u "$JOINT_DIR/repeat_mask_across_echoes.py" \
        --mask "$MERGED_OUTPUT" \
        --reference "$INPUT" \
        --output "$ALL_ECHOES_OUTPUT"
fi

printf '\nCompleted. Final 3D mask: %s\n' "$MERGED_OUTPUT"
if [[ "$ALL_ECHOES" == 1 ]]; then
    printf 'Final all-echo mask: %s\n' "$ALL_ECHOES_OUTPUT"
fi
printf 'Cartilage mask: %s\nMeniscus-only mask: %s\n' "$CART_OUTPUT" "$MENISCUS_OUTPUT"
printf 'Merge report: %s\n' "$REPORT_OUTPUT"
