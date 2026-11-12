# Knee segmentation release

Standalone inference application for knee cartilage and meniscus segmentation.
This directory is the application/repository root. Keep its files and
subdirectories together; paths are relative to the application, and the research
checkout is not needed. Trained weights are distributed separately and excluded
from Git.

## Installation

Install MRIForge with I/O and PyTorch support, plus nvidia-ml-py:

```bash
python -m pip install "mriforge[io,torch]" nvidia-ml-py
```

Alternatively, install all MRIForge extras:

```bash
python -m pip install "mriforge[all]" nvidia-ml-py
```

Both options include the dependencies required for segmentation. nvidia-ml-py
provides GPU-utilization reporting. MONAI is not required.

For GPU inference, use a CUDA-enabled PyTorch installation compatible with the
server's NVIDIA driver.

## Model weights

Obtain the trained checkpoints separately and copy them into these paths
relative to the application root. The default workflow requires:

- `verified_cartilage_models_2026-11-10/top3_unet2_4echo.pt`
- `verified_meniscus_and_cartilage_models_2026-11-11/joint_unet_4echo_five_losses.pt`

Optional alternative cartilage checkpoints:

- `verified_cartilage_models_2026-11-10/top1_unet_4echo.pt`
- `verified_cartilage_models_2026-11-10/top2_unet2_3echo.pt`

Keep the supplied manifest.json beside each set of weights. Inference verifies
the selected checkpoint's SHA-256 against its manifest. The .gitignore keeps
these weights local; Python network definitions remain part of the repository.

## Start the browser UI

From the release directory on the server:

```bash
python start_ui.py
```

Optional initial browser directory and port:

```bash
python start_ui.py --start-dir /path/to/MRI_data --port 8501
```

Keep the terminal running and open the FULL printed URL, including its token.
For a remote server, run this in a separate terminal on your workstation:

```bash
ssh -N -L 8501:127.0.0.1:8501 USER@SERVER
```

Replace USER@SERVER with your normal SSH destination. Then open the printed
localhost URL in your workstation browser. Ctrl+C stops the UI and its active run.

The UI provides:

- NIfTI file or GRE DICOM series-folder selection on server storage.
- Automatic MRIForge conversion before segmentation for DICOM input.
- Output parent folder selection plus a separate new results folder name.
- Visible-GPU availability and a populated compute-device dropdown. Auto selects
  the visible GPU with the most free memory when the run starts; CPU is supported.
- Meniscus correction radius and optional all-echo output.
- Progress, logs, stop control and output paths.
- A separate Networks panel below Progress, with Cartilage above Meniscus.
  The networks can be changed independently; current defaults stay selected
  unless you manually change them. Custom weights require matching manifests.

The UI uses the active Python interpreter for all processing. GPU visibility is
inherited from the launching session; it does not acquire a scheduler allocation.
Free memory is a snapshot, not a reservation. Utilization is shown when available.

## GRE DICOM input

In the UI, choose **GRE DICOM series folder** under Input format. Browse to the
folder containing one GRE magnitude series with all its echoes, then select
**Use this DICOM series**. Choose your output parent and new results folder
name, then run segmentation.

MRIForge converts the series first. The UI saves these files in the results
folder and uses that NIfTI for both networks and the all-echo mask:

- `converted_input/gre.nii.gz`
- `converted_input/gre.json` (MRIForge metadata)

Conversion appears as the first progress step and supports Stop run. The
existing cartilage, joint, correction, merge and optional all-echo stages then
follow. DICOM input has six steps with all-echo output, or five without it.

Select the actual series folder containing the files, rather than a parent
patient/study folder. Mixed series are rejected instead of silently choosing
one. Converted images must contain enough echoes for both selected networks
and meet their spatial size requirements. The defaults require at least seven
echoes and X/Y >=512. Conversion failures stop the run before inference.

**Always use the MRIForge DICOM-to-NIfTI converter for compatible results, even
when preparing NIfTI input separately.** Keep its echo ordering and geometry;
use the resulting NIfTI directly, without additional reorientation or resampling.

For manual conversion with MRIForge:

```bash
mriforge-dicom-to-nifti /path/to/GRE_DICOM_series /path/to/multi_echo_GRE.nii.gz
```

See the [MRIForge conversion documentation](https://pypi.org/project/mriforge/).
The release adapter is
`verified_meniscus_and_cartilage_models_2026-11-11/convert_gre_dicom.py`;
it calls `mrif.utilities.io.dicom_to_nifti` in the UI's Python environment.
The UI leaves source DICOM files unchanged and preserves the converted files.

## Run the pipeline without the UI

The Bash launcher takes a NIfTI file. For DICOM input, convert with MRIForge
first using the command above, or use the UI for automatic conversion.

```bash
bash run_segmentation.sh /path/to/multi_echo_GRE.nii.gz /path/to/new_results
```

Optional positional arguments: DEVICE (default cuda:0) and correction radius in
mm (default 1.0):

```bash
bash run_segmentation.sh /path/scan.nii.gz /path/new_results cuda:1 1.0
```

ALL_ECHOES=0 omits the fifth step. PYTHON_BIN selects a Python executable.
CARTILAGE_MODEL_PATH and MENISCUS_MODEL_PATH can independently override the
checkpoint paths; each needs manifest.json beside it. UI auto-device selection
is performed by the UI; the Bash/standalone inference defaults remain unchanged.

Stages:

1. Cartilage-only inference: default top3 UNet2, echoes [0,2,4,6].
2. Joint inference for menisci: default five-loss UNet, echoes [0,2,4,6].
3. Automatic meniscus correction and extraction of labels 3/4.
4. Merge with cartilage labels 1/2, flagging overlaps as label 5.
5. Optional repetition of the merged mask across every input echo.

Saved outputs:

- cartilage_only.nii.gz
- meniscus_only.nii.gz
- merged_cartilage_meniscus.nii.gz
- merged_cartilage_meniscus_all_echoes.nii.gz (when enabled)
- merged_cartilage_meniscus_merge.json

Existing output files are refused. The raw joint prediction is temporary and is
removed when the launcher exits. No residue-review directory is generated by
the combined pipeline. Keep the 3-D masks for further processing; the repeated
4-D mask is for viewing on all echoes.

## Current automatic meniscus treatment

Trusted references default to the largest original component of each meniscus.
The script uses 26-connectivity and physical voxel spacing. It first corrects
label-3 residue toward trusted label 4, then inspects original label-4 residue
near trusted label 3. References are fixed before either pass.

Proximity relabelling requires 90% of component voxels to be within the chosen
radius and closer to the opposite trusted meniscus by more than 0.5 mm.

An additional rule selects whole unprotected original label-4 clusters if ANY
part comes within 1.0 mm of trusted label 3. A selected cluster is removed, then
restored WHOLE as label 3 if it touches corrected label 3 from at least TWO of
six face directions. The directions can be adjacent or opposite; diagonal-only
contacts do not count. Contacts are measured before repair, so filling cannot
spread iteratively. Clusters with fewer contacts stay background. Cartilage,
original background and both trusted menisci are protected.

To rerun this on an available original joint mask without inference:

```bash
python verified_meniscus_and_cartilage_models_2026-11-11/correct_meniscus_residue.py \
  --input /path/original_joint_mask.nii.gz --radius-mm 1.0 \
  --output /path/new_meniscus_only.nii.gz --meniscus-only
```

The standalone script also supports optional review outputs via --output-dir,
custom trusted component ranks, --one-way, --no-hole-repair, and
--erase-label4-near-label3-mm (0 disables the additional removal rule).
Use --help for the complete interface.

Repeat only the merge after optional cleanup:

```bash
python verified_meniscus_and_cartilage_models_2026-11-11/merge_cartilage_meniscus.py \
  --cartilage-mask /path/cartilage_only.nii.gz \
  --meniscus-mask /path/meniscus_only.nii.gz \
  --output /path/new_merged.nii.gz --no-overlap-map
```

Repeat an updated merged mask across echoes:

```bash
python verified_meniscus_and_cartilage_models_2026-11-11/repeat_mask_across_echoes.py \
  --mask /path/new_merged.nii.gz --reference /path/original_MRI.nii.gz
```

## Optional interactive cluster trimming

```bash
bash verified_meniscus_and_cartilage_models_2026-11-11/postprocess_interactive.sh /path/mask.nii.gz
```

This creates review/ beside the mask and waits for visual inspection of the
four numbered tissue maps. IDs 1-6 are individual clusters; ID 7 groups the rest.
It asks which IDs to remove from each tissue and records each cleaned pass.
The shared implementation is in segmentation_postprocessing/. Cartilage-only
trimming is also available in verified_cartilage_models_2026-11-10/.
These tools do not resolve merged overlap label 5.

## Models and manifests

Supported cartilage checkpoint filenames (distributed separately):

- top3_unet2_4echo.pt (default)
- top1_unet_4echo.pt
- top2_unet2_3echo.pt

Supported joint checkpoint filename (distributed separately):

- joint_unet_4echo_five_losses.pt (default)

The rejected three-loss joint model is excluded from this release and its
release catalog entry removed. The originals remain in the research directory.
CSV training configurations, notebooks, datasets and old training code are not
required or included. Settings and checkpoint hashes come from manifest.json.
The UI catalogs all retained entries. Do not rename weight files without also
updating their manifest entries.

Both runners use the shared `torch_unet.py` class in the release root. It
reproduces the archived MONAI UNet/UNet2 definitions using standard PyTorch
layers: three output classes for cartilage, five for joint/meniscus. MONAI is
not imported or required at runtime. Existing checkpoints load strictly without
conversion or retraining. Keep `torch_unet.py` with the release.

This adaptation preserves module names, residual/skip paths, padding,
InstanceNorm3d and scalar PReLU. MONAI attribution and the Apache 2.0 license
are in THIRD_PARTY_NOTICES.md and licenses/MONAI-LICENSE.txt.

Strict loading of all four checkpoints and exact synthetic CPU output parity
against MONAI were checked locally. Full-volume server inference with the
updated release remains to be verified; the original verified runners still
use MONAI and have not been changed.

Binary meniscus-only networks and arbitrary architectures require adapters. Custom manifests must also specify
residual units, echoes, normalization, crop/window sizes and SHA-256.

## Input and system requirements

- GRE magnitude DICOM series folder (UI), or original 4-D GRE NIfTI with echoes
  last. Always use MRIForge for DICOM-to-NIfTI conversion and retain its layout.
- At least seven echoes for the default networks, and X/Y dimensions >=512.
- MRI/masks on matching spatial grids. No registration/resampling is performed.
- Valid spatial units for distance-based correction; the standalone correction
  accepts explicit --spacing-mm when units are unknown.
- Labels: 0 background, 1 femoral cartilage, 2 tibial cartilage, 3/4 menisci,
  5 unresolved cartilage/meniscus overlap. Medial/lateral mapping is not asserted.
- Python 3.10+ code syntax; exact deployment Python/package versions pending.
- Direct inference/postprocessing packages: torch, numpy, scipy, nibabel.
- DICOM conversion also requires MRIForge I/O support (`mriforge[io]`), including
  pydicom and SimpleITK; both installation commands above already include it.
- MONAI is optional, used only by the developer reference-parity test.
- Optional nvidia-ml-py for GPU-utilization reporting; a compatible NVIDIA driver
  and CUDA-enabled PyTorch installation for GPU inference. Missing/import-failed
  nvidia-ml-py is named explicitly in GPU status with its install command. When
  memory queries succeed, missing utilization support does not block selection.
  Driver/query failures retain their own diagnostic messages.
- Linux, Bash and standard Unix utilities; a browser for the UI. SSH tunnelling
  is useful for remote access. Native Windows deployment is not the target.

No Streamlit, Flask, Jupyter, pandas, cmpl or original internal_libs are needed.
Install dependencies using either command in the Installation section.
Exact dependency versions are not pinned; no requirements lock or Conda
environment file is included.

## Integrity and checks

From the release root on Linux:

```bash
sha256sum -c SHA256SUMS
```

The root checksum file covers the published code, documentation and manifests;
it excludes itself, trained weights, Git metadata and generated Python caches.
It can be checked before obtaining the weights. Each model directory has a
separate weight-only SHA256SUMS for checking the checkpoints after copying them. copy_provenance.json records the
source files and their initial copy hashes. release_manifest.json identifies
the snapshot and records intentional changes to the copied joint manifest and
release-only inference code. The initial-copy hashes remain historical; use
SHA256SUMS to verify the current release.

Tests use synthetic masks/tensors/DICOM headers and mocked conversion,
pipelines and GPUs, with no patient data or MRI inference. DICOM tests call a
mock MRIForge writer and check ordering, geometry preservation, failure and
cancellation behavior. Network tests load checkpoints without a forward pass and compare
randomly initialized networks on small synthetic tensors. The reference-parity
test is skipped when optional MONAI is absent; it is not a deployment dependency.
Restore all four checkpoint files before running the full Python test suite:

```bash
python -B -m unittest discover -s verified_meniscus_and_cartilage_models_2026-11-11/tests -p 'test_*.py' -v
```

Optional browser-logic tests (Node is needed only for these checks):

```bash
node verified_meniscus_and_cartilage_models_2026-11-11/tests/test_device_dropdown.js
node verified_meniscus_and_cartilage_models_2026-11-11/tests/test_model_selection.js
node verified_meniscus_and_cartilage_models_2026-11-11/tests/test_dicom_input.js
```
