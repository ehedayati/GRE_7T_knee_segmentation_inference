# Shared segmentation post-processing

One implementation supports cartilage labels 1,2 or joint labels 1,2,3,4.
Requires Python 3.10+, NumPy, SciPy and nibabel. No model or GPU is needed.

Interactive cartilage review:

```bash
bash postprocess_interactive.sh /path/mask.nii.gz
```

Interactive joint review:

```bash
POSTPROCESS_LABELS="1 2 3 4" bash postprocess_interactive.sh /path/mask.nii.gz
```

Outputs go into review/ beside the supplied mask. Existing pass numbers are
skipped. Each tissue gets a numbered map; IDs 1-6 are individual components,
7 groups the seventh-largest and all smaller components. The launcher waits
for visual review, asks what to REMOVE, and requires confirmation. Enter means
no removal; q exits. Original masks and unselected tissues are preserved.

Manual prepare: python postprocess_clusters.py prepare --input MASK --review-dir NEW_DIR --labels 1 2 3 4
Manual apply: python postprocess_clusters.py apply --review-dir DIR --output NEW_MASK --remove-label3 2,7

Use comma-separated display IDs independently via --remove-label1 through
--remove-label4. Omitted lists remove nothing. Preparing without --labels defaults
to cartilage only (1,2). Source/map hashes reject stale or changed review files.
