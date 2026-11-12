#!/usr/bin/env python3
"""Review and remove numbered connected components from tissue labels 1 and 2.

prepare writes review maps and exits. apply removes only your selected IDs.
Display IDs 1-6 are individual components; 7 groups ALL remaining components.
Requires Python 3.10+, NumPy, SciPy and nibabel. No GPU or model is needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_mask(path):
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not all(data.shape):
        raise ValueError(f"Expected a nonempty 3-D label mask; got {data.shape}")
    if not np.isfinite(data).all() or not np.equal(data, np.rint(data)).all():
        raise ValueError("Mask must contain finite integer labels, not probabilities")
    if data.min() < 0 or data.max() > np.iinfo(np.int32).max:
        raise ValueError("Mask labels must be between 0 and 2147483647")
    return image, data.astype(np.int32)


def save_mask(path, data, reference):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    header.set_intent("label")
    header["cal_min"] = 0
    header["cal_max"] = int(data.max())
    image = reference.__class__(data, reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    image.header.set_slope_inter(1, 0)
    nib.save(image, str(path))


def rank_components(mask, label, connectivity):
    """Rank by size, breaking ties by scipy's original component ID."""
    structure = ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    components, count = ndi.label(mask == label, structure=structure)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    ids = np.arange(1, count + 1)
    ranked_ids = ids[np.lexsort((ids, -sizes[ids]))]
    lookup = np.zeros(count + 1, dtype=np.uint8)
    lookup[ranked_ids] = np.minimum(np.arange(1, count + 1), 7)
    return lookup[components], sizes[ranked_ids]


def prepare(args):
    source = args.input.expanduser().resolve()
    destination = args.review_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new review directory: {destination}")
    source_hash = sha256(source)
    image, mask = load_mask(source)
    if sha256(source) != source_hash:
        raise ValueError("Source mask changed while being read; retry preparation")
    units = image.header.get_xyzt_units()[0]
    scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(units)
    native_volume = abs(float(np.linalg.det(image.affine[:3, :3])))
    voxel_mm3 = native_volume * scale ** 3 if scale is not None else None
    destination.mkdir(parents=True, exist_ok=False)
    review = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": source_hash},
        "connectivity": args.connectivity,
        "display_rule": "1-6 are individual size ranks; 7 is all ranks >= 7",
        "spatial_units": units,
        "voxel_volume_mm3": voxel_mm3,
        "labels": {},
    }
    rows = []
    for label in (1, 2):
        numbered, sizes = rank_components(mask, label, args.connectivity)
        name = f"label{label}_clusters.nii.gz"
        path = destination / name
        save_mask(path, numbered, image)
        total = int(sizes.sum())
        for rank, size in enumerate(sizes, start=1):
            rows.append({
                "tissue_label": label, "component_rank": rank,
                "display_id": min(rank, 7), "voxels": int(size),
                "volume_mm3": float(size * voxel_mm3) if voxel_mm3 is not None else "",
                "percent_of_tissue": float(100 * size / total),
            })
        review["labels"][str(label)] = {
            "map": name, "sha256": sha256(path),
            "component_count": len(sizes), "total_voxels": total,
            "display_ids": list(range(1, min(len(sizes), 7) + 1)),
        }
        print(f"Tissue {label}: {len(sizes)} components, {total} voxels -> {path}")
    with (destination / "components.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "tissue_label", "component_rank", "display_id", "voxels",
            "volume_mm3", "percent_of_tissue",
        ])
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "review.json", review)
    if voxel_mm3 is None:
        print("NIfTI spatial units are unknown; mm3 fields are blank (voxel counts are valid).")
    print("\nOpen both cluster maps over your MRI and choose IDs to REMOVE for each tissue.")
    print("ID 7 removes the entire group of ranks 7 and below. No mask was trimmed yet.")
    print("Next: use 'apply --review-dir ... --output ... --remove-label1 ... --remove-label2 ...'")
    print("Omit a removal option to leave that tissue unchanged. Keep review files unedited.")


def parse_ids(value):
    if not value.strip() or value.lower() == "none":
        return []
    try:
        ids = sorted(set(int(part.strip()) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated IDs, e.g. 2,5,7, or none") from error
    if any(number < 1 or number > 7 for number in ids):
        raise argparse.ArgumentTypeError("Removal IDs must be between 1 and 7")
    return ids


def apply(args):
    folder = args.review_dir.expanduser().resolve()
    with (folder / "review.json").open(encoding="utf-8") as stream:
        review = json.load(stream)
    if review.get("schema_version") != 1:
        raise ValueError("Unsupported review schema")
    source = (args.input or Path(review["source"]["path"])).expanduser().resolve()
    if sha256(source) != review["source"]["sha256"]:
        raise ValueError("Source mask differs from the reviewed mask. Prepare a new review.")
    image, mask = load_mask(source)
    if sha256(source) != review["source"]["sha256"]:
        raise ValueError("Source mask changed while being read")
    output = args.output.expanduser().resolve()
    if not output.name.lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Output must end in .nii or .nii.gz")
    record_path = output.with_name(output.name + ".removals.json")
    for path in (output, record_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Will not overwrite: {path}")
    cleaned = mask.copy()
    selections = {"1": args.remove_label1, "2": args.remove_label2}
    changes = {}
    for key, ids in selections.items():
        entry = review["labels"][key]
        map_path = (folder / entry["map"]).resolve()
        if map_path.parent != folder:
            raise ValueError("Review maps must be located inside the review directory")
        if sha256(map_path) != entry["sha256"]:
            raise ValueError(f"Review map changed: {map_path}. Prepare a new review.")
        map_image, numbered = load_mask(map_path)
        if sha256(map_path) != entry["sha256"]:
            raise ValueError(f"Review map changed while being read: {map_path}")
        if numbered.shape != mask.shape or not np.allclose(
            map_image.affine, image.affine, atol=1e-5, rtol=0
        ):
            raise ValueError(f"Review map geometry does not match source: {map_path}")
        if numbered.max() > 7 or not np.array_equal(numbered > 0, mask == int(key)):
            raise ValueError(f"Review map does not match tissue {key}")
        present = set(np.unique(numbered).tolist()) - {0}
        if set(ids) - present:
            raise ValueError(f"Tissue {key}: requested IDs {ids}; available IDs are {sorted(present)}")
        removed = np.isin(numbered, ids)
        cleaned[removed] = 0
        changes[key] = {
            "removed_display_ids": ids,
            "removed_voxels": int(removed.sum()),
            "remaining_voxels": int(np.count_nonzero(cleaned == int(key))),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve label values, choosing an integer storage type that fits all labels.
    dtype = np.uint8 if mask.max() <= 255 else (np.uint16 if mask.max() <= 65535 else np.int32)
    save_mask(output, cleaned.astype(dtype), image)
    write_json(record_path, {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": review["source"]["sha256"]},
        "review_path": str(folder / "review.json"),
        "review_sha256": sha256(folder / "review.json"),
        "output": {"path": str(output), "sha256": sha256(output)},
        "changes": changes,
    })
    print(f"Saved cleaned mask: {output}")
    for key, change in changes.items():
        print(f"Tissue {key}: removed IDs {change['removed_display_ids']}, "
              f"{change['removed_voxels']} voxels; {change['remaining_voxels']} remain")
    print(f"Saved removal record: {record_path}")
    print("For another pass, prepare a NEW review directory using this cleaned mask.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare", help="Create ranked maps for visual review; do not trim")
    prep.add_argument("--input", required=True, type=Path, help="Original 3-D segmentation NIfTI")
    prep.add_argument("--review-dir", required=True, type=Path, help="New directory for this review pass")
    prep.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=26)
    prep.set_defaults(func=prepare)
    trim = subparsers.add_parser("apply", help="Remove selected display IDs and save a NEW mask")
    trim.add_argument("--review-dir", required=True, type=Path)
    trim.add_argument("--output", required=True, type=Path)
    trim.add_argument("--input", type=Path, help="Optional relocated source; must match the recorded hash")
    trim.add_argument("--remove-label1", type=parse_ids, default=[], help="IDs to REMOVE, e.g. 2,5,7; default none")
    trim.add_argument("--remove-label2", type=parse_ids, default=[], help="IDs to REMOVE, e.g. 3,7; default none")
    trim.set_defaults(func=apply)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
