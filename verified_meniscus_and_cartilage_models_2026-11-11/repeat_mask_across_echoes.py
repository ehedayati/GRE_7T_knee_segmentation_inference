#!/usr/bin/env python3
"""Repeat a 3-D label mask over every echo of a matching 4-D MRI.

Copies labels unchanged; no registration or interpolation. Requires nibabel/numpy.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask', required=True, type=Path, help='3-D segmentation NIfTI')
    parser.add_argument('--reference', required=True, type=Path, help='4-D MRI with echoes last')
    parser.add_argument('--output', type=Path, help='Default: MASK_all_echoes.nii.gz')
    args = parser.parse_args()
    mask_path = args.mask.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    stem = mask_path.name[:-7] if mask_path.name.lower().endswith('.nii.gz') else mask_path.stem
    output = (args.output or mask_path.with_name(stem + '_all_echoes.nii.gz')).expanduser().resolve()
    if output.exists() or output.is_symlink() or output in (mask_path, reference_path):
        parser.error(f'Will not overwrite an input or existing output: {output}')
    if not output.name.lower().endswith(('.nii', '.nii.gz')):
        parser.error('Output must end in .nii or .nii.gz')

    mask = nib.load(str(mask_path))
    reference = nib.load(str(reference_path))  # Only the MRI header is needed.
    if len(mask.shape) != 3 or len(reference.shape) != 4 or reference.shape[3] < 1:
        parser.error('Expected a 3-D mask and a 4-D MRI with echoes on the last axis')
    if mask.shape != reference.shape[:3] or not np.allclose(mask.affine, reference.affine, atol=1e-4, rtol=0):
        parser.error('Mask and MRI must have matching spatial shape and affine')
    if mask.header.get_xyzt_units()[0] != reference.header.get_xyzt_units()[0]:
        parser.error('Mask and MRI spatial units differ; check their geometry')
    labels = np.asanyarray(mask.dataobj)
    if not np.isfinite(labels).all() or not np.equal(labels, np.rint(labels)).all():
        parser.error('Mask must contain finite integer labels')
    if labels.min() < 0 or labels.max() > np.iinfo(np.int32).max:
        parser.error('Labels must be nonnegative and fit in int32')
    dtype = np.uint8 if labels.max() <= 255 else (np.uint16 if labels.max() <= 65535 else np.int32)
    labels = labels.astype(dtype)
    repeated = np.broadcast_to(labels[..., None], reference.shape)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header.set_intent('label')
    header['cal_min'], header['cal_max'] = 0, int(labels.max())
    result = reference.__class__(repeated, reference.affine, header)
    result.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    result.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    result.header.set_slope_inter(1, 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output))
    print(f'Saved {reference.shape[3]} identical echo masks: {output}')
    print('Keep the original 3-D mask for the existing post-processing tools.')


if __name__ == '__main__':
    main()
