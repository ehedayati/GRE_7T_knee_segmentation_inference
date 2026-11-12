#!/usr/bin/env python3
"""Replace joint-model cartilage with a cartilage-only prediction.

Retain meniscus labels 3/4 from the reviewed candidate mask. Geometry must match.
Requires NumPy and nibabel. Inputs are never modified or resampled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path, allowed):
    before = checksum(path)
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not all(data.shape):
        raise ValueError(f'Expected a nonempty 3-D label mask: {path}')
    if not np.isfinite(data).all() or not np.equal(data, np.rint(data)).all():
        raise ValueError(f'Mask contains noninteger or nonfinite values: {path}')
    unexpected = set(np.unique(data).tolist()) - set(allowed)
    if unexpected:
        raise ValueError(f'Unexpected labels {sorted(unexpected)} in {path}; expected {sorted(allowed)}')
    if checksum(path) != before:
        raise ValueError(f'Input changed while reading: {path}')
    return image, data.astype(np.uint8), before


def merge_masks(candidate, cartilage):
    """Return merged labels and cartilage IDs at overlap voxels (0 elsewhere).

    ALL original cartilage from candidate is discarded. Meniscus comes only
    from candidate; cartilage comes only from the separate prediction.
    Overlaps are flagged as label 5; neither tissue takes priority.
    """
    candidate = np.asarray(candidate)
    cartilage = np.asarray(cartilage)
    if candidate.shape != cartilage.shape or candidate.ndim != 3:
        raise ValueError('Both masks must have the same 3-D shape')
    if not np.isin(candidate, [0,1,2,3,4]).all() or not np.isin(cartilage, [0,1,2]).all():
        raise ValueError('Expected candidate labels 0..4 and cartilage-only labels 0..2')
    men = np.isin(candidate, (3,4))
    cart = np.isin(cartilage, (1,2))
    overlap = np.where(men & cart, cartilage, 0).astype(np.uint8)
    merged = np.zeros(candidate.shape, np.uint8)
    merged[cart] = cartilage[cart]
    merged[men] = candidate[men]
    merged[men & cart] = 5
    return merged, overlap


def save_labels(path, data, reference):
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_intent('label')
    header['cal_min'], header['cal_max'] = 0, int(data.max())
    image = reference.__class__(data, reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    image.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    image.header.set_slope_inter(1,0)
    nib.save(image, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meniscus-mask', type=Path, required=True, help='Corrected meniscus-only mask or joint mask with labels 3 and 4')
    parser.add_argument('--cartilage-mask', type=Path, required=True, help='Cartilage-only prediction with labels 1 and 2')
    parser.add_argument('--output', type=Path, required=True, help='NEW merged segmentation .nii or .nii.gz')
    parser.add_argument('--no-overlap-map', action='store_true', help='Do not save the separate overlap NIfTI; merged overlaps still use label 5')
    args = parser.parse_args()
    source = args.meniscus_mask.expanduser().resolve()
    cartilage_path = args.cartilage_mask.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not output.name.lower().endswith(('.nii','.nii.gz')):
        parser.error('Output must end in .nii or .nii.gz')
    stem = output.name[:-7] if output.name.lower().endswith('.nii.gz') else output.stem
    overlap_path = output.with_name(stem + '_overlap.nii.gz')
    report_path = output.with_name(stem + '_merge.json')
    for path in ((output, report_path) if args.no_overlap_map else (output, overlap_path, report_path)):
        if path in (source, cartilage_path) or path.exists() or path.is_symlink():
            raise FileExistsError(f'Will not replace existing file or input: {path}')
    reference, candidate, source_hash = load_labels(source, range(5))
    cart_image, cartilage, cart_hash = load_labels(cartilage_path, range(3))
    if candidate.shape != cartilage.shape or not np.allclose(reference.affine, cart_image.affine, atol=1e-4, rtol=0):
        raise ValueError('Shape/affine mismatch: both masks must represent the same scan and voxel grid; no automatic resampling')
    if reference.header.get_xyzt_units()[0] != cart_image.header.get_xyzt_units()[0]:
        raise ValueError('Spatial-unit mismatch between the masks; verify their geometry first')
    merged, overlap = merge_masks(candidate, cartilage)
    count = int(np.count_nonzero(overlap))
    report = {
        'meniscus_source': {'path':str(source), 'sha256':source_hash},
        'cartilage_source': {'path':str(cartilage_path), 'sha256':cart_hash},
        'overlap_action':'flag as label 5',
        'overlap_voxels':count,
        'overlap_by_cartilage_label':{str(k):int(np.count_nonzero(overlap==k)) for k in (1,2)},
        'overlap_by_meniscus_label':{str(k):int(np.count_nonzero((overlap>0)&(candidate==k))) for k in (3,4)},
        'original_joint_cartilage_voxels_discarded':int(np.count_nonzero(np.isin(candidate,(1,2)))),
        'output_label_voxels':{str(k):int(np.count_nonzero(merged==k)) for k in range(6)},
        'output':str(output), 'overlap_map':None if args.no_overlap_map else str(overlap_path),
        'note':'Overlap-map values 1/2 identify the cartilage labels competing with candidate meniscus; zero elsewhere.',
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    save_labels(output,merged,reference)
    if not args.no_overlap_map:
        save_labels(overlap_path,overlap,reference)
    report['output_sha256']=checksum(output)
    if not args.no_overlap_map:
        report['overlap_map_sha256']=checksum(overlap_path)
    with report_path.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(f'Saved merged mask: {output}')
    print(f'Overlap voxels flagged as label 5: {count}')
    if not args.no_overlap_map:
        print(f'Saved overlap map: {overlap_path}')
    print(f'Saved provenance report: {report_path}')
    print('Original joint cartilage was discarded. Check the merged mask and overlap map over your MRI.')


if __name__ == '__main__':
    main()
