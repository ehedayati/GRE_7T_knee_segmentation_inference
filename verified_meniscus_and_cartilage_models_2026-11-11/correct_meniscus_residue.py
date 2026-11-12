#!/usr/bin/env python3
"""Two-pass correction of mixed meniscus labels: 3 to 4, then 4 to 3.

Standalone: Python 3.10+, NumPy, SciPy, nibabel. Never overwrites the input.
Component IDs are TRUE size ranks, not grouped review-map display IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def component_ids(text):
    try:
        values = sorted(set(int(x.strip()) for x in text.split(',')))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use positive comma-separated component ranks, e.g. 1,2') from error
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError('Component ranks must be positive')
    return values


def ranked_components(binary, connectivity):
    structure = ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    components, count = ndi.label(binary, structure=structure)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    ids = np.arange(1, count + 1)
    order = ids[np.lexsort((ids, -sizes[ids]))]
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[order] = np.arange(1, count + 1)
    return lookup[components], sizes[order]


def validate_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 3 or not all(mask.shape):
        raise ValueError('Expected a nonempty 3-D integer label mask')
    if not np.isfinite(mask).all() or not np.equal(mask, np.rint(mask)).all():
        raise ValueError('Mask must contain finite integer labels, not probabilities')
    if mask.min() < 0 or mask.max() > np.iinfo(np.int32).max:
        raise ValueError('Labels must be nonnegative int32 values')
    return mask.astype(np.int32)


def correct_meniscus_residue(
    mask, spacing, radius_mm, *, protected_label3_components=(1,),
    reference_label4_components=(1,), min_fraction=0.9, margin_mm=0.5,
    action='relabel', connectivity=26, only_components=None,
    bidirectional=True, only_label4_components=None, erase_label4_near_label3_mm=1.0,
    fill_erased_label4_holes=True,
):
    """Correct 3->4 first, then original label-4 residue near trusted label 3.

    Trusted components, ranks and distances are fixed from the ORIGINAL input.
    Both trusted menisci remain protected. A component qualifies when at least
    min_fraction of its voxels are within radius_mm of the opposite trusted
    meniscus and closer to it by more than margin_mm. Whole components change.
    Pass two considers only untrusted ORIGINAL label-4 components; it cannot
    undo pass one or use newly reassigned residue as a trusted reference.
    Additional rule in pass two: an untrusted ORIGINAL label-4 component with
    ANY voxel within erase_label4_near_label3_mm of trusted label 3 is erased
    WHOLE. This overrides relabelling/fraction/margin criteria for that component.
    Trusted label 4 stays protected. Set the erasure distance to 0 to disable.
    A deleted cluster touching corrected label 3 from at least TWO of the six
    face directions is restored WHOLE to label 3. The directions need not be
    opposite and may touch different voxels of the cluster. Diagonals do not
    count. Contacts are frozen before repair; there is no iterative growth.
    Only original label-4 voxels erased by this rule can be restored; original
    background/cartilage stay unchanged. Clusters with <2 contacts remain zero.
    Set fill_erased_label4_holes=False to compare deletion-only behavior.

    only_components restricts eligible label-3 ranks; only_label4_components
    restricts eligible label-4 ranks. None selects all eligible components.
    Set bidirectional=False for the original one-way behavior.

    Returns corrected labels, positive proposed component ranks, and report rows.
    The proposed map contains all eligible ranks, independent of selection.
    Its ranks are per source tissue, identified by the input mask/report; they
    are NOT grouped at 7. Non-meniscus labels and background never change.
    """
    mask = validate_mask(mask)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('spacing must contain three positive finite values in mask-axis order')
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError('radius_mm must be positive and finite')
    if not np.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError('margin_mm must be nonnegative and finite')
    if not np.isfinite(min_fraction) or not 0 < min_fraction <= 1:
        raise ValueError('min_fraction must be in (0,1]')
    if not np.isfinite(erase_label4_near_label3_mm) or erase_label4_near_label3_mm < 0:
        raise ValueError('erase_label4_near_label3_mm must be nonnegative and finite')
    if connectivity not in (6, 18, 26) or action not in ('relabel', 'erase'):
        raise ValueError('Invalid connectivity or action')
    protected = set(protected_label3_components)
    reference = set(reference_label4_components)
    for ids in (protected, reference):
        if not ids or any(not isinstance(x, (int, np.integer)) or x < 1 for x in ids):
            raise ValueError('At least one positive component rank is required for each trusted tissue')
    if not bidirectional and only_label4_components is not None:
        raise ValueError('Label-4 selections require bidirectional correction')

    bounds = ndi.find_objects(np.isin(mask, (3, 4)).astype(np.uint8))
    if not bounds:
        raise ValueError('Mask has no meniscus labels 3 or 4')
    roi = bounds[0]
    crop = mask[roi]
    cc3, sizes3 = ranked_components(crop == 3, connectivity)
    cc4, sizes4 = ranked_components(crop == 4, connectivity)
    if protected - set(range(1, len(sizes3) + 1)):
        raise ValueError(f'Protected label-3 ranks absent; available ranks: 1..{len(sizes3)}')
    if reference - set(range(1, len(sizes4) + 1)):
        raise ValueError(f'Reference label-4 ranks absent; available ranks: 1..{len(sizes4)}')
    distance3 = ndi.distance_transform_edt(~np.isin(cc3, list(protected)), sampling=spacing)
    distance4 = ndi.distance_transform_edt(~np.isin(cc4, list(reference)), sampling=spacing)
    corrected = mask.copy()
    corrected_crop = corrected[roi]
    proposed = np.zeros(mask.shape, dtype=np.int32)
    proposed_crop = proposed[roi]
    rows = []
    near_erased4 = np.zeros(crop.shape, dtype=bool)
    passes = [(3, 4, cc3, sizes3, protected, distance3, distance4, only_components)]
    if bidirectional:
        passes.append((4, 3, cc4, sizes4, reference, distance4, distance3, only_label4_components))
    for source, target, components, sizes, trusted, own_distance, target_distance, chosen in passes:
        near = target_distance <= radius_mm
        qualifies = near & (target_distance + margin_mm < own_distance)
        counts = np.bincount(components[qualifies], minlength=len(sizes) + 1)[1:]
        near_counts = np.bincount(components[near], minlength=len(sizes) + 1)[1:]
        ranks = np.arange(1, len(sizes) + 1)
        minimum_target = ndi.minimum(target_distance, components, index=ranks)
        minimum_own = ndi.minimum(own_distance, components, index=ranks)
        eligible = {int(rank) for rank, count, size in zip(ranks, counts, sizes)
                    if rank not in trusted and count / size >= min_fraction}
        erase_ranks = set()
        if source == 4 and erase_label4_near_label3_mm > 0:
            erase_ranks = {int(rank) for rank, distance in zip(ranks, minimum_target)
                           if rank not in trusted and distance <= erase_label4_near_label3_mm}
        eligible |= erase_ranks
        selected = eligible if chosen is None else set(chosen)
        if any(not isinstance(rank, (int, np.integer)) or rank < 1 for rank in selected):
            raise ValueError('Selected component ranks must be positive integers')
        if selected - eligible:
            raise ValueError(f'Label {source}: requested ranks are not eligible: '
                             f'{sorted(selected - eligible)}; eligible ranks: {sorted(eligible)}')
        # Only original source-label voxels are candidates in either pass.
        changed = np.isin(components, list(selected)) & (crop == source)
        corrected_crop[changed] = target if action == 'relabel' else 0
        erase_selected = selected & erase_ranks
        erase_mask = np.isin(components, list(erase_selected)) & (crop == source)
        corrected_crop[erase_mask] = 0
        if source == 4:
            near_erased4 |= erase_mask
        proposed_here = np.isin(components, list(eligible)) & (crop == source)
        proposed_crop[proposed_here] = components[proposed_here]
        for rank, size, n_near, n_good, d_target, d_own in zip(
            ranks, sizes, near_counts, counts, minimum_target, minimum_own
        ):
            rank = int(rank)
            rows.append({
                'source_label': source, 'target_label': target, 'component_rank': rank,
                'voxels': int(size), 'volume_mm3': float(size * np.prod(spacing)),
                'protected': rank in trusted,
                'minimum_distance_to_target_mm': float(d_target),
                'minimum_distance_to_own_trusted_meniscus_mm': float(d_own),
                'fraction_near_target': float(n_near / size),
                'fraction_near_and_closer_to_target': float(n_good / size),
                'eligible': rank in eligible, 'selected': rank in selected,
                'near_label3_erasure_rule': rank in erase_ranks,
                'action': ('erase' if rank in erase_selected else action) if rank in selected else 'unchanged',
                'filled_label3_voxels': 0,
                'label3_contact_sides': 0,
                'erased_voxels': int(size) if rank in selected and (rank in erase_selected or action=='erase') else 0,
                'relabelled_voxels': int(size) if rank in selected and rank not in erase_selected and action=='relabel' else 0,
            })
    if fill_erased_label4_holes and near_erased4.any():
        # Count distinct face-contact directions PER ORIGINAL CLUSTER, not per
        # voxel. Frozen label 3 prevents sequential filling from causing growth.
        label3 = np.pad(corrected_crop == 3, 1, constant_values=False)
        contact_sides = np.zeros(len(sizes4)+1, dtype=np.uint8)
        for axis in range(3):
            for step in (-1, 1):
                view = [slice(1, 1+n) for n in crop.shape]
                view[axis] = slice(1+step, 1+step+crop.shape[axis])
                neighbor3 = label3[tuple(view)]
                touching_ids = np.unique(cc4[near_erased4 & neighbor3])
                contact_sides[touching_ids] += 1
        restore_ids = np.flatnonzero(contact_sides >= 2)
        repaired = near_erased4 & np.isin(cc4, restore_ids) & (corrected_crop == 0)
        corrected_crop[repaired] = 3
        repaired_sizes = np.bincount(cc4[repaired], minlength=len(sizes4)+1)
        for row in rows:
            if row['source_label'] != 4:
                continue
            row['label3_contact_sides'] = int(contact_sides[row['component_rank']])
            count = int(repaired_sizes[row['component_rank']])
            if count:
                row['filled_label3_voxels'] = count
                row['erased_voxels'] -= count
                row['relabelled_voxels'] += count
                row['action'] = 'relabel' if row['erased_voxels']==0 else 'erase_and_fill_label3'
    return corrected, proposed, rows


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_mask(path, array, source):
    header = source.header.copy()
    header.set_data_dtype(array.dtype)
    header.set_intent('label')
    header['cal_min'], header['cal_max'] = 0, int(array.max())
    result = source.__class__(array, source.affine, header)
    result.set_qform(source.get_qform(), int(source.header['qform_code']))
    result.set_sform(source.get_sform(), int(source.header['sform_code']))
    result.header.set_slope_inter(1, 0)
    nib.save(result, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--radius-mm', required=True, type=float, help='Maximum reference distance; choose after visual review')
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument('--output-dir', type=Path, help='New review directory; default beside mask')
    outputs.add_argument('--output', type=Path, help='Write only the corrected mask; omit review files')
    parser.add_argument('--meniscus-only', action='store_true', help='With --output, retain only labels 3/4')
    parser.add_argument('--protect-label3', type=component_ids, default=[1], help='True ranks to protect; default 1')
    parser.add_argument('--reference-label4', type=component_ids, default=[1], help='Trusted label-4 ranks: reference for pass 1 and protected in pass 2; default 1')
    parser.add_argument('--min-fraction', type=float, default=0.9, help='Fraction of a component meeting BOTH distance rules; default 0.9')
    parser.add_argument('--margin-mm', type=float, default=0.5, help='Required advantage over distance to protected label 3; default 0.5')
    parser.add_argument('--action', choices=('relabel', 'erase'), default='relabel')
    parser.add_argument('--connectivity', choices=(6,18,26), type=int, default=26)
    parser.add_argument('--only-components', type=component_ids, help='Only treat these eligible label-3 ranks, e.g. 2,5')
    parser.add_argument('--only-label4-components', type=component_ids, help='Only treat these eligible label-4 ranks in the second pass')
    parser.add_argument('--one-way', action='store_true', help='Disable all second-pass label-4 treatment for comparison')
    parser.add_argument('--erase-label4-near-label3-mm', type=float, default=1.0,
                        help='Erase whole untrusted label-4 clusters with any voxel within this distance of trusted label 3; default 1.0 mm, 0 disables')
    parser.add_argument('--no-hole-repair', action='store_true',
                        help='Disable restoration of deleted label-4 clusters touching label 3 from at least two face directions')
    parser.add_argument('--spacing-mm', nargs=3, type=float, help='Explicit spacing in array-axis order when NIfTI units are unknown')
    args = parser.parse_args()
    source_path = args.input.expanduser().resolve()
    stem = source_path.name[:-7] if source_path.name.lower().endswith('.nii.gz') else source_path.stem
    if args.meniscus_only and args.output is None:
        parser.error('--meniscus-only requires --output')
    output = (args.output or args.output_dir or source_path.with_name(stem + '_meniscus_residue_review')).expanduser().resolve()
    if args.output is not None and not output.name.lower().endswith(('.nii', '.nii.gz')):
        parser.error('--output must end in .nii or .nii.gz')
    if output.exists() or output.is_symlink():
        parser.error(f'Output already exists; choose a new output path: {output}')
    before_hash = file_hash(source_path)
    image = nib.load(str(source_path))
    mask = validate_mask(np.asanyarray(image.dataobj))
    if file_hash(source_path) != before_hash:
        parser.error('Input changed while being read')
    axes = image.affine[:3, :3]
    lengths = np.linalg.norm(axes, axis=0)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        parser.error('Invalid spatial affine')
    unit_axes = axes / lengths
    if not np.allclose(unit_axes.T @ unit_axes, np.eye(3), atol=1e-4):
        parser.error('Sheared NIfTI geometry is unsupported by the spacing-based distance calculation')
    if args.spacing_mm is not None:
        spacing = args.spacing_mm
    else:
        unit = image.header.get_xyzt_units()[0]
        scale = {'mm':1., 'meter':1000., 'micron':.001}.get(unit)
        if scale is None:
            parser.error('Spatial units unknown: supply --spacing-mm X Y Z using actual voxel sizes')
        spacing = (lengths * scale).tolist()
    corrected, proposed, report = correct_meniscus_residue(
        mask, spacing, args.radius_mm,
        protected_label3_components=args.protect_label3,
        reference_label4_components=args.reference_label4,
        min_fraction=args.min_fraction, margin_mm=args.margin_mm,
        action=args.action, connectivity=args.connectivity, only_components=args.only_components,
        bidirectional=not args.one_way, only_label4_components=args.only_label4_components,
        erase_label4_near_label3_mm=args.erase_label4_near_label3_mm,
        fill_erased_label4_holes=not args.no_hole_repair,
    )
    for source_label in (3,4):
        selected_rows = [row for row in report if row['source_label']==source_label and row['selected']]
        erased = sum(row['erased_voxels'] for row in selected_rows)
        relabelled = sum(row['relabelled_voxels'] for row in selected_rows)
        print(f'Label {source_label}: {relabelled} residue voxels relabelled to {7-source_label}; {erased} erased')
    if args.output is not None:
        if args.meniscus_only:
            corrected = np.where(np.isin(corrected, (3,4)), corrected, 0).astype(np.uint8)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_mask(output, corrected, image)
        print(f'Saved corrected mask: {output}')
        print('No residue review files generated.')
        return
    output.mkdir(parents=True, exist_ok=False)
    save_mask(output/'candidate_mask.nii.gz', corrected, image)
    save_mask(output/'proposed_residue.nii.gz', proposed, image)
    for source_label in (3,4):
        save_mask(output/f'proposed_label{source_label}_residue.nii.gz',
                  np.where(mask==source_label, proposed, 0).astype(np.int32), image)
    save_mask(output/'changed_voxels.nii.gz', (corrected != mask).astype(np.uint8), image)
    # Full ranks are deliberately NOT capped at 7, unlike the older cluster review maps.
    for label in (3,4):
        components, _ = ranked_components(mask == label, args.connectivity)
        save_mask(output/f'label{label}_component_ranks.nii.gz', components, image)
    with (output/'components.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    metadata = {
        'source':str(source_path), 'source_sha256':before_hash,
        'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'effective_spacing_mm':list(spacing),
        'correction_mode':'one_way' if args.one_way else 'two_pass_fixed_references',
        'eligible_components':[{'source_label':row['source_label'], 'rank':row['component_rank']}
                               for row in report if row['eligible']],
        'selected_components':[{'source_label':row['source_label'], 'rank':row['component_rank']}
                               for row in report if row['selected']],
        'changed_voxels':int(np.count_nonzero(corrected != mask)),
        'label3_repair_rule':'whole_deleted_cluster_with_at_least_two_face_contacts',
        'label3_hole_repair_voxels':sum(row['filled_label3_voxels'] for row in report),
        'outputs_sha256':{path.name:file_hash(path) for path in output.glob('*.nii.gz')},
        'note':'Candidate for visual review, not an approved replacement. True component ranks; ID 7 is NOT a grouped remainder.',
    }
    with (output/'settings.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(f'Saved candidate and review files: {output}')
    print(f"Eligible components: {metadata['eligible_components']}")
    print(f"Selected components: {metadata['selected_components']}; changed voxels: {metadata['changed_voxels']}")
    print('Inspect proposed_residue, changed_voxels and candidate_mask over the MRI.')
    print('Trusted label-3/4 components, cartilage and other labels were preserved.')
    print('Component IDs here are full ranks, not the old grouped display IDs. Original input unchanged.')


if __name__ == '__main__':
    main()
