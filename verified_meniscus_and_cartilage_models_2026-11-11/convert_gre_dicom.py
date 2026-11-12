#!/usr/bin/env python3
"""Convert one GRE DICOM series with MRIForge, preserving the NIfTI and sidecar.

No reorientation, resampling or echo reordering is applied after MRIForge.
Requires mriforge[io]. Refuses to overwrite outputs or silently select one of
multiple DICOM series. Checks the converted image before segmentation starts.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def find_series(source, pydicom):
    series = set()
    for path in source.iterdir():
        if not path.is_file() or path.name.startswith('.'):
            continue
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True,
                                     specific_tags=['SeriesInstanceUID', 'SOPClassUID', 'Modality'],
                                     force=True)
        except Exception:
            # Ignore unreadable files and non-DICOM sidecars while discovering series.
            continue
        uid = getattr(header, 'SeriesInstanceUID', None)
        if not uid or not getattr(header, 'SOPClassUID', None):
            continue
        if getattr(header, 'Modality', '') != 'MR':
            raise ValueError('Select an MR GRE DICOM series folder, without other modalities.')
        series.add(str(uid))
        if len(series) > 1:
            raise ValueError('Multiple DICOM series found. Select a folder containing only the GRE magnitude series and all its echoes.')
    if not series:
        raise ValueError('No MR DICOM series found directly in this folder. Select the folder containing the DICOM files.')
    return next(iter(series))


def convert(source: Path, output: Path, required_echoes=7, min_xy=512):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if not source.is_dir():
        raise ValueError('DICOM input must be an existing series directory')
    if not output.name.lower().endswith('.nii.gz'):
        raise ValueError('Converted output must end in .nii.gz')
    if required_echoes < 1 or min_xy < 1:
        raise ValueError('Required echo count and minimum image size must be positive')
    sidecar = output.with_name(output.name[:-7] + '.json')
    for path in (output, sidecar):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Conversion output already exists: {path}')
    try:
        import pydicom
        import nibabel as nib
        from mrif.utilities.io import dicom_to_nifti
    except ImportError as error:
        raise RuntimeError('MRIForge DICOM support is unavailable. Install it in the UI environment: '
                           'python -m pip install "mriforge[io,torch]". Details: ' + str(error)) from error
    series_id = find_series(source, pydicom)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'MRIForge DICOM source: {source}', flush=True)
    print(f'SeriesInstanceUID: {series_id}', flush=True)
    dicom_to_nifti(str(source), str(output), series_id=series_id, verbose=True)
    if not output.is_file() or not sidecar.is_file():
        raise RuntimeError('MRIForge did not produce both the requested NIfTI and JSON sidecar')
    image = nib.load(str(output))
    if len(image.shape) != 4 or image.shape[3] < required_echoes:
        raise ValueError(f'Converted GRE must be 4-D with at least {required_echoes} echoes in the last dimension; got {image.shape}. No segmentation was started.')
    if min(image.shape[:2]) < min_xy or image.shape[2] < 1:
        raise ValueError(f'Converted GRE needs X/Y >= {min_xy} and nonzero depth; got {image.shape}. No segmentation was started.')
    print(f'Converted MRI: {output} | shape={image.shape}', flush=True)
    print(f'MRIForge metadata: {sidecar}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='One GRE magnitude DICOM series folder')
    parser.add_argument('--output', required=True, type=Path, help='New .nii.gz path; matching .json is also saved')
    parser.add_argument('--required-echoes', type=int, default=7)
    parser.add_argument('--min-xy', type=int, default=512)
    args = parser.parse_args()
    try:
        convert(args.input, args.output, args.required_echoes, args.min_xy)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'DICOM conversion failed: {error}\n')


if __name__ == '__main__':
    main()
