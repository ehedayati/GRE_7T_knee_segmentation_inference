"""Synthetic DICOM headers and mocked MRIForge conversion; no patient data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

path=Path(__file__).resolve().parents[1]/'convert_gre_dicom.py'
spec=importlib.util.spec_from_file_location('convert_gre_dicom',path)
converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dicom_conversion_test_')
        self.root=Path(self.temp.name)
        self.source=self.root/'GRE series';self.source.mkdir()
        self.output=self.root/'results/converted_input/gre.nii.gz'
        self.uid=generate_uid()
        self.header('first.dcm',self.uid)
        self.shape=(8,8,2,7)
        self.affine=np.diag([-0.5,0.7,1.2,1.0])
        self.affine[:3,3]=[10,20,30]
        self.data=np.broadcast_to(np.arange(7,dtype=np.float32),(8,8,2,7)).copy()
        self.write_sidecar=True
        self.call=Mock(side_effect=self.fake_convert)
        modules={name:ModuleType(name) for name in ('mrif','mrif.utilities','mrif.utilities.io')}
        modules['mrif.utilities.io'].dicom_to_nifti=self.call
        self.patch=patch.dict(sys.modules,modules);self.patch.start()

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def header(self,name,uid,modality='MR'):
        meta=FileMetaDataset()
        meta.MediaStorageSOPClassUID=MRImageStorage
        meta.MediaStorageSOPInstanceUID=generate_uid()
        meta.TransferSyntaxUID=ExplicitVRLittleEndian
        ds=FileDataset(str(self.source/name),{},file_meta=meta,preamble=bytes(128))
        ds.SOPClassUID=meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID=uid;ds.Modality=modality
        ds.save_as(str(self.source/name),enforce_file_format=True)

    def fake_convert(self,source,output,*,series_id,verbose):
        data=self.data if self.shape==self.data.shape else np.zeros(self.shape,np.float32)
        nib.save(nib.Nifti1Image(data,self.affine),output)
        if self.write_sidecar:
            Path(output).with_name('gre.json').write_text(json.dumps({'SeriesInstanceUID':series_id}))

    def test_calls_mriforge_and_preserves_echoes_and_geometry(self):
        self.header('second',self.uid)  # DICOM files need no .dcm extension.
        original=(self.source/'first.dcm').read_bytes()
        result=converter.convert(self.source,self.output,required_echoes=7,min_xy=8)
        self.assertEqual(result,self.output)
        self.call.assert_called_once_with(str(self.source),str(self.output),series_id=self.uid,verbose=True)
        image=nib.load(str(self.output))
        np.testing.assert_allclose(image.affine,self.affine)
        np.testing.assert_array_equal(np.asanyarray(image.dataobj),self.data)
        self.assertEqual((self.source/'first.dcm').read_bytes(),original)
        self.assertTrue(self.output.with_name('gre.json').is_file())

    def test_mixed_series_are_rejected_before_conversion(self):
        self.header('different.dcm',generate_uid())
        with self.assertRaisesRegex(ValueError,'Multiple DICOM series'):
            converter.convert(self.source,self.output,7,8)
        self.call.assert_not_called()
        self.assertFalse(self.output.parent.exists())

    def test_non_mr_is_rejected(self):
        self.header('first.dcm',self.uid,modality='CT')
        with self.assertRaisesRegex(ValueError,'MR GRE'):
            converter.convert(self.source,self.output,7,8)
        self.call.assert_not_called()

    def test_no_dicoms_and_nested_files_are_rejected(self):
        source=self.root/'empty';source.mkdir()
        (source/'nested').mkdir()
        with self.assertRaisesRegex(ValueError,'No MR DICOM series'):
            converter.convert(source,self.output,7,8)
        self.call.assert_not_called()

    def test_existing_nifti_or_sidecar_is_never_overwritten(self):
        self.output.parent.mkdir(parents=True)
        for path in (self.output,self.output.with_name('gre.json')):
            with self.subTest(path=path.name):
                path.write_text('preserve this')
                with self.assertRaises(FileExistsError):
                    converter.convert(self.source,self.output,7,8)
                self.assertEqual(path.read_text(),'preserve this')
                path.unlink()
        self.call.assert_not_called()

    def test_insufficient_echoes_or_spatial_size_are_rejected(self):
        for index,shape in enumerate(((8,8,2,3),(8,8,2),(4,8,2,7))):
            with self.subTest(shape=shape):
                self.shape=shape
                output=self.root/f'case{index}/gre.nii.gz'
                with self.assertRaisesRegex(ValueError,'No segmentation was started'):
                    converter.convert(self.source,output,7,8)

    def test_missing_sidecar_is_rejected(self):
        self.write_sidecar=False
        with self.assertRaisesRegex(RuntimeError,'both the requested NIfTI and JSON'):
            converter.convert(self.source,self.output,7,8)

    def test_missing_mriforge_has_install_instruction(self):
        with patch.dict(sys.modules,{'mrif.utilities.io':None}):
            with self.assertRaisesRegex(RuntimeError,'pip install "mriforge'):
                converter.convert(self.source,self.output,7,8)
        self.assertFalse(self.output.parent.exists())

    def test_invalid_parameters_are_rejected_before_conversion(self):
        for source,output,echoes,size in ((self.root/'missing',self.output,7,8),
                                           (self.source,self.root/'wrong.nii',7,8),
                                           (self.source,self.output,0,8)):
            with self.subTest(source=source,output=output,echoes=echoes),self.assertRaises(ValueError):
                converter.convert(source,output,echoes,size)
        self.call.assert_not_called()


if __name__=='__main__':unittest.main()
