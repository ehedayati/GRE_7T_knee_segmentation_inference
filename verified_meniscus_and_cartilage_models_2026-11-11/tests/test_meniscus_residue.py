"""Synthetic meniscus correction checks; no models or patient masks are used."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'correct_meniscus_residue.py'
spec=importlib.util.spec_from_file_location('residue',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ResidueTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((40,30,20),np.uint8)
        self.mask[2:5,5:12,5:12]=3
        self.mask[22:25,5:12,5:12]=4
        self.mask[21,6:9,6:9]=3  # wrong 3 touches trusted 4
        self.mask[5,6:9,6:9]=4   # wrong 4 touches trusted 3
        self.mask[35,25,15]=3   # far away: preserve
        self.mask[36,25,15]=4   # near a stray 3, NOT the trusted 3
        self.mask[1,20,1]=1;self.mask[2,20,1]=2;self.mask[3,20,1]=5

    def correct(self,mask=None,**kwargs):
        return m.correct_meniscus_residue(self.mask if mask is None else mask,
                                        kwargs.pop('spacing',(1,1,1)),
                                        kwargs.pop('radius_mm',1.1),**kwargs)

    def test_both_attached_residues(self):
        result,proposed,rows=self.correct()
        self.assertTrue((result[21,6:9,6:9]==4).all())
        self.assertTrue((result[5,6:9,6:9]==0).all())
        self.assertEqual(np.count_nonzero(result!=self.mask),18)
        self.assertEqual(np.count_nonzero(proposed),18)
        self.assertEqual({row['source_label'] for row in rows if row['selected']},{3,4})
        self.assertEqual(result[35,25,15],3);self.assertEqual(result[36,25,15],4)

    def test_nearby_disconnected_reverse_residue(self):
        mask=self.mask.copy();mask[5,6:9,6:9]=0;mask[6,6:9,6:9]=4
        small,_,_=self.correct(mask)
        self.assertTrue((small[6,6:9,6:9]==4).all())
        larger,_,_=self.correct(mask,radius_mm=2.1)
        self.assertTrue((larger[6,6:9,6:9]==3).all())

    def test_frozen_references_no_reversal(self):
        result,_,rows=self.correct()
        protected3=self.mask==3
        protected3[21,6:9,6:9]=False;protected3[35,25,15]=False
        protected4=self.mask==4
        protected4[5,6:9,6:9]=False;protected4[36,25,15]=False
        self.assertTrue((result[protected3]==3).all())
        self.assertTrue((result[protected4]==4).all())
        self.assertTrue((result[21,6:9,6:9]==4).all())
        self.assertTrue(all(not row['selected'] for row in rows if row['protected']))
        self.assertTrue(np.array_equal(result[self.mask<3],self.mask[self.mask<3]))
        self.assertTrue(np.array_equal(result[self.mask==5],self.mask[self.mask==5]))

    def test_erase_both_directions(self):
        result,_,_=self.correct(action='erase')
        self.assertTrue((result[21,6:9,6:9]==0).all())
        self.assertTrue((result[5,6:9,6:9]==0).all())

    def test_spacing_margin_and_fraction(self):
        result,_,_=self.correct(spacing=(2,1,1))
        self.assertTrue(np.array_equal(result,self.mask))
        result,_,_=self.correct(margin_mm=30,erase_label4_near_label3_mm=0)
        self.assertTrue(np.array_equal(result,self.mask))
        mask=self.mask.copy();mask[5:9,6,6]=4
        result,_,_=self.correct(mask,min_fraction=1.0,erase_label4_near_label3_mm=0)
        self.assertTrue((result[5:9,6,6]==4).all())

    def test_explicit_selections(self):
        result,_,_=self.correct(only_components=[],only_label4_components=[2])
        self.assertTrue((result[21,6:9,6:9]==3).all())
        self.assertTrue((result[5,6:9,6:9]==0).all())
        with self.assertRaises(ValueError):self.correct(only_label4_components=[1])
        with self.assertRaises(ValueError):self.correct(only_components=[3])

    def test_one_way_compatibility(self):
        result,_,rows=self.correct(bidirectional=False)
        self.assertTrue((result[21,6:9,6:9]==4).all())
        self.assertTrue((result[5,6:9,6:9]==4).all())
        self.assertEqual({row['source_label'] for row in rows},{3})
        with self.assertRaises(ValueError):self.correct(bidirectional=False,only_label4_components=[2])

    def test_same_label_bridge_to_trusted_component_is_protected(self):
        mask=self.mask.copy();mask[5:23,6,6]=4
        result,_,_=self.correct(mask)
        self.assertTrue((result[5:23,6,6]==4).all())

    def test_cli_compact_and_review_geometry(self):
        with tempfile.TemporaryDirectory(prefix='bidirectional_residue_') as temp:
            root=Path(temp);source=root/'input.nii.gz'
            affine=np.diag([1.,1.,1.,1.]);affine[:3,3]=[8,9,10]
            image=nib.Nifti1Image(self.mask,affine)
            image.header.set_xyzt_units('mm');image.set_qform(affine,1);image.set_sform(affine,2)
            nib.save(image,source);before=source.read_bytes()
            common=['correct_meniscus_residue.py','--input',str(source),'--radius-mm','1.1']
            output=root/'meniscus_only.nii.gz'
            with patch.object(sys,'argv',common+['--output',str(output),'--meniscus-only']),contextlib.redirect_stdout(io.StringIO()):
                m.main()
            out=nib.load(output);labels=np.asanyarray(out.dataobj)
            self.assertEqual(set(np.unique(labels)),{0,3,4})
            self.assertTrue((labels[21,6:9,6:9]==4).all())
            self.assertTrue((labels[5,6:9,6:9]==0).all())
            self.assertTrue(np.allclose(out.affine,affine))
            self.assertEqual(int(out.header['qform_code']),1);self.assertEqual(int(out.header['sform_code']),2)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual({p.name for p in root.iterdir()},{'input.nii.gz','meniscus_only.nii.gz'})
            review=root/'review'
            with patch.object(sys,'argv',common+['--output-dir',str(review)]),contextlib.redirect_stdout(io.StringIO()):
                m.main()
            metadata=json.loads((review/'settings.json').read_text())
            self.assertEqual({entry['source_label'] for entry in metadata['selected_components']},{3,4})
            self.assertEqual(metadata['changed_voxels'],18)
            for label in (3,4):self.assertTrue((review/f'proposed_label{label}_residue.nii.gz').exists())
            with patch.object(sys,'argv',common+['--output',str(output)]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                m.main()

    def test_erase_whole_large_component_even_when_fraction_rule_fails(self):
        mask=self.mask.copy()
        mask[5:12,6:9,6:9]=4  # only the first slice is within 1 mm; the rest is farther
        result,proposed,rows=self.correct(mask,min_fraction=1.0,margin_mm=30)
        self.assertTrue((result[5:12,6:9,6:9]==0).all())
        affected=[row for row in rows if row['source_label']==4 and row['near_label3_erasure_rule']]
        self.assertEqual(len(affected),1)
        self.assertLess(affected[0]['fraction_near_target'],.9)
        self.assertEqual(affected[0]['action'],'erase')
        self.assertEqual(np.count_nonzero(proposed[5:12,6:9,6:9]),63)

    def test_erasure_is_label4_only_and_keeps_trusted_meniscus(self):
        mask=self.mask.copy()
        mask[5,9:12,5:8]=1  # cartilage touching trusted label 3
        mask[5,9:12,8:11]=2
        result,_,rows=self.correct(mask)
        self.assertTrue(np.array_equal(result[mask==1],mask[mask==1]))
        self.assertTrue(np.array_equal(result[mask==2],mask[mask==2]))
        self.assertTrue((result[2:5,5:12,5:12]==3).all())
        self.assertTrue((result[22:25,5:12,5:12]==4).all())
        self.assertTrue(all(not row['near_label3_erasure_rule'] for row in rows if row['source_label']==3))

    def test_erasure_disabled_and_selection_respected(self):
        result,_,_=self.correct(erase_label4_near_label3_mm=0)
        self.assertTrue((result[5,6:9,6:9]==3).all())
        result,_,rows=self.correct(only_label4_components=[])
        self.assertTrue((result[5,6:9,6:9]==4).all())
        self.assertTrue(any(row['near_label3_erasure_rule'] and not row['selected'] for row in rows))

    def test_physical_erasure_threshold(self):
        # 2-mm X spacing means adjacent voxel centers are outside the 1-mm rule.
        result,_,_=self.correct(spacing=(2,1,1),radius_mm=.1)
        self.assertTrue(np.array_equal(result,self.mask))
        result,_,_=self.correct(spacing=(2,1,1),radius_mm=.1,erase_label4_near_label3_mm=2.0)
        self.assertTrue((result[5,6:9,6:9]==0).all())

    def embedded_case(self):
        mask=np.zeros((40,30,20),np.uint8)
        mask[2:15,2:17,2:17]=3
        mask[25:29,2:17,2:17]=4  # larger than the wrong-label cavity
        mask[5:11,6:12,6:12]=4
        mask[11:13,13:15,13:15]=0  # a pre-existing real hole: leave untouched
        mask[11,4,4]=1;mask[12,4,4]=2;mask[13,4,4]=5
        return mask

    def test_enclosed_deleted_label4_becomes_label3(self):
        mask=self.embedded_case()
        result,_,rows=self.correct(mask,min_fraction=1.0)
        self.assertTrue((result[5:11,6:12,6:12]==3).all())
        self.assertTrue((result[25:29,2:17,2:17]==4).all())
        self.assertTrue(np.array_equal(result[mask==0],mask[mask==0]))
        for label in (1,2,5):self.assertTrue(np.array_equal(result[mask==label],mask[mask==label]))
        affected=[row for row in rows if row['filled_label3_voxels']]
        self.assertEqual(len(affected),1)
        self.assertEqual(affected[0]['filled_label3_voxels'],216)
        self.assertEqual(affected[0]['relabelled_voxels'],216)
        self.assertEqual(affected[0]['erased_voxels'],0)
        self.assertEqual(affected[0]['action'],'relabel')
        self.assertLess(affected[0]['fraction_near_target'],1)

    def test_open_cavity_with_multiple_contacts_is_restored(self):
        mask=self.embedded_case()
        mask[2:6,8,8]=0  # channel to the exterior, including one original cavity voxel
        result,_,rows=self.correct(mask)
        self.assertTrue((result[(mask==4) & (np.indices(mask.shape)[0]<15)]==3).all())
        self.assertEqual(sum(row['filled_label3_voxels'] for row in rows),215)
        self.assertTrue(np.array_equal(result[mask==0],mask[mask==0]))

    def test_hole_repair_can_be_disabled(self):
        result,_,rows=self.correct(self.embedded_case(),fill_erased_label4_holes=False)
        self.assertTrue((result[5:11,6:12,6:12]==0).all())
        self.assertEqual(sum(row['filled_label3_voxels'] for row in rows),0)

    def test_unselected_cavity_is_not_repaired(self):
        mask=self.embedded_case()
        result,_,rows=self.correct(mask,only_label4_components=[])
        self.assertTrue(np.array_equal(result,mask))
        self.assertEqual(sum(row['filled_label3_voxels'] for row in rows),0)

    def test_whole_cluster_repair_reports_actual_counts(self):
        mask=np.zeros((20,20,20),np.uint8)
        mask[2:7,2:7,2:7]=3
        mask[12:17,2:7,2:7]=4
        mask[5,5,5]=4;mask[6,6,6]=4;mask[7,7,7]=4
        result,_,rows=self.correct(mask,radius_mm=2)
        self.assertEqual(result[5,5,5],3)
        self.assertEqual(result[6,6,6],3)
        self.assertEqual(result[7,7,7],3)
        row=next(row for row in rows if row['source_label']==4 and row['selected'])
        self.assertEqual(row['filled_label3_voxels'],3)
        self.assertEqual(row['relabelled_voxels'],3)
        self.assertEqual(row['erased_voxels'],0)
        self.assertEqual(row['action'],'relabel')

    def test_hole_repair_compact_cli_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix='hole_repair_cli_') as temp:
            root=Path(temp);source=root/'input.nii.gz'
            image=nib.Nifti1Image(self.embedded_case(),np.eye(4));image.header.set_xyzt_units('mm')
            nib.save(image,source)
            common=['correct_meniscus_residue.py','--input',str(source),'--radius-mm','1']
            with patch.object(sys,'argv',common+['--output',str(root/'men.nii.gz'),'--meniscus-only']),contextlib.redirect_stdout(io.StringIO()):m.main()
            data=np.asanyarray(nib.load(root/'men.nii.gz').dataobj)
            self.assertTrue((data[5:11,6:12,6:12]==3).all())
            self.assertEqual(set(np.unique(data)),{0,3,4})
            with patch.object(sys,'argv',common+['--output-dir',str(root/'review')]),contextlib.redirect_stdout(io.StringIO()):m.main()
            report=json.loads((root/'review/settings.json').read_text())
            self.assertEqual(report['label3_hole_repair_voxels'],216)
            with patch.object(sys,'argv',common+['--output',str(root/'old.nii.gz'),'--no-hole-repair']),contextlib.redirect_stdout(io.StringIO()):m.main()
            old=np.asanyarray(nib.load(root/'old.nii.gz').dataobj)
            self.assertTrue((old[5:11,6:12,6:12]==0).all())

    def two_side_case(self):
        mask=np.zeros((30,20,15),np.uint8)
        mask[2:6,2:12,2:12]=3
        mask[6:12,2:6,2:12]=3  # connected L-shaped label 3
        mask[20:25,2:12,2:12]=4
        mask[6:10,6:10,5:8]=4  # touches label 3 only from -X and -Y, open otherwise
        return mask

    def test_two_adjacent_sides_restore_entire_open_cluster(self):
        mask=self.two_side_case()
        result,_,rows=self.correct(mask,min_fraction=1.0,margin_mm=30)
        self.assertTrue((result[6:10,6:10,5:8]==3).all())
        row=next(row for row in rows if row['source_label']==4 and row['selected'])
        self.assertEqual(row['label3_contact_sides'],2)
        self.assertEqual(row['filled_label3_voxels'],48)
        self.assertLess(row['fraction_near_target'],1)
        self.assertTrue(np.array_equal(result[mask==0],mask[mask==0]))

    def test_one_side_contact_is_still_erased(self):
        mask=self.two_side_case();mask[6:10,6:10,5:8]=0;mask[6:10,8:12,5:8]=4
        result,_,rows=self.correct(mask,min_fraction=1.0,margin_mm=30)
        self.assertTrue((result[6:10,8:12,5:8]==0).all())
        row=next(row for row in rows if row['source_label']==4 and row['selected'])
        self.assertEqual(row['label3_contact_sides'],1)
        self.assertEqual(row['filled_label3_voxels'],0)

    def test_diagonal_contacts_do_not_count(self):
        mask=np.zeros((30,20,15),np.uint8)
        mask[2:6,2:6,2:12]=3
        mask[20:25,2:12,2:12]=4
        mask[6:9,6:9,5:8]=4
        result,_,rows=self.correct(mask,radius_mm=.1,erase_label4_near_label3_mm=1.5)
        self.assertTrue((result[6:9,6:9,5:8]==0).all())
        row=next(row for row in rows if row['source_label']==4 and row['selected'])
        self.assertEqual(row['label3_contact_sides'],0)

    def test_repair_does_not_trigger_more_repair(self):
        mask=self.two_side_case()
        # A second untrusted 4 cluster separated from the two-sided cluster by
        # a background voxel must not grow/relabel as a result of repairing it.
        mask[10,11,6]=4
        result,_,_=self.correct(mask)
        self.assertTrue((result[6:10,6:10,5:8]==3).all())
        self.assertEqual(result[10,11,6],4)
        self.assertEqual(result[10,10,6],0)

    def test_missing_trusted_tissue_and_invalid_parameters(self):
        mask=self.mask.copy();mask[mask==4]=0
        with self.assertRaises(ValueError):self.correct(mask)
        for kwargs in ({'radius_mm':0},{'min_fraction':0},{'margin_mm':-1},
                       {'protected_label3_components':[999]},{'reference_label4_components':[999]},
                       {'erase_label4_near_label3_mm':-1},{'erase_label4_near_label3_mm':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.correct(**kwargs)


if __name__=='__main__':unittest.main()
