"""UI checks using a fake pipeline; never imports or runs a segmentation model."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODULE_PATH = Path(__file__).resolve().parents[1]/'segmentation_ui.py'
spec = importlib.util.spec_from_file_location('segmentation_ui',MODULE_PATH)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='segmentation_ui_test_')
        self.root = Path(self.temp.name)
        self.source = self.root/'scan with spaces.nii.gz'
        self.source.write_text('mock input, no image voxels')
        self.pipeline = self.root/'fake_pipeline.py'
        self.pipeline.write_text('''import json,os,sys,time
from pathlib import Path
source,out,device,radius=sys.argv[1:]
print('[1/'+('5' if os.environ['ALL_ECHOES']=='1' else '4')+'] Mock cartilage stage',flush=True)
if 'slow' in source:
    time.sleep(30)
else:
    time.sleep(.2)
if 'fail' in source:
    print('Synthetic pipeline error',flush=True)
    sys.exit(3)
out=Path(out);out.mkdir(parents=True,exist_ok=True)
names=['cartilage_only.nii.gz','meniscus_only.nii.gz','merged_cartilage_meniscus.nii.gz','merged_cartilage_meniscus_merge.json']
if os.environ['ALL_ECHOES']=='1':names.append('merged_cartilage_meniscus_all_echoes.nii.gz')
for name in names:(out/name).write_text('mock output')
(out/'forwarded.json').write_text(json.dumps({'device':device,'radius':radius,'python':os.environ['PYTHON_BIN'],'echoes':os.environ['ALL_ECHOES']}))
(out/'forwarded_input.txt').write_text(source)
(out/'forwarded_models.json').write_text(json.dumps({'cartilage':os.environ['CARTILAGE_MODEL_PATH'],'meniscus':os.environ['MENISCUS_MODEL_PATH']}))
print('Finished fake run',flush=True)
''',encoding='utf-8')
        self.models_temp = tempfile.TemporaryDirectory(prefix='ui_models_test_')
        model_root=Path(self.models_temp.name)
        self.cart_dir=model_root/'cart';self.men_dir=model_root/'men'
        self.cart_dir.mkdir();self.men_dir.mkdir()
        self.write_models(self.cart_dir,3,['top3_unet2_4echo','top1_unet_4echo'])
        self.write_models(self.men_dir,5,['joint_unet_4echo_five_losses','joint_unet_4echo_three_losses'])
        self.panel = ui.Panel(self.root,self.pipeline,bash=sys.executable,
                              cartilage_dir=self.cart_dir,meniscus_dir=self.men_dir)

    def tearDown(self):
        self.panel.stop()
        if self.panel.thread:
            self.panel.thread.join(timeout=5)
        self.temp.cleanup()
        self.models_temp.cleanup()

    def write_models(self,folder,classes,names):
        entries=[]
        for name in names:
            (folder/(name+'.pt')).write_text('mock weights, never loaded')
            entries.append({'name':name,'checkpoint':name+'.pt','sha256':'a'*64,
                            'inference':{'Network':'UNet2' if 'unet2' in name else 'UNet',
                                         'out_channel':classes,'num_echos':4,'specific_echos':[0,2,4,6],
                                         'Unet_res_no':3,'normalize':True,'z_score':True,
                                         'crop_size':256,'window_size':32}})
        (folder/'manifest.json').write_text(json.dumps({'schema_version':1,'models':entries}))

    def payload(self,**extra):
        return {'input':str(self.source),'output':str(self.root/'output with spaces'),
                'device':'cpu','radius':1.5,'all_echoes':True,**extra}

    def wait(self):
        self.panel.thread.join(timeout=8)
        self.assertFalse(self.panel.thread.is_alive())
        return self.panel.status()

    def dicom_setup(self, name='GRE series'):
        source=self.root/name;source.mkdir()
        (source/'image.dcm').write_text('mock DICOM, never read')
        converter=self.root/'fake_converter.py'
        converter.write_text("""import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--input');p.add_argument('--output')
p.add_argument('--required-echoes');p.add_argument('--min-xy')
a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
(out.parent/'conversion_args.json').write_text(json.dumps(vars(a)))
print('Mock MRIForge conversion',flush=True)
if 'slow' in a.input:time.sleep(30)
if 'fail' in a.input:raise SystemExit(5)
out.write_text('mock converted MRI')
if 'missing' not in a.input:out.with_name('gre.json').write_text('{}')
""",encoding='utf-8')
        self.panel.converter=converter
        return source

    def test_dicom_conversion_precedes_inference(self):
        source=self.dicom_setup()
        self.panel.start(self.payload(input=str(source),input_type='dicom'))
        status=self.wait()
        out=self.root/'output with spaces'
        self.assertEqual(status['state'],'completed')
        self.assertEqual(status['total'],6)
        self.assertEqual(len(status['outputs']),7)
        converted=str(out/'converted_input/gre.nii.gz')
        self.assertEqual((out/'forwarded_input.txt').read_text(),converted)
        self.assertEqual(status['settings']['input'],str(source))
        self.assertEqual(status['settings']['inference_input'],converted)
        args=json.loads((out/'converted_input/conversion_args.json').read_text())
        self.assertEqual(args['required_echoes'],'7')
        self.assertEqual(args['min_xy'],'512')
        self.assertTrue(any('[1/6] Convert GRE DICOM' in line for line in status['logs']))
        self.assertTrue(any('[2/6] Mock cartilage' in line for line in status['logs']))

    def test_dicom_without_echo_output(self):
        source=self.dicom_setup()
        self.panel.start(self.payload(input=str(source),input_type='dicom',all_echoes=False))
        status=self.wait()
        self.assertEqual(status['state'],'completed')
        self.assertEqual(status['total'],5)
        self.assertEqual(len(status['outputs']),6)
        self.assertFalse(any('all_echoes.nii.gz' in name for name in status['outputs']))

    def test_dicom_conversion_failure_never_starts_inference(self):
        source=self.dicom_setup('fail GRE')
        self.panel.start(self.payload(input=str(source),input_type='dicom'))
        status=self.wait()
        self.assertEqual(status['state'],'failed')
        self.assertEqual(status['exit_code'],5)
        self.assertIn('Segmentation was not started',status['message'])
        self.assertFalse((self.root/'output with spaces/forwarded_input.txt').exists())

    def test_dicom_missing_sidecar_never_starts_inference(self):
        source=self.dicom_setup('missing sidecar GRE')
        self.panel.start(self.payload(input=str(source),input_type='dicom'))
        status=self.wait()
        self.assertEqual(status['state'],'failed')
        self.assertIn('JSON sidecar',status['message'])
        self.assertFalse((self.root/'output with spaces/forwarded_input.txt').exists())
        self.assertNotEqual(status['exit_code'],0)

    def test_pipeline_launch_error_after_conversion_is_failure(self):
        source=self.dicom_setup()
        def execute(command,env,stage_offset=0):
            if stage_offset is None:
                folder=self.root/'output with spaces/converted_input'
                folder.mkdir(parents=True)
                (folder/'gre.nii.gz').write_text('mock')
                (folder/'gre.json').write_text('{}')
                return 0
            raise OSError('Mock pipeline launch failure')
        with patch.object(self.panel,'_execute',side_effect=execute):
            self.panel.start(self.payload(input=str(source),input_type='dicom'))
            status=self.wait()
        self.assertEqual(status['state'],'failed')
        self.assertIn('launch failure',status['message'])
        self.assertNotEqual(status['exit_code'],0)

    def test_stop_during_dicom_conversion(self):
        source=self.dicom_setup('slow GRE')
        self.panel.start(self.payload(input=str(source),input_type='dicom'))
        started=self.root/'output with spaces/converted_input/conversion_args.json'
        deadline=time.monotonic()+4
        while not started.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertTrue(started.exists())
        self.panel.stop()
        status=self.wait()
        self.assertEqual(status['state'],'cancelled')
        self.assertFalse((self.root/'output with spaces/forwarded_input.txt').exists())

    def test_dicom_validation(self):
        source=self.dicom_setup()
        for extra in ({'input_type':'unknown'}, {'input_type':'dicom'},
                      {'input_type':'nifti','input':str(source)}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                self.panel.start(self.payload(**extra))
        out=self.root/'output with spaces/converted_input';out.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'Converted input directory already exists'):
            self.panel.start(self.payload(input=str(source),input_type='dicom'))

    def test_default_weight_directory_layout(self):
        application=MODULE_PATH.parents[1]
        panel=ui.Panel(self.root,self.pipeline,bash=sys.executable)
        expected={'cartilage':application/'weights/cartilage',
                  'meniscus':application/'weights/joint'}
        self.assertEqual(panel.model_dirs,{role:folder.resolve() for role,folder in expected.items()})
        for role,folder in expected.items():
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(any(entry['checkpoint']==panel.default_models[role].name
                                for entry in manifest['models']))
            for entry in manifest['models']:
                classes=3 if role=='cartilage' else 5
                self.assertEqual(entry['inference']['out_channel'],classes)
            self.assertTrue((folder/'SHA256SUMS').is_file())

    def test_browser(self):
        (self.root/'folder').mkdir()
        (self.root/'note.txt').write_text('not an MRI')
        files=self.panel.browse(str(self.root),'file')['entries']
        self.assertEqual([e['name'] for e in files],['folder','scan with spaces.nii.gz'])
        folders=self.panel.browse(str(self.root),'directory')['entries']
        self.assertEqual([e['name'] for e in folders],['folder'])
        with self.assertRaises(ValueError):self.panel.browse(str(self.source),'file')

    def test_success_forwarding_and_existing_outputs(self):
        self.panel.start(self.payload())
        with self.assertRaises(ValueError):self.panel.start(self.payload())
        status=self.wait()
        self.assertEqual(status['state'],'completed')
        self.assertEqual(len(status['outputs']),5)
        self.assertTrue(any('Mock cartilage' in line for line in status['logs']))
        forwarded=json.loads((self.root/'output with spaces/forwarded.json').read_text())
        self.assertEqual(forwarded,{'device':'cpu','radius':'1.5000000000','python':sys.executable,'echoes':'1'})
        with self.assertRaises(ValueError):self.panel.start(self.payload())

    def test_no_echoes(self):
        self.panel.start(self.payload(all_echoes=False))
        status=self.wait()
        self.assertEqual(status['total'],4)
        self.assertEqual(len(status['outputs']),4)

    def test_failure(self):
        source=self.root/'fail.nii.gz';source.write_text('mock')
        self.panel.start(self.payload(input=str(source)))
        status=self.wait()
        self.assertEqual(status['state'],'failed')
        self.assertEqual(status['exit_code'],3)
        self.assertIn('Synthetic pipeline error',status['logs'])

    def test_stop(self):
        source=self.root/'slow.nii.gz';source.write_text('mock')
        self.panel.start(self.payload(input=str(source)))
        self.panel.stop()
        status=self.wait()
        self.assertEqual(status['state'],'cancelled')

    def test_validation(self):
        for extra in ({'radius':'nan'},{'radius':0},{'device':'cpu;echo no'},
                      {'all_echoes':'false'},{'input':''},{'output':''}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                self.panel.start(self.payload(**extra))
        self.assertEqual(self.panel.status()['state'],'idle')

    def test_parent_and_new_folder(self):
        payload=self.payload(output_parent=str(self.root),output_name='FU_03')
        payload.pop('output')
        self.panel.start(payload)
        status=self.wait()
        self.assertEqual(status['state'],'completed')
        self.assertEqual(status['output_dir'],str(self.root/'FU_03'))
        self.assertEqual(status['settings']['output_parent'],str(self.root))
        self.assertEqual(status['settings']['output_name'],'FU_03')
        self.assertTrue((self.root/'FU_03/cartilage_only.nii.gz').is_file())
        with self.assertRaises(ValueError):self.panel.start(payload)

    def test_output_name_validation(self):
        for name in ('','.', '..', '../FU_03', 'a/b', 'a\\b', '/FU_03', 'bad\nname'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.panel.start(self.payload(output_parent=str(self.root),output_name=name))
        with self.assertRaises(ValueError):
            self.panel.start(self.payload(output_parent=str(self.root/'missing'),output_name='FU_03'))
        self.assertFalse((self.root/'FU_03').exists())
        self.assertEqual(self.panel.status()['state'],'idle')

    def gpu_snapshot(self):
        return {'checked_at':'test', 'cuda_visible_devices':'5,2', 'cuda_available':True,
                'error':None, 'devices':[
                    {'device':'cuda:0','name':'Visible GPU A','free_bytes':2*1024**3,
                     'total_bytes':24*1024**3,'utilization_percent':50,'error':None},
                    {'device':'cuda:1','name':'Visible GPU B','free_bytes':18*1024**3,
                     'total_bytes':24*1024**3,'utilization_percent':10,'error':None}]}

    def test_missing_optional_nvml_does_not_block_auto_gpu(self):
        snapshot=self.gpu_snapshot()
        snapshot.update(warnings=['GPU utilization unavailable: nvidia-ml-py is not available'],
                        missing_packages=['nvidia-ml-py'])
        for device in snapshot['devices']:
            device.update(utilization_percent=None,utilization_error='nvidia-ml-py is not available')
        with patch.object(self.panel,'gpu_availability',return_value=snapshot):
            device,message=self.panel.resolve_device('auto')
        self.assertEqual(device,'cuda:1')
        self.assertIn('Selected cuda:1',message)

    def test_auto_chooses_most_free_visible_gpu(self):
        with patch.object(self.panel,'gpu_availability',return_value=self.gpu_snapshot()) as probe:
            self.panel.start(self.payload(device='auto'))
            status=self.wait()
            probe.assert_called_once_with(fresh=True)
        self.assertEqual(status['state'],'completed')
        self.assertEqual(status['settings']['device'],'auto')
        self.assertEqual(status['settings']['resolved_device'],'cuda:1')
        forwarded=json.loads((self.root/'output with spaces/forwarded.json').read_text())
        self.assertEqual(forwarded['device'],'cuda:1')
        self.assertTrue(any('18.0 GiB free' in line for line in status['logs']))

    def test_manual_and_hidden_gpu_selection(self):
        with patch.object(self.panel,'gpu_availability',return_value=self.gpu_snapshot()):
            self.assertEqual(self.panel.resolve_device('cuda:0')[0],'cuda:0')
            self.assertEqual(self.panel.resolve_device('cuda')[0],'cuda:0')
            with self.assertRaises(ValueError):self.panel.resolve_device('cuda:5')
        with patch.object(self.panel,'gpu_availability') as probe:
            self.assertEqual(self.panel.resolve_device('cpu')[0],'cpu')
            probe.assert_not_called()

    def test_no_cuda_auto_uses_cpu(self):
        snapshot=self.gpu_snapshot();snapshot.update(cuda_available=False,devices=[])
        with patch.object(self.panel,'gpu_availability',return_value=snapshot):
            self.assertEqual(self.panel.resolve_device('auto')[0],'cpu')
            with self.assertRaises(ValueError):self.panel.resolve_device('cuda:0')

    def test_probe_failure_does_not_start_pipeline(self):
        snapshot=self.gpu_snapshot();snapshot['error']='driver query failed'
        with patch.object(self.panel,'gpu_availability',return_value=snapshot):
            self.panel.start(self.payload(device='auto'))
            status=self.wait()
        self.assertEqual(status['state'],'failed')
        self.assertIn('driver query failed',status['message'])
        self.assertFalse((self.root/'output with spaces').exists())

    def test_unreadable_gpu_skipped_and_ties_stable(self):
        snapshot=self.gpu_snapshot()
        snapshot['devices'][1].update(free_bytes=None,error='permission denied')
        with patch.object(self.panel,'gpu_availability',return_value=snapshot):
            self.assertEqual(self.panel.resolve_device('auto')[0],'cuda:0')
        snapshot=self.gpu_snapshot()
        snapshot['devices'][1]['free_bytes']=snapshot['devices'][0]['free_bytes']
        with patch.object(self.panel,'gpu_availability',return_value=snapshot):
            self.assertEqual(self.panel.resolve_device('auto')[0],'cuda:0')
        for gpu in snapshot['devices']:gpu['free_bytes']=0
        with patch.object(self.panel,'gpu_availability',return_value=snapshot):
            with self.assertRaises(ValueError):self.panel.resolve_device('auto')

    def test_gpu_probe_cache_and_force_refresh(self):
        response=SimpleNamespace(returncode=0,stdout=json.dumps(self.gpu_snapshot()),stderr='')
        with patch.object(ui.subprocess,'run',return_value=response) as probe:
            self.panel.gpu_availability()
            self.panel.gpu_availability()
            self.assertEqual(probe.call_count,1)
            self.panel.gpu_availability(fresh=True)
            self.assertEqual(probe.call_count,2)
            self.assertEqual(probe.call_args.kwargs['timeout'],25)
            # The subprocess inherits the environment: visibility is never overridden.
            self.assertNotIn('env',probe.call_args.kwargs)

    def test_model_catalog_defaults(self):
        catalog=self.panel.models()
        self.assertEqual(len(catalog['cartilage']),2)
        self.assertEqual(len(catalog['meniscus']),2)
        self.assertEqual(catalog['errors'],[])
        self.assertEqual(Path(catalog['defaults']['cartilage']).name,'top3_unet2_4echo.pt')
        self.assertEqual(Path(catalog['defaults']['meniscus']).name,'joint_unet_4echo_five_losses.pt')
        self.assertEqual(sum(model['default'] for model in catalog['cartilage']),1)
        self.panel.start(self.payload())
        self.assertEqual(self.wait()['state'],'completed')
        forwarded=json.loads((self.root/'output with spaces/forwarded_models.json').read_text())
        self.assertEqual(forwarded,catalog['defaults'])

    def test_independent_model_choices(self):
        alternate=self.cart_dir/'top1_unet_4echo.pt'
        self.panel.start(self.payload(cartilage_model=str(alternate)))
        status=self.wait()
        self.assertEqual(status['state'],'completed')
        self.assertEqual(status['settings']['cartilage_model'],str(alternate))
        self.assertEqual(status['settings']['meniscus_model'],str(self.men_dir/'joint_unet_4echo_five_losses.pt'))
        second=self.root/'second_output'
        men=self.men_dir/'joint_unet_4echo_three_losses.pt'
        self.panel.start(self.payload(output=str(second),meniscus_model=str(men)))
        self.assertEqual(self.wait()['state'],'completed')
        forwarded=json.loads((second/'forwarded_models.json').read_text())
        self.assertEqual(forwarded['cartilage'],str(self.cart_dir/'top3_unet2_4echo.pt'))
        self.assertEqual(forwarded['meniscus'],str(men))
        self.assertEqual(self.panel.models()['defaults']['cartilage'],str(self.cart_dir/'top3_unet2_4echo.pt'))

    def test_custom_model_metadata_and_class_validation(self):
        custom=Path(self.models_temp.name)/'custom';custom.mkdir()
        self.write_models(custom,3,['new_cart'])
        self.panel.start(self.payload(cartilage_model=str(custom/'new_cart.pt')))
        self.assertEqual(self.wait()['state'],'completed')
        with self.assertRaises(ValueError):
            self.panel.start(self.payload(output=str(self.root/'invalid'),meniscus_model=str(custom/'new_cart.pt')))
        naked=self.root/'naked.pt';naked.write_text('mock')
        with self.assertRaises(ValueError):self.panel.model_info(naked,'cartilage')
        with self.assertRaises(ValueError):self.panel.model_info(custom/'missing.pt','cartilage')
        manifest=json.loads((custom/'manifest.json').read_text())
        manifest['models'][0]['inference']['Network']='UnknownNetwork'
        (custom/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):self.panel.model_info(custom/'new_cart.pt','cartilage')

    def test_checkpoint_browser(self):
        entries=self.panel.browse(str(self.cart_dir),'checkpoint')['entries']
        self.assertEqual({item['name'] for item in entries},{'top3_unet2_4echo.pt','top1_unet_4echo.pt'})

    def test_http(self):
        server=ui.ThreadingHTTPServer(('127.0.0.1',0),ui.make_handler(self.panel))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def call(path,data=None,auth=True):
            headers={'Authorization':'Bearer '+self.panel.token} if auth else {}
            if data is not None:
                headers['Content-Type']='application/json'
                data=json.dumps(data).encode()
            with urlopen(Request(base+path,data=data,headers=headers),timeout=5) as response:
                return response.read()
        try:
            self.assertIn(b'Knee segmentation',call('/'))
            with self.assertRaises(HTTPError) as error:call('/api/status',auth=False)
            self.assertEqual(error.exception.code,403)
            self.assertEqual(json.loads(call('/api/status'))['state'],'idle')
            with patch.object(self.panel,'gpu_availability',return_value=self.gpu_snapshot()) as probe:
                gpus=json.loads(call('/api/gpus'))
                self.assertEqual(len(gpus['devices']),2)
                probe.assert_called_once_with(fresh=True)
            self.assertTrue(json.loads(call('/api/browse'))['entries'])
            models=json.loads(call('/api/models'))
            self.assertEqual(Path(models['defaults']['cartilage']).name,'top3_unet2_4echo.pt')
            call('/api/run',self.payload(all_echoes=False,output_parent=str(self.root),output_name='HTTP_results'))
            self.assertEqual(self.wait()['state'],'completed')
            self.assertEqual(json.loads(call('/api/status'))['state'],'completed')
            self.assertEqual(json.loads(call('/api/stop',{}))['state'],'completed')
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':
    unittest.main()
