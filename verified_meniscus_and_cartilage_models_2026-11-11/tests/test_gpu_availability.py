"""GPU query tests with a simulated CUDA API, without importing PyTorch."""
import importlib.util
import io
import json
import sys
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'gpu_availability.py'
spec=importlib.util.spec_from_file_location('gpu_availability',path)
gpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(gpu)


class FakeCUDA:
    def __init__(self):self.queried=[]
    def is_available(self):return True
    def device_count(self):return 2
    def get_device_name(self,index):return ['GPU first visible','GPU second visible'][index]
    def mem_get_info(self,index):
        self.queried.append(index)
        if index==0:return 2*1024**3,24*1024**3
        return 18*1024**3,24*1024**3
    def utilization(self,index):
        if index==0:return 40
        raise RuntimeError('NVML unavailable for this device')


class GPUTests(unittest.TestCase):
    def test_only_process_visible_ordinals_are_queried(self):
        cuda=FakeCUDA()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-uuid-B,GPU-uuid-A'}):
            result=gpu.collect(SimpleNamespace(cuda=cuda))
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-uuid-B,GPU-uuid-A')
        self.assertEqual(cuda.queried,[0,1])
        self.assertEqual([item['device'] for item in result['devices']],['cuda:0','cuda:1'])
        self.assertEqual(result['devices'][0]['utilization_percent'],40)
        self.assertIsNone(result['devices'][1]['utilization_percent'])
        self.assertEqual(result['devices'][1]['free_bytes'],18*1024**3)
        self.assertEqual(result['cuda_visible_devices'],'GPU-uuid-B,GPU-uuid-A')

    def test_no_cuda(self):
        cuda=FakeCUDA();cuda.is_available=lambda:False
        result=gpu.collect(SimpleNamespace(cuda=cuda))
        self.assertEqual(result['devices'],[])
        self.assertFalse(result['cuda_available'])
        self.assertEqual(cuda.queried,[])

    def test_one_device_error_does_not_hide_others(self):
        cuda=FakeCUDA()
        def memory(index):
            if index==0:raise RuntimeError('Device inaccessible')
            return 10,20
        cuda.mem_get_info=memory
        result=gpu.collect(SimpleNamespace(cuda=cuda))
        self.assertIn('Device inaccessible',result['devices'][0]['error'])
        self.assertEqual(result['devices'][1]['free_bytes'],10)


    def test_missing_nvml_names_package_and_keeps_memory_results(self):
        cuda=FakeCUDA()
        def utilization(index):
            raise ModuleNotFoundError("nvidia-ml-py does not seem to be installed or it can't be imported.")
        cuda.utilization=utilization
        result=gpu.collect(SimpleNamespace(cuda=cuda))
        self.assertIsNone(result['error'])
        self.assertEqual(result['missing_packages'],['nvidia-ml-py'])
        self.assertEqual(len(result['warnings']),1)
        self.assertIn('python -m pip install nvidia-ml-py',result['warnings'][0])
        self.assertEqual(result['devices'][1]['free_bytes'],18*1024**3)
        self.assertIsNone(result['devices'][1]['error'])
        self.assertIn('nvidia-ml-py is not available',result['devices'][1]['utilization_error'])

    def test_nvml_driver_error_is_not_reported_as_missing_package(self):
        cuda=FakeCUDA()
        def utilization(index):
            raise RuntimeError("NVML driver can't be loaded")
        cuda.utilization=utilization
        result=gpu.collect(SimpleNamespace(cuda=cuda))
        self.assertEqual(result['missing_packages'],[])
        self.assertIn("driver can't be loaded",result['warnings'][0])
        self.assertIsNone(result['error'])

    def test_fatal_missing_nvml_also_reports_dependency(self):
        cuda=FakeCUDA()
        def available():
            raise ModuleNotFoundError("No module named 'pynvml'",name='pynvml')
        cuda.is_available=available
        with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=cuda)}), patch('sys.stdout',new_callable=io.StringIO) as output:
            gpu.main()
        result=json.loads(output.getvalue())
        self.assertEqual(result['missing_packages'],['nvidia-ml-py'])
        self.assertIn('nvidia-ml-py is not available',result['error'])
        self.assertEqual(result['devices'],[])


if __name__=='__main__':unittest.main()
