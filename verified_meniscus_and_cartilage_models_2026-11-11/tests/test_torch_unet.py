"""Strict checkpoint loading and synthetic CPU parity; never opens MRI data.

MONAI is optional and only used as a reference in the parity test. Deployment
and all other tests work without it. No checkpoint weights are used for forward
passes: parity uses deterministic random weights and small synthetic tensors.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from torch_unet import UNet3D

FAMILIES = {
    "UNet": ((16, 32, 64, 128, 256), (2, 2, 2, 2)),
    "UNet2": ((16, 32, 64, 128, 128), (2, 2, 2, 1)),
}
RUNNERS = (ROOT / "verified_cartilage_models_2026-11-10/inference_single.py",
           ROOT / "verified_meniscus_and_cartilage_models_2026-11-11/inference_single.py")


def load_runner(path):
    spec = importlib.util.spec_from_file_location("test_runner_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_four_checkpoints_load_strictly_without_forward_pass(self):
        for path in RUNNERS:
            runner = load_runner(path)
            manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
            for entry in manifest["models"]:
                with self.subTest(checkpoint=entry["checkpoint"]):
                    checkpoint = path.with_name(entry["checkpoint"])
                    cfg = runner.load_settings(checkpoint)
                    model = runner.load_model(cfg, checkpoint, torch.device("cpu"))
                    self.assertIsInstance(model, UNet3D)
                    self.assertFalse(model.training)

    @unittest.skipUnless(importlib.util.find_spec("monai"), "Optional MONAI reference not installed")
    def test_synthetic_logits_and_labels_match_monai(self):
        from monai.networks.nets import UNet
        configurations = (("UNet", 4, 3, 3), ("UNet2", 3, 3, 3),
                          ("UNet2", 4, 3, 3), ("UNet", 4, 5, 3),
                          ("UNet2", 4, 3, 0))
        for family, inc, outc, residuals in configurations:
            with self.subTest(family=family, echoes=inc, classes=outc, residuals=residuals):
                channels, strides = FAMILIES[family]
                with torch.random.fork_rng():
                    torch.manual_seed(731)
                    reference = UNet(spatial_dims=3, in_channels=inc, out_channels=outc,
                                     channels=channels, strides=strides,
                                     num_res_units=residuals, norm="INSTANCE").eval()
                    local = UNet3D(inc, outc, channels, strides, residuals).eval()
                    local.load_state_dict(reference.state_dict(), strict=True)
                    # Unequal axes and two seeds exercise skip concatenation and normalizers.
                    for shape in ((1, inc, 16, 32, 32), (1, inc, 32, 32, 16)):
                        sample = torch.randn(shape)
                        with torch.inference_mode():
                            expected = reference(sample)
                            actual = local(sample)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(torch.equal(actual.argmax(1), expected.argmax(1)))

    def test_both_runners_import_with_monai_blocked(self):
        script = """import importlib.abc, runpy, sys
class BlockMonai(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'monai' or fullname.startswith('monai.'):
            raise ImportError('MONAI deliberately blocked by regression test')
sys.meta_path.insert(0, BlockMonai())
path = sys.argv[1]
sys.argv = [path, '--help']
runpy.run_path(path, run_name='__main__')
"""
        with tempfile.TemporaryDirectory() as working_directory:
            for path in RUNNERS:
                with self.subTest(runner=path.parent.name):
                    result = subprocess.run([sys.executable, "-B", "-c", script, str(path)],
                                            cwd=working_directory, capture_output=True,
                                            text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--model-path", result.stdout)

    def test_invalid_architecture_parameters_are_rejected(self):
        for channels, strides, residuals in (((16,), (), 3),
                                             ((16, 32), (2, 2), 3),
                                             ((16, 32), (0,), 3),
                                             ((16, 32), (2,), -1)):
            with self.subTest(channels=channels, strides=strides, residuals=residuals):
                with self.assertRaises(ValueError):
                    UNet3D(4, 3, channels, strides, residuals)


if __name__ == "__main__":
    unittest.main()
