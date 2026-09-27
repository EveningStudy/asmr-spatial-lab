"""Integration checks; run with the existing IndexTTS Python (torch/scipy)."""

import unittest

import numpy as np
import torch

from new_spatial_worker import HRTF, ROOT, meta_render
from spatial import analyze, rms
from spatial_assets import verify_assets


class SpatialRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_assets()
        from src.models import BinauralNetwork

        torch.set_num_threads(4)
        cls.hrtf = HRTF()
        cls.net = BinauralNetwork(wavenet_blocks=3, use_cuda=False)
        cls.net.load_state_dict(
            torch.load(
                ROOT / "work/spatial-assets/meta-3blocks.net", weights_only=True, map_location="cpu"
            )
        )
        cls.net.eval().cuda()

    def test_hrtf_direction(self):
        x = np.random.default_rng(17).normal(0, 0.02, 48000).astype(np.float32)
        cues = analyze(np.column_stack([x, x]), 48000)
        outputs = []
        for angle in [60, -60]:
            y = self.hrtf.render(
                x, cues, np.full(len(cues.times), angle), np.full(len(cues.times), 0.35)
            )
            self.assertEqual(y.shape, (48000, 2))
            self.assertTrue(np.isfinite(y).all())
            outputs.append(20 * np.log10(rms(y[:, 0]) / rms(y[:, 1])))
        self.assertGreater(outputs[0], 0)
        self.assertLess(outputs[1], 0)

    def test_meta_final_head_equivalence(self):
        x = torch.randn(1, 1, 4800, device="cuda") * 0.02
        v = torch.zeros(1, 7, 12, device="cuda")
        v[:, 0] = 0.8
        v[:, 6] = 1
        with torch.inference_mode():
            full = self.net(x, v)["output"]
            _, skips = self.net.hyperconv_wavenet(self.net.input(self.net.warper(x, v)), v)
            optimized = self.net.output_net[-1](torch.stack(skips).mean(0))
        torch.testing.assert_close(full, optimized)

    def test_meta_direction_and_chunking(self):
        x = np.random.default_rng(18).normal(0, 0.02, 100800).astype(np.float32)
        cues = analyze(np.column_stack([x, x]), 48000)
        differences = []
        for angle in [60, -60]:
            y = meta_render(
                self.net, x, cues, np.full(len(cues.times), angle), np.full(len(cues.times), 0.5)
            )
            self.assertEqual(y.shape, (len(x), 2))
            self.assertTrue(np.isfinite(y).all())
            differences.append(20 * np.log10(rms(y[:, 0]) / rms(y[:, 1])))
            self.assertLess(abs(y[48000] - y[47999]).max(), 1)
        self.assertGreater(differences[0], 0)
        self.assertLess(differences[1], 0)


if __name__ == "__main__":
    unittest.main()
