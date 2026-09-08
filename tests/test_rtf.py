import unittest

import numpy as np

from spatial import rms
from spatial_rtf import filters, render_rtf, stft


def noise(seed, n=48000):
    return np.random.default_rng(seed).normal(0, 0.05, n).astype(np.float32)


def relative_phase(audio):
    spec = stft(audio, 2048, 256)[4:-4]
    return np.angle(np.mean(spec[:, :, 1] * spec[:, :, 0].conj(), axis=0))


class RTFTests(unittest.TestCase):
    def test_identity(self):
        x = noise(1)
        y, report = render_rtf(x, np.column_stack([x, x]), 24000)
        self.assertEqual(y.shape, (len(x), 2))
        np.testing.assert_allclose(y[:, 0], y[:, 1], atol=1e-6)
        self.assertTrue(np.isfinite(y).all())
        self.assertGreater(report["mean_rtf_confidence"], 0.5)

    def test_direction_and_swap(self):
        x = noise(2)
        reference = np.column_stack([x, np.pad(x, (10, 0))[: len(x)] * 0.5])
        y, _ = render_rtf(noise(3), reference, 24000)
        swapped, _ = render_rtf(noise(3), reference[:, ::-1], 24000)
        self.assertAlmostEqual(20 * np.log10(rms(y[:, 0]) / rms(y[:, 1])), 6.02, delta=0.3)
        np.testing.assert_allclose(y, swapped[:, ::-1], atol=1e-5)

    def test_nonlinear_phase_is_retained(self):
        x = noise(4, 96000)
        kernel = np.zeros(100)
        kernel[9], kernel[37] = 0.65, 0.3
        reference = np.column_stack([x, np.convolve(x, kernel)[: len(x)]])
        response, _, _ = filters(reference, 24000, 2048, 256)
        ratio = np.mean(response[8:-8, :, 1] / response[8:-8, :, 0], axis=0)
        target = np.fft.rfft(kernel, 2048)
        hz = np.fft.rfftfreq(2048, 1 / 24000)
        keep = (hz > 200) & (hz < 9000)
        error = np.angle(ratio[keep] * target[keep].conj())
        self.assertLess(rms(error), 0.18)

    def test_noise_and_quiet_ear_bounded(self):
        x = noise(7)
        for ref in (np.column_stack([x, noise(8)]), np.column_stack([x, x * 1e-7])):
            y, _ = render_rtf(x, ref, 24000)
            self.assertTrue(np.isfinite(y).all())
            self.assertLess(np.max(np.abs(y)), 2)

    def test_different_lengths(self):
        x = noise(2)
        y, _ = render_rtf(noise(3, 35000), np.column_stack([x, x * 0.5]), 24000)
        self.assertEqual(y.shape, (35000, 2))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            render_rtf(np.zeros(5000), np.ones((5000, 2)), 24000)

    def test_motion_at_output_sample_rate(self):
        rate = 48000
        x = noise(9, rate * 3)
        grid = np.arange(len(x))
        trajectory = np.linspace(-1, 1, len(x))
        left = np.interp(grid - np.maximum(trajectory * 25, 0), grid, x)
        right = np.interp(grid - np.maximum(-trajectory * 25, 0), grid, x)
        ref = np.column_stack([left * (1 - 0.6 * trajectory), right * (1 + 0.6 * trajectory)])
        y, _ = render_rtf(noise(10, len(x)), ref, rate)
        self.assertGreater(rms(y[:rate, 0]), rms(y[:rate, 1]) * 1.4)
        self.assertGreater(rms(y[-rate:, 1]), rms(y[-rate:, 0]) * 1.4)
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
