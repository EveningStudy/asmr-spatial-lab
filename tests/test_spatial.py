import unittest

import numpy as np

from bridge import expand_ctc_boundaries
from dub import validate_segments
from spatial import analyze, render, rms, smooth


class SpatialTests(unittest.TestCase):
    rate = 24000

    def signal(self, seconds=2):
        rng = np.random.default_rng(17)
        noise = rng.normal(0, 0.05, round(self.rate * seconds))
        # Broadband deterministic signal, band-limited to a speech-like range.
        spectrum = np.fft.rfft(noise)
        hz = np.fft.rfftfreq(len(noise), 1 / self.rate)
        spectrum[(hz < 180) | (hz > 8000)] = 0
        return np.fft.irfft(spectrum, n=len(noise)).astype(np.float32)

    def test_smoothing_length_and_constant(self):
        for size in (1, 2, 10):
            x = np.ones((size, 3))
            np.testing.assert_allclose(smooth(x, 19), x)

    def test_mono_is_centered(self):
        x = self.signal()
        cues = analyze(np.column_stack([x, x]), self.rate)
        self.assertLess(np.max(np.abs(cues.itd_seconds)), 1e-6)
        self.assertLess(np.max(np.abs(cues.ild_db)), 1e-6)
        out, _ = render(x, np.column_stack([x, x]), self.rate)
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-7)
        self.assertEqual(out.shape, (len(x), 2))
        self.assertTrue(np.isfinite(out).all())

    def test_positive_itd_and_ild(self):
        x = self.signal()
        delayed = np.pad(x, (10, 0))[: len(x)] * 0.5
        reference = np.column_stack([x, delayed])
        cues = analyze(reference, self.rate)
        self.assertAlmostEqual(
            float(np.median(cues.itd_seconds)), 10 / self.rate, delta=1 / self.rate
        )
        self.assertAlmostEqual(float(np.median(cues.ild_db)), 6.0206, delta=0.15)
        out, _ = render(self.signal(), reference, self.rate)
        recovered = analyze(out, self.rate)
        self.assertAlmostEqual(
            float(np.median(recovered.itd_seconds)),
            10 / self.rate,
            delta=1.5 / self.rate,
        )
        self.assertAlmostEqual(20 * np.log10(rms(out[:, 0]) / rms(out[:, 1])), 6.0206, delta=0.3)

    def test_channel_swap_reverses_direction(self):
        x = self.signal()
        ref = np.column_stack([x, np.pad(x, (8, 0))[: len(x)] * 0.7])
        a, b = analyze(ref, self.rate), analyze(ref[:, ::-1], self.rate)
        np.testing.assert_allclose(a.itd_seconds, -b.itd_seconds, atol=1e-6)
        np.testing.assert_allclose(a.ild_db, -b.ild_db, atol=1e-6)

    def test_motion_and_distance_level(self):
        x = self.signal(4)
        p = np.linspace(0, 1, len(x))
        ref = np.column_stack([x * (1 - 0.8 * p), x * (0.2 + 0.8 * p)])
        out, _ = render(x, ref, self.rate)
        n = self.rate
        self.assertGreater(rms(out[:n, 0]), rms(out[:n, 1]) * 1.5)
        self.assertGreater(rms(out[-n:, 1]), rms(out[-n:, 0]) * 1.5)
        envelope = 1 - p * 0.8
        far, _ = render(x, np.column_stack([x * envelope, x * envelope]), self.rate)
        self.assertGreater(rms(far[:n]), rms(far[-n:]) * 1.6)

    def test_silence_gaps_do_not_reset_direction(self):
        x = self.signal()
        x[self.rate // 2 : self.rate] = 0
        ref = np.column_stack([x, np.pad(x, (8, 0))[: len(x)] * 0.5])
        cues = analyze(ref, self.rate)
        self.assertGreater(np.median(cues.ild_db), 5.8)
        self.assertTrue(np.isfinite(cues.level_db).all())

    def test_invalid_input(self):
        for invalid in (
            np.zeros((2400, 2)),
            np.full((2400, 2), np.nan),
            np.zeros(2400),
        ):
            with self.assertRaises(ValueError):
                analyze(invalid, self.rate)

    def test_ctc_boundary_padding_and_short_merge(self):
        items = [
            {"id": "a", "start": 0.5, "end": 0.58, "ja": "え？", "zh": ""},
            {"id": "b", "start": 1.0, "end": 2.0, "ja": "本当？", "zh": ""},
            {"id": "c", "start": 3.0, "end": 4.0, "ja": "はい", "zh": ""},
        ]
        padded = expand_ctc_boundaries(items, 4.1)
        self.assertEqual(len(padded), 2)
        self.assertEqual(padded[0]["ja"], "え？本当？")
        self.assertAlmostEqual(padded[0]["start"], 0.22)
        self.assertAlmostEqual(padded[-1]["end"], 4.1)
        validate_segments(padded, 4.1)

    def test_one_quiet_ear_stays_finite(self):
        x = self.signal()
        out, cues = render(x, np.column_stack([x, x * 1e-6]), self.rate)
        self.assertTrue(np.isfinite(out).all())
        self.assertLessEqual(float(np.max(cues.ild_db)), 24.0001)
        self.assertGreater(rms(out[:, 0]), rms(out[:, 1]) * 10)

    def test_baseline_does_not_pan(self):
        x = self.signal()
        out, _ = render(x, np.column_stack([x, x * 0.2]), self.rate, spatial=False)
        np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-7)

    def test_reject_overlapping_and_out_of_bounds_transcript(self):
        valid = [{"id": "a", "start": 0, "end": 1, "ja": "はい", "zh": "好"}]
        validate_segments(valid, 2)
        for bad in (
            [{**valid[0], "end": 3}],
            [*valid, {**valid[0], "id": "b", "start": 0.5}],
            [{**valid[0], "end": float("nan")}],
            [*valid, valid[0]],
        ):
            with self.assertRaises(ValueError):
                validate_segments(bad, 2)


if __name__ == "__main__":
    unittest.main()
