"""Independent analytic-filter benchmark; not an ASMR perceptual benchmark.

Reference and render input use different random signals. Ground truth is generated
by known FIR filters, not by either tested renderer. Report all cases, no cherry-pick.
"""

import json
from pathlib import Path

import numpy as np

from spatial import render, rms
from spatial_rtf import render_rtf, stft


def measurement(audio):
    spec = stft(audio, 2048, 256)[8:-8]
    power = np.mean(np.abs(spec) ** 2, axis=0)
    cross = np.mean(spec[:, :, 1] * spec[:, :, 0].conj(), axis=0)
    return 10 * np.log10(np.maximum(power[:, 0], 1e-15) / np.maximum(power[:, 1], 1e-15)), np.angle(
        cross
    )


def main():
    rate = 24000
    rng = np.random.default_rng(20260908)
    ref_signal = rng.normal(0, 0.035, rate * 4)
    target_signal = rng.normal(0, 0.035, rate * 4)
    cases = {
        "delay_only": [(10, 0.6)],
        "short_reflection": [(9, 0.65), (37, 0.3)],
        "differential_notches": [(7, 0.7), (53, -0.38), (91, 0.14)],
    }
    hz = np.fft.rfftfreq(2048, 1 / rate)
    keep = (hz > 200) & (hz < 9000)
    result = {}
    for name, taps in cases.items():
        kernel = np.zeros(128)
        for delay, amplitude in taps:
            kernel[delay] = amplitude
        reference = np.column_stack(
            [ref_signal, np.convolve(ref_signal, kernel)[: len(ref_signal)]]
        )
        ground_truth = np.column_stack(
            [target_signal, np.convolve(target_signal, kernel)[: len(target_signal)]]
        )
        truth_ild, truth_phase = measurement(ground_truth)
        methods = {
            "v1": render(target_signal, reference, rate)[0],
            "rtf": render_rtf(target_signal, reference, rate)[0],
        }
        result[name] = {}
        for method, audio in methods.items():
            ild, phase = measurement(audio)
            result[name][method] = {
                "ild_rmse_db": rms((ild - truth_ild)[keep]),
                "interaural_phase_rmse_radians": rms(
                    np.angle(np.exp(1j * (phase - truth_phase)))[keep]
                ),
            }
    path = Path(__file__).parent / "outputs/rtf-benchmark.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
