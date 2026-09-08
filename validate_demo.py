"""Create a labeled synthetic moving source from an existing Japanese speech fixture.

This is an integration fixture, not evidence of reconstruction of a real binaural room.
"""

import argparse

import numpy as np
import soundfile as sf
import soxr

from bridge import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--long", action="store_true", help="Use a 15-second conversational fixture"
    )
    args = parser.parse_args()
    filename = "japanese_human_180s.wav" if args.long else "japanese_human_emotion.wav"
    original = ROOT.parent / "asmr-next/tests/assets/smoke" / filename
    audio, rate = sf.read(original, dtype="float32", always_2d=True)
    if args.long:
        audio = audio[rate * 5 : rate * 20]
    mono = soxr.resample(audio[:, 0], rate, 48000)
    t = np.linspace(0, 1, len(mono))
    # Move left -> right; approach around the middle, then recede.
    side = np.sin((t - 0.5) * np.pi)
    delay_samples = side * 0.00055 * 48000
    grid = np.arange(len(mono))
    left = np.interp(grid - np.maximum(delay_samples, 0), grid, mono, left=0, right=0)
    right = np.interp(grid - np.maximum(-delay_samples, 0), grid, mono, left=0, right=0)
    ratio = 10 ** (-side * 10 / 20)
    right_gain = np.sqrt(2 / (1 + ratio**2))
    distance_gain = 0.35 + 0.65 * np.sin(t * np.pi)
    result = (
        np.column_stack([left * right_gain * ratio, right * right_gain]) * distance_gain[:, None]
    )
    result *= min(1, 0.8 / np.max(np.abs(result)))
    folder = ROOT / "outputs/integration-fixture"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("ja_synthetic_motion_15s.wav" if args.long else "ja_synthetic_motion.wav")
    sf.write(path, result, 48000, subtype="PCM_24")
    print(path)


if __name__ == "__main__":
    main()
