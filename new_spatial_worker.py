"""Local HRTF and Meta renderers. Pose is an estimate, never ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from spatial import analyze, rms, smooth

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "work/spatial-deps"))
sys.path.insert(0, str(ROOT / "work/spatial-assets/meta"))


class HRTF:
    def __init__(self):
        import h5py

        with h5py.File(ROOT / "work/spatial-assets/nearfield.sofa") as f:
            if float(f["Data.SamplingRate"][0]) != 48000 or np.any(f["Data.Delay"][:]):
                raise ValueError("Unsupported SOFA rate/delay")
            pos = f["SourcePosition"][:]
            ir = f["Data.IR"][:]
        # No elevation/front-back evidence: restrict fitting to horizontal front hemisphere.
        az = (pos[:, 0] + 180) % 360 - 180
        keep = (np.abs(pos[:, 1]) < 1) & (np.abs(az) <= 90)
        self.pos = np.column_stack([az[keep], pos[keep, 2]])
        self.ir = ir[keep]
        if not len(self.ir):
            raise ValueError("SOFA has no horizontal samples")
        self.delay = (
            np.array(
                [np.argmax(np.correlate(h[1], h[0], "full")) - (len(h[0]) - 1) for h in self.ir]
            )
            / 48000
        )
        self.hz = np.fft.rfftfreq(1024, 1 / 48000)
        power = np.abs(np.fft.rfft(self.ir, 1024, axis=2)) ** 2
        self.bands = np.geomspace(250, 7000, 16)
        p = np.stack(
            [
                power[:, :, (self.hz > v / 1.18) & (self.hz < v * 1.18)].mean(axis=2)
                for v in self.bands
            ],
            axis=2,
        )
        self.ild = 10 * np.log10(np.maximum(p[:, 0], 1e-12) / np.maximum(p[:, 1], 1e-12))
        # Common calibration delay removed equally in both ears, preserving ITD.
        onset = int(np.median(np.min(np.argmax(np.abs(self.ir), axis=2), axis=1)))
        self.onset = onset
        # Preserve measured spectral shape and interaural gain, normalize only pair energy.
        self.ir /= np.sqrt(np.sum(self.ir**2, axis=(1, 2))[:, None, None] / 2)

    def trajectory(self, reference):
        cues = analyze(reference, 48000)
        target_ild = np.stack([np.interp(self.bands, cues.band_hz, row) for row in cues.ild_db])
        # Relative distance is underdetermined: a bounded loudness proxy, not a measurement.
        distance = np.clip(0.35 * 10 ** ((np.median(cues.level_db) - cues.level_db) / 40), 0.2, 0.5)
        costs = np.mean(((target_ild[:, None] - self.ild[None]) / 6) ** 2, axis=2)
        costs += ((cues.itd_seconds[:, None] - self.delay[None]) / 0.0002) ** 2
        costs += ((distance[:, None] - self.pos[None, :, 1]) / 0.12) ** 2
        transition = ((self.pos[:, None, 0] - self.pos[None, :, 0]) / 25) ** 2
        transition += ((self.pos[:, None, 1] - self.pos[None, :, 1]) / 0.12) ** 2
        score = costs[0].copy()
        back = []
        for row in costs[1:]:
            all_scores = score[:, None] + transition
            prev = np.argmin(all_scores, axis=0)
            back.append(prev)
            score = row + all_scores[prev, np.arange(len(score))]
        route = [int(np.argmin(score))]
        for prev in back[::-1]:
            route.append(int(prev[route[-1]]))
        route = route[::-1]
        az = smooth(self.pos[route, 0], 9)
        return cues, az, distance

    def render(self, mono, cues, az, distance):
        hop = 512
        size = 2 * hop
        window = np.hanning(size)
        padded = np.pad(mono, (hop, size))
        output = np.zeros((len(padded) + self.ir.shape[-1] - 1, 2))
        for a in range(0, len(padded) - size + 1, hop):
            time = a / 48000
            angle = np.interp(time, cues.times, az)
            dist = np.interp(time, cues.times, distance)
            metric = ((self.pos[:, 0] - angle) / 30) ** 2 + ((self.pos[:, 1] - dist) / 0.1) ** 2
            idx = np.argsort(metric)[:3]
            weights = 1 / np.maximum(metric[idx], 1e-4)
            weights /= weights.sum()
            ir = np.einsum("i,ijk->jk", weights, self.ir[idx])
            frame = padded[a : a + size] * window
            for ear in range(2):
                y = fftconvolve(frame, ir[ear])
                output[a : a + len(y), ear] += y
        return output[hop + self.onset : hop + self.onset + len(mono)]


def level_match(audio, reference, rate=48000):
    """Shared-ear slow level matching, preserving model-generated ILD."""
    hop = rate // 5
    times = np.arange(0, len(audio), hop)
    gains = []
    for t in times:
        a, b = max(0, t - rate // 4), min(len(audio), t + rate // 4)
        target = rms(reference[a:b])
        actual = rms(audio[a:b])
        gains.append(np.clip(target / max(actual, 1e-7), 0.05, 20))
    gain = np.interp(np.arange(len(audio)), times, smooth(np.array(gains), 5))
    return audio * gain[:, None]


def meta_render(net, mono, cues, az, distance):
    import torch

    n = len(mono)
    length = ((n + 399) // 400) * 400
    mono = np.pad(mono, (0, length - n))
    times = np.arange(length // 400) / 120
    angle = np.deg2rad(np.interp(times, cues.times, az))
    # Meta x=front,y=right,z=up; SOFA azimuth positive to the left.
    # Clamp distance to 0.35 m to avoid an unvalidated few-cm neural near-field extrapolation.
    radius = np.maximum(0.35, np.interp(times, cues.times, distance))
    view = np.zeros((7, len(times)), np.float32)
    view[0] = radius * np.cos(angle) - 0.09
    view[1] = -radius * np.sin(angle)
    view[2] = -0.02
    view[6] = 1  # Fixed identity orientation; not inferred from audio.
    context = ((net.receptive_field() + 2000 + 399) // 400) * 400
    output = []
    with torch.inference_mode():
        for a in range(0, length, 48000):
            start = max(0, a - context)
            x = torch.from_numpy(mono[start : a + 48000].astype(np.float32))[None, None].cuda()
            v = torch.from_numpy(view[:, start // 400 : (a + 48000) // 400])[None].cuda()
            # Exact final output of upstream forward; omit unused auxiliary heads.
            warped = net.warper(x, v)
            _, skips = net.hyperconv_wavenet(net.input(warped), v)
            y = net.output_net[-1](torch.stack(skips).mean(0))
            output.append(y[0, :, a - start :].T.cpu().numpy())
    result = np.concatenate(output)[:n]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    from spatial_assets import verify_assets

    verify_assets()
    import torch
    from src.models import BinauralNetwork

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in existing runtime")
    torch.cuda.reset_peak_memory_stats()
    net = BinauralNetwork(wavenet_blocks=3, use_cuda=False)
    net.load_state_dict(
        torch.load(
            ROOT / "work/spatial-assets/meta-3blocks.net", map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    net.eval().cuda()
    hrtf = HRTF()
    tasks = json.loads(args.batch.read_text(encoding="utf-8"))
    report = []
    for i, task in enumerate(tasks):
        mono, rate = sf.read(task["mono"], dtype="float32")
        reference, ref_rate = sf.read(task["reference"], dtype="float32", always_2d=True)
        if rate != 48000 or ref_rate != rate or mono.ndim != 1 or len(mono) != len(reference):
            raise ValueError("Expected aligned 48 kHz mono/reference")
        cues, az, distance = hrtf.trajectory(reference)
        for name, result in [
            ("hrtf", hrtf.render(mono, cues, az, distance)),
            ("meta", meta_render(net, mono, cues, az, distance)),
        ]:
            result = level_match(result, reference)
            if not np.isfinite(result).all() or rms(result) < 1e-8:
                raise ValueError(f"{name} produced invalid/silent audio")
            sf.write(task[name], result, rate, subtype="FLOAT")
        report.append(
            {
                "id": task["id"],
                "azimuth_range": [float(az.min()), float(az.max())],
                "distance_proxy_range_m": [float(distance.min()), float(distance.max())],
            }
        )
        print(f"HRTF/Meta {i + 1}/{len(tasks)}", flush=True)
    args.batch.with_suffix(".report.json").write_text(
        json.dumps(
            {
                "segments": report,
                "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "pose": "estimated horizontal front hemisphere, level-based distance proxy, fixed orientation",
                "meta": "official pretrained 3-block; final head only; float32; 1s chunks with causal context",
                "calibration": "joint-ear slow reference RMS; no Japanese waveform mixed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
