"""Render the same cached Chinese samples through both baselines; no ASR/API/TTS."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from bridge import ROOT
from spatial import render, rms
from spatial_rtf import render_rtf


def compare_run(installation, previous: Path, destination: Path | None = None):
    from dub import fit_voice, save_json, validate_segments

    manifest = json.loads((previous / "run.json").read_text(encoding="utf-8"))
    workspace = Path(manifest["workspace"]).resolve()
    if not workspace.is_relative_to(ROOT / "work"):
        raise ValueError("工作目录不在本测试项目内。")
    segments = json.loads((previous / "transcript.json").read_text(encoding="utf-8"))
    info = sf.info(workspace / "source.wav")
    validate_segments(segments, info.duration)
    if info.channels != 2:
        raise ValueError("需要双声道参考音频。")
    output = (destination or previous / ("compare-rtf-" + uuid.uuid4().hex[:8])).resolve()
    if output.is_relative_to(installation.repo) or output == previous:
        raise ValueError("对照输出必须使用新目录，不能覆盖原结果或原项目。")
    output.mkdir(parents=True, exist_ok=False)
    rate = info.samplerate
    methods = {name: np.zeros((info.frames, 2), np.float32) for name in ("v1", "rtf", "center")}
    reports = []
    voice_bytes = (workspace / "speaker.wav").read_bytes()
    with sf.SoundFile(workspace / "source.wav") as source:
        for i, segment in enumerate(segments):
            a, b = round(segment["start"] * rate), round(segment["end"] * rate)
            fingerprint = hashlib.sha256(
                json.dumps(
                    {"zh": segment["zh"], "target": b - a, "version": 1},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
                + voice_bytes
                + (workspace / f"emotion_{i:04d}.wav").read_bytes()
            ).hexdigest()[:20]
            clip = workspace / f"tts_{i:04d}_{fingerprint}.wav"
            if not clip.is_file():
                raise ValueError(
                    f"{segment['id']} 配音缓存不匹配；先 resume 生成，compare 不会调用付费服务。"
                )
            mono, speed = fit_voice(clip, b - a, output)
            source.seek(a)
            reference = source.read(b - a, dtype="float32", always_2d=True)
            v1, _ = render(mono, reference, rate)
            rtf, diagnostic = render_rtf(mono, reference, rate)
            center, _ = render(mono, reference, rate, spatial=False)
            for name, audio in (("v1", v1), ("rtf", rtf), ("center", center)):
                methods[name][a:b] = audio
            reports.append({"id": segment["id"], "alignment_tempo": speed, **diagnostic})
            print(f"空间对照 {i + 1}/{len(segments)}", flush=True)
    # Joint-channel whole-track RMS matching, never independent ear normalization.
    target = rms(methods["v1"])
    gains = {name: target / max(rms(audio), 1e-12) for name, audio in methods.items()}
    peak = max(float(np.max(np.abs(audio))) * gains[name] for name, audio in methods.items())
    ceiling = min(1.0, 0.95 / max(peak, 1e-12))
    for name, audio in methods.items():
        sf.write(output / f"zh_{name}.wav", audio * gains[name] * ceiling, rate, subtype="PCM_24")
    save_json(
        output / "comparison.json",
        {
            "previous_run": str(previous),
            "same_tts_samples": True,
            "new_api_or_tts_calls": False,
            "matching": "whole-track joint-channel RMS (not perceptual loudness)",
            "rms_gains": gains,
            "shared_peak_gain": ceiling,
            "segments": reports,
            "quality_winner": "not established without listening on real binaural material",
        },
    )
    print(f"对照完成：{output}", flush=True)
