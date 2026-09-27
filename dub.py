from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import soxr

from bridge import ROOT, Installation, run_process
from spatial import render, rms
from spatial_rtf import render_rtf

RATE = 48000


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def strongest_channel(audio: np.ndarray) -> np.ndarray:
    return audio[:, int(np.argmax(np.mean(audio**2, axis=0)))]


def validate_segments(segments: list[dict], duration: float) -> None:
    if not isinstance(segments, list) or not segments:
        raise ValueError("transcript.json 必须是非空句子数组。")
    previous_end = 0.0
    seen = set()
    for s in segments:
        if not isinstance(s, dict) or not isinstance(s.get("id"), str) or s["id"] in seen:
            raise ValueError("句子 ID 无效或重复。")
        seen.add(s["id"])
        start, end = float(s["start"]), float(s["end"])
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= duration + 0.01:
            raise ValueError(f"{s['id']}: 时间超出原音频。")
        if start < previous_end - 0.001:
            raise ValueError(f"{s['id']}: 句子重叠；此原型仅支持单人、不重叠台词。")
        if end - start < 0.15:
            raise ValueError(f"{s['id']}: 句子不足 0.15 秒，请在 transcript.json 中合并。")
        if not isinstance(s.get("ja"), str) or not s["ja"].strip():
            raise ValueError(f"{s['id']}: 缺少日文。")
        if "zh" in s and not isinstance(s["zh"], str):
            raise ValueError(f"{s['id']}: 中文必须是字符串。")
        previous_end = end


def write_reference(path: Path, mono: np.ndarray) -> None:
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-7:
        raise ValueError("音色参考片段没有有效声音。")
    gain = min(0.8 / peak, 0.06 / max(rms(mono), 1e-8))
    sf.write(
        path,
        soxr.resample(mono * gain, RATE, 24000, quality="VHQ"),
        24000,
        subtype="PCM_16",
    )


def trim_voice(audio: np.ndarray, rate: int) -> np.ndarray:
    active = np.flatnonzero(np.abs(audio) > max(float(np.max(np.abs(audio))) * 0.018, 1e-5))
    if not len(active):
        raise ValueError("TTS 输出为静音。")
    pad = round(rate * 0.03)
    return audio[max(0, active[0] - pad) : min(len(audio), active[-1] + pad + 1)]


def fit_voice(path: Path, target_frames: int, workspace: Path) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = trim_voice(strongest_channel(audio), sr)
    speed = len(mono) / sr / (target_frames / RATE)
    if speed > 2.0:
        raise ValueError(
            f"配音时长与原句差异过大（需 {speed:.2f} 倍速）。请修改 transcript.json 中文长度后 resume。"
        )
    # Interjections can be much shorter than the ASR window (which includes pauses).
    # Do not turn a short "eh" into a long drone just to occupy the whole window.
    speed = max(0.8, speed)
    intermediate = workspace / "fit_input.wav"
    fitted = workspace / "fit_output.wav"
    sf.write(intermediate, mono, sr, subtype="FLOAT")
    run_process(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(intermediate),
            "-af",
            f"rubberband=tempo={speed:.9f}:pitch=1",
            "-ar",
            str(RATE),
            "-c:a",
            "pcm_f32le",
            str(fitted),
        ],
        cwd=workspace,
        log=workspace / "fit.log",
        timeout=180,
    )
    result, _ = sf.read(fitted, dtype="float32")
    # Rubber Band can leave a short silent tail; never cut audible overrun.
    if len(result) > target_frames:
        tail = result[target_frames:]
        if len(tail) > RATE * 0.05 and rms(tail) > max(1e-5, rms(result) * 0.02):
            raise ValueError("时长对齐出现非静音溢出；停止，避免截断台词。")
        result = result[:target_frames]
    remaining = max(0, target_frames - len(result))
    result = np.pad(result, (remaining // 2, remaining - remaining // 2))
    return result, speed


def srt(segments: list[dict]) -> str:
    def clock(value):
        ms = round(float(value) * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"

    return (
        "\n\n".join(
            f"{i}\n{clock(s['start'])} --> {clock(s['end'])}\n{s['zh']}"
            for i, s in enumerate(segments, 1)
        )
        + "\n"
    )


def process(installation: Installation, output: Path, stop_after: str | None = None):
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if Path(manifest["asmr_root"]).resolve() != installation.repo:
        raise ValueError(
            "原项目路径与这次测试不一致；请用 --asmr-root 指定 run.json 中记录的路径。"
        )
    workspace = Path(manifest["workspace"]).resolve()
    if not workspace.is_relative_to(ROOT / "work") or not workspace.is_dir():
        raise ValueError("工作目录无效或被移动，请新建测试。")
    source, rate = sf.read(workspace / "source.wav", dtype="float32", always_2d=True)
    duration = len(source) / rate
    transcript = output / "transcript.json"
    if transcript.is_file():
        segments = json.loads(transcript.read_text(encoding="utf-8"))
    else:
        print("[1/4] Parakeet 1.1B 日语识别；原录音不上传。", flush=True)
        analysis = soxr.resample(strongest_channel(source), rate, 16000, quality="VHQ")
        segments = installation.transcribe(analysis, 16000, workspace)
        save_json(transcript, segments)
    validate_segments(segments, duration)
    if stop_after == "asr":
        print(f"已停在 ASR，可编辑：{transcript}")
        return
    if any(not s.get("zh", "").strip() for s in segments):
        print("[2/4] DeepSeek V4 Flash 翻译；仅日文文本发送至 DeepSeek。", flush=True)
        try:
            installation.translate(segments, checkpoint=lambda: save_json(transcript, segments))
        finally:
            save_json(transcript, segments)
    if stop_after == "translate":
        print(f"已停在翻译，可编辑：{transcript}")
        return
    # A shared timbre prompt keeps identity consistent; phrase-local prompts carry style.
    best = max(segments, key=lambda s: min(s["end"] - s["start"], 8))
    a, b = (
        round(best["start"] * rate),
        round(min(best["end"], best["start"] + 9) * rate),
    )
    if b - a < rate:
        a, b = max(0, a - rate), min(len(source), b + rate)
    voice_path = workspace / "speaker.wav"
    write_reference(voice_path, strongest_channel(source[a:b]))
    tasks, clips = [], []
    for i, segment in enumerate(segments):
        a, b = round(segment["start"] * rate), round(segment["end"] * rate)
        emotion_path = workspace / f"emotion_{i:04d}.wav"
        pad = round(max(0, 1.2 - (b - a) / rate) * rate / 2)
        write_reference(
            emotion_path,
            strongest_channel(source[max(0, a - pad) : min(len(source), b + pad)]),
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {"zh": segment["zh"], "target": b - a, "version": 2, "tts_backend": "indextts2.0"},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
            + voice_path.read_bytes()
            + emotion_path.read_bytes()
        ).hexdigest()[:20]
        clip = workspace / f"tts_{i:04d}_{fingerprint}.wav"
        clips.append(clip)
        if not clip.is_file():
            tasks.append(
                {
                    "id": segment["id"],
                    "text": segment["zh"],
                    "voice": str(voice_path),
                    "emotion": str(emotion_path),
                    "output": str(clip),
                    "target_seconds": (b - a) / rate,
                }
            )
    if tasks:
        print(
            f"[3/4] IndexTTS-2.0 克隆音色与逐句风格（{len(tasks)} 句）；进度见 {workspace / 'tts.log'}",
            flush=True,
        )
        installation.synthesize(tasks, workspace)
    print("[4/4] 对齐中文时长，迁移双耳特征，输出空间版与居中对照版。", flush=True)
    stereo = np.zeros_like(source)
    centered = np.zeros_like(source)
    rtf_stereo = np.zeros_like(source)
    reports, curve_rows = (
        [],
        ["sentence,time_seconds,itd_ms,median_ild_db,relative_level_db,confidence"],
    )
    for segment, clip in zip(segments, clips, strict=True):
        a, b = round(segment["start"] * rate), round(segment["end"] * rate)
        mono, speed = fit_voice(clip, b - a, workspace)
        phrase, cues = render(mono, source[a:b], rate)
        baseline, _ = render(mono, source[a:b], rate, spatial=False)
        rtf_phrase, rtf_report = render_rtf(mono, source[a:b], rate)
        stereo[a:b] += phrase
        centered[a:b] += baseline
        rtf_stereo[a:b] += rtf_phrase
        reports.append(
            {"id": segment["id"], "alignment_tempo": speed, **cues.summary(), "rtf": rtf_report}
        )
        for j, t in enumerate(cues.times):
            curve_rows.append(
                f"{segment['id']},{t + a / rate:.4f},{cues.itd_seconds[j] * 1000:.5f},"
                f"{np.median(cues.ild_db[j]):.4f},{cues.level_db[j]:.4f},{cues.confidence[j]:.4f}"
            )
    # Global attenuation only; independent ear normalization would destroy the ILD.
    peak = max(float(np.max(np.abs(stereo))), float(np.max(np.abs(centered))))
    limiter_gain = min(1.0, 0.95 / max(peak, 1e-8))
    sf.write(output / "zh_spatial.wav", stereo * limiter_gain, rate, subtype="PCM_24")
    sf.write(output / "zh_center.wav", centered * limiter_gain, rate, subtype="PCM_24")
    rtf_gain = min(1.0, 0.95 / max(float(np.max(np.abs(rtf_stereo))), 1e-8))
    sf.write(output / "zh_rtf.wav", rtf_stereo * rtf_gain, rate, subtype="PCM_24")
    (output / "zh.srt").write_text(srt(segments), encoding="utf-8-sig")
    (output / "spatial_cues.csv").write_text("\n".join(curve_rows) + "\n", encoding="utf-8")
    save_json(
        output / "report.json",
        {
            "sample_rate": rate,
            "duration_seconds": duration,
            "global_peak_gain": limiter_gain,
            "rtf_peak_gain": rtf_gain,
            "source_audio_mixed": False,
            "models": {
                "asr": "Parakeet CTC 1.1B Japanese",
                "translation": "deepseek-v4-flash",
                "tts": "IndexTTS-2.0",
                "spatial": "v1 cue transfer + complex RTF v2; neither is a neural model",
            },
            "limitations": [
                "No metric distance/front-back/elevation reconstruction",
                "No room impulse response recovery",
                "No original breaths or background retained outside recognized speech",
                "One speaker only",
            ],
            "segments": reports,
        },
    )
    print(
        f"完成，新 RTF 版：{output / 'zh_rtf.wav'}\n旧版：{output / 'zh_spatial.wav'}\n居中：{output / 'zh_center.wav'}",
        flush=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="日语音声 → 中文音色克隆 + 实验性双耳特征迁移")
    parser.add_argument("--asmr-root", type=Path, default=ROOT.parent / "asmr-next")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="检查模型和密钥是否存在，不显示密钥")
    run = subs.add_parser("run", help="新建一次测试，输入必须是真正的双声道录音")
    run.add_argument("input", type=Path)
    run.add_argument("--out", type=Path)
    run.add_argument(
        "--allow-mono",
        action="store_true",
        help="允许单声道，仅测试配音，不会凭空恢复空间",
    )
    run.add_argument("--stop-after", choices=["asr", "translate"])
    run.add_argument("--all-methods", action="store_true", help="完成配音后生成五种空间方案对照")
    resume = subs.add_parser("resume", help="继续运行，可先修改 transcript.json 中的中文")
    resume.add_argument("output", type=Path)
    resume.add_argument("--stop-after", choices=["asr", "translate"])
    resume.add_argument("--all-methods", action="store_true", help="完成配音后生成五种空间方案对照")
    compare = subs.add_parser(
        "compare", help="复用已有中文缓存，在新目录生成 RTF/旧版等响度对照，不调用 API"
    )
    compare.add_argument("output", type=Path)
    compare.add_argument("--out", type=Path)
    compare.add_argument(
        "--all-methods", action="store_true", help="增加实测 HRTF 和 Meta 神经双耳模型"
    )
    args = parser.parse_args(argv)
    installation = Installation(args.asmr_root)
    if getattr(args, "all_methods", False):
        from spatial_assets import verify_assets

        verify_assets()
    if args.command == "compare":
        from compare import compare_run

        compare_run(installation, args.output.resolve(), args.out, args.all_methods)
        return 0
    if args.command == "doctor":
        checks = installation.doctor()
        for name, ok in checks.items():
            print(f"{'OK' if ok else 'MISSING'}  {name}")
        return 0 if all(checks.values()) else 1
    if args.command == "resume":
        process(installation, args.output.resolve(), args.stop_after)
        if args.all_methods and not args.stop_after:
            from compare import compare_run

            compare_run(installation, args.output.resolve(), all_methods=True)
        return 0
    original = args.input.resolve()
    if not original.is_file():
        raise ValueError(f"输入不存在：{original}")
    from asmr_dubber.audio import probe_audio

    info = probe_audio(original)
    if info.channels not in (1, 2):
        raise ValueError("仅支持单/双声道，不能把环绕声当成双耳录音。")
    if info.channels == 1 and not args.allow_mono:
        raise ValueError(
            "输入是单声道，没有可迁移的双耳线索。请提供双声道，或用 --allow-mono 只测配音。"
        )
    if info.duration_seconds > 600:
        raise ValueError("这是短片段测试程序，单次最多 10 分钟；请先截取代表性片段。")
    output = (
        args.out
        or ROOT
        / "outputs"
        / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    ).resolve()
    if output.is_relative_to(installation.repo):
        raise ValueError("请把测试输出留在新项目，不写入原 ASMR Dubber 项目。")
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录已有内容；使用 resume，或换一个新目录。")
    output.mkdir(parents=True, exist_ok=True)
    workspace = ROOT / "work" / uuid.uuid4().hex
    workspace.mkdir(parents=True)
    if not str(workspace).isascii():
        raise ValueError("测试项目路径需使用英文，以兼容 CrispASR。输入音频可使用中文路径。")
    run_process(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(original),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(RATE),
            "-c:a",
            "pcm_f32le",
            str(workspace / "source.wav"),
        ],
        cwd=workspace,
        log=workspace / "decode.log",
        timeout=180,
    )
    save_json(
        output / "run.json",
        {
            "source": str(original),
            "source_sha256": info.sha256,
            "original_channels": info.channels,
            "workspace": str(workspace),
            "asmr_root": str(installation.repo),
        },
    )
    print(f"测试目录：{output}", flush=True)
    if info.channels == 1:
        print("注意：单声道测试，输出没有恢复原本不存在的空间信息。", flush=True)
    process(installation, output, args.stop_after)
    if args.all_methods and not args.stop_after:
        from compare import compare_run

        compare_run(installation, output, all_methods=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已停止。保留中间产物，可用 resume 继续。", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from None
