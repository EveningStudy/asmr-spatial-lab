"""Small, read-only adapter to the adjacent ASMR Dubber installation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent


def run_process(
    command: list[str],
    *,
    cwd: Path,
    log: Path,
    env: dict[str, str] | None = None,
    timeout: float = 1800,
) -> None:
    """Capture noisy native output locally; kill the child on cancellation/timeout."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            process.kill()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"子进程退出码 {code}；详情见 {log}")


@dataclass
class Installation:
    repo: Path

    def __post_init__(self):
        self.repo = self.repo.resolve()
        if not (self.repo / "src/asmr_dubber/asr.py").is_file():
            raise ValueError(f"找不到 ASMR Dubber 源码：{self.repo}")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(self.repo / "src"))

    @property
    def home(self) -> Path:
        return self.repo / ".asmr-dubber"

    @property
    def runtime(self) -> Path:
        return self.home / "runtimes/index-tts"

    @property
    def tts_python(self) -> Path:
        return self.runtime / ".venv/Scripts/python.exe"

    @property
    def asr_model(self) -> Path:
        return self.home / "models/parakeet/parakeet-ctc-1.1b-ja-f16.gguf"

    @property
    def crisp(self) -> Path:
        return self.home / "runtimes/crispasr/bin/crispasr.exe"

    def key(self) -> str:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        path = self.home / "config/secrets.json"
        if not key and path.is_file():
            key = str(
                json.loads(path.read_text(encoding="utf-8-sig"))
                .get("api_keys", {})
                .get("deepseek", "")
            ).strip()
        return key

    def doctor(self) -> dict:
        resources = {
            "Parakeet 1.1B": self.asr_model,
            "CrispASR": self.crisp,
            "IndexTTS-2.0 Python": self.tts_python,
            **{
                f"IndexTTS-2.0 {name}": self.runtime / "checkpoints" / name
                for name in ("config.yaml", "gpt.pth", "s2mel.pth", "bpe.model")
            },
        }
        return {
            **{name: path.is_file() for name, path in resources.items()},
            "DeepSeek API Key (value never displayed)": bool(self.key()),
        }

    def environment(self, workspace: Path, name: str) -> dict[str, str]:
        env = os.environ.copy()
        state = workspace / "runtime-state" / name
        for key, sub in {
            "APPDATA": "roaming",
            "LOCALAPPDATA": "local",
            "XDG_CACHE_HOME": "cache",
            "XDG_CONFIG_HOME": "config",
            "TEMP": "temp",
            "TMP": "temp",
        }.items():
            path = state / sub
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
        env.update(
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        return env

    def transcribe(self, mono: np.ndarray, rate: int, workspace: Path) -> list[dict]:
        from asmr_dubber.asr import _audio_file_chunk_ranges, _parse_crispasr_payload
        from asmr_dubber.models import ProjectSettings

        analysis_path = workspace / "analysis.wav"
        sf.write(analysis_path, mono, rate, subtype="PCM_16")
        _, ranges = _audio_file_chunk_ranges(analysis_path, 25.0)
        paths = []
        for index, (a, b) in enumerate(ranges):
            path = workspace / f"asr_{index:04d}.wav"
            sf.write(path, mono[a:b], rate, subtype="PCM_16")
            paths.append(path)
        env = self.environment(workspace, "asr")
        cache = workspace / "asr-cache"
        cache.mkdir(exist_ok=True)
        # Reuse the existing punctuation weight without redownloading or altering it.
        punctuation = self.home / "cache/crispasr/fireredpunc-q4_k.gguf"
        if punctuation.is_file() and not (cache / punctuation.name).exists():
            try:
                os.link(punctuation, cache / punctuation.name)
            except OSError:
                import shutil

                shutil.copyfile(punctuation, cache / punctuation.name)
        env["CRISPASR_CACHE_DIR"] = str(cache)
        base = [
            str(self.crisp),
            "--backend",
            "parakeet",
            "--cache-dir",
            str(cache),
            "-m",
            str(self.asr_model),
            "-l",
            "ja",
            "-ojf",
            "-pp",
            "-t",
            "8",
        ]
        # Bound Windows command length; retain a single model load for short tests.
        for start in range(0, len(paths), 50):
            run_process(
                base + [str(p) for p in paths[start : start + 50]],
                cwd=workspace,
                log=workspace / f"asr_{start:04d}.log",
                env=env,
            )
        result = []
        settings = ProjectSettings(pause_split_seconds=0.7, max_sentence_seconds=8.0)
        for path, (a, _) in zip(paths, ranges, strict=True):
            output = path.with_suffix(".json")
            if not output.is_file():
                output = path.with_suffix(path.suffix + ".json")
            if not output.is_file():
                raise RuntimeError(f"Parakeet 没有写出时间戳文件：{output}")
            payload = json.loads(output.read_text(encoding="utf-8-sig"))
            if not payload.get("transcription"):
                continue
            sentences, _ = _parse_crispasr_payload(payload, settings)
            for sentence in sentences:
                result.append(
                    {
                        "id": f"s{len(result) + 1:04d}",
                        "start": sentence.start_seconds + a / rate,
                        "end": sentence.end_seconds + a / rate,
                        "ja": sentence.source_text,
                        "zh": "",
                    }
                )
        if not result:
            raise RuntimeError("Parakeet 未识别到带时间戳的日语，请先检查原音频。")
        return expand_ctc_boundaries(result, len(mono) / rate)

    def translate(self, segments: list[dict], checkpoint=None) -> None:
        from asmr_dubber.models import Sentence
        from asmr_dubber.translation import DeepSeekTranslator, TranslationChunk

        from translation_compat import TranslationClient, translate_batches

        if not self.key():
            raise RuntimeError("缺少 DeepSeek 密钥；在原项目中保存，或设置 DEEPSEEK_API_KEY。")
        if not any(not s.get("zh", "").strip() for s in segments):
            return
        with (
            TranslationClient(timeout=180) as client,
            DeepSeekTranslator(
                api_key=self.key(),
                model="deepseek-v4-flash",
                max_retries=1,
                timeout_seconds=180,
                client=client,
            ) as translator,
        ):
            translator.system_prompt += (
                "\n这是耳机音声的中文配音稿：保留原意、亲密程度、语气和称谓；使用简短自然口语，"
                "避免扩写，以接近日文的说话时长。不要额外添加动作、旁白或括号中的表演指令。"
                "\n输出 JSON 的 translations 数组，每项只允许 id 和 zh；不要返回 source 字段。"
            )

            def request_batch(items):
                batch = [
                    Sentence(
                        id=s["id"],
                        start_seconds=s["start"],
                        end_seconds=s["end"],
                        source_text=s["ja"],
                    )
                    for s in items
                ]
                return translator.translate_chunk(
                    TranslationChunk(sentences=batch), "[]", "[]", "asmr-spatial-lab"
                )

            translate_batches(segments, request_batch, checkpoint)

    def synthesize(self, tasks: list[dict], workspace: Path) -> None:
        batch = workspace / "tts_tasks.json"
        batch.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        env = self.environment(workspace, "tts")
        env.update(PYTHONPATH=str(self.runtime), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        run_process(
            [
                str(self.tts_python),
                str(ROOT / "tts_worker.py"),
                "--model-dir",
                str(self.runtime / "checkpoints"),
                "--tasks",
                str(batch),
            ],
            cwd=workspace,
            log=workspace / "tts.log",
            env=env,
            timeout=7200,
        )


def expand_ctc_boundaries(segments: list[dict], duration: float) -> list[dict]:
    """CTC token spans are emission spikes, not complete phoneme boundaries.

    Add bounded margins and merge very short interjections with a nearby phrase.
    This is a heuristic, explicitly not forced alignment; transcript remains editable.
    """
    merged: list[dict] = []
    for segment in segments:
        if (
            merged
            and merged[-1]["end"] - merged[-1]["start"] < 0.6
            and segment["start"] - merged[-1]["end"] < 1.2
            and segment["end"] - merged[-1]["start"] < 9
        ):
            merged[-1]["end"] = segment["end"]
            merged[-1]["ja"] += segment["ja"]
        else:
            merged.append(dict(segment))
    originals = [(s["start"], s["end"]) for s in merged]
    for i, segment in enumerate(merged):
        a, b = originals[i]
        left = (originals[i - 1][1] + a) / 2 if i else 0
        right = (b + originals[i + 1][0]) / 2 if i + 1 < len(merged) else duration
        segment.update(id=f"s{i + 1:04d}", start=max(left, a - 0.28), end=min(right, b + 0.35))
    return merged
