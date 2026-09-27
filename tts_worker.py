"""Runs only inside the existing isolated IndexTTS-2.0 environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def active_duration(path: Path) -> float:
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, always_2d=True)
    level = np.max(np.abs(audio), axis=1)
    active = np.flatnonzero(level > max(float(level.max()) * 0.018, 1e-5))
    if not len(active):
        raise RuntimeError("IndexTTS 输出静音")
    return (active[-1] - active[0] + 1) / rate + 0.06


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from indextts.infer_v2 import IndexTTS2

    if not torch.cuda.is_available():
        raise RuntimeError("现有 IndexTTS-2.0 环境的 CUDA 不可用")
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    model = IndexTTS2(
        cfg_path=str(args.model_dir / "config.yaml"),
        model_dir=str(args.model_dir),
        device="cuda",
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
    )
    results = []
    for index, task in enumerate(tasks):
        torch.manual_seed(42 + index)
        np.random.seed(42 + index)
        final_output = Path(task["output"])
        pending_output = final_output.with_suffix(".pending.wav")
        kwargs = dict(
            spk_audio_prompt=task["voice"],
            text=task["text"],
            output_path=str(pending_output),
            emo_audio_prompt=task["emotion"],
            emo_alpha=0.6,
            use_random=False,
            interval_silence=80,
            verbose=False,
            max_text_tokens_per_segment=120,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.7,
            num_beams=3,
            repetition_penalty=10.0,
            max_mel_tokens=1500,
        )
        print(f"[{index + 1}/{len(tasks)}] {task['id']}: synthesize", flush=True)
        model.infer(**kwargs)
        duration = active_duration(pending_output)
        # IndexTTS 2.0 has no supported duration_factor; align with Rubber Band later.
        pending_output.replace(final_output)
        results.append(
            {
                "id": task["id"],
                "generated_seconds": duration,
                "tts_backend": "indextts2.0",
            }
        )
        print(f"Generated: {task['id']}", flush=True)
    args.tasks.with_name("tts_report.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
