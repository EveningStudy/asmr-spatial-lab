# Five-way spatial comparison / 五方案试听

## Quick start

This is a local experimental comparison, not a claim that either new method is better.
Keep headphones and playback volume unchanged when comparing.

```powershell
cd D:\Dev\Projects\asmr-spatial-lab
# First-time resource setup (already completed on the current machine)
.\setup-spatial.ps1

# New Japanese audio: existing Parakeet + DeepSeek + IndexTTS 2.0, then five renderers
.\run.ps1 run "D:\audio\sample.wav" --all-methods

# Or reuse an existing completed task; no ASR, translation or TTS calls
.\run.ps1 compare "outputs\your-task" --all-methods
```

新音声运行第一条 `run --all-methods` 即可。最后打印的 `compare-rtf-*` 子目录包含五种同源配音结果。
目录名沿用旧版，不代表只有 RTF。已有结果用 `compare --all-methods`，不重复花翻译费用。
旧结果如果由 2.5 生成，对照仍用 2.5；新任务默认 2.0。报告记录实际 TTS 版本。

| Output | Method |
|---|---|
| `zh_center.wav` | Centered control / 居中对照 |
| `zh_v1.wav` | Original ITD/ILD cue transfer / 原双耳线索迁移 |
| `zh_rtf.wav` | Complex RTF / 复数相对传递函数 |
| `zh_hrtf.wav` | Measured near-field HRTF / 实测近场 HRTF |
| `zh_meta.wav` | Meta pretrained 3-block binaural neural renderer |

All outputs use identical aligned Chinese samples. Whole-track joint-channel RMS is matched, then one shared peak reduction is applied. No independent left/right normalization. RMS matching is not exact perceptual loudness matching.

## New methods

### Measured HRTF

Uses Aalto's MoRa mannequin near-field laser-spark HRIRs with low-frequency extension:
[dataset](https://zenodo.org/records/7316545), [paper](https://doi.org/10.1016/j.apacoust.2022.109173).
Creators: Márton Marschall, Javier Gómez Bolaños, Sebastian Prepelita, Ville Pulkki.
The SOFA file embeds **CC BY 4.0**, sampling rate 48 kHz, 196 measurements, 512-sample responses, zero additional Data.Delay.

Original binaural ITD and 16-band ILD are matched against measured responses, with temporal path regularization. Only the horizontal front hemisphere is considered: this avoids falsely claiming recovery of unknown elevation/front-back geometry. Distance is a bounded slow-level proxy (0.2–0.5 m), not measured distance. Three nearby filters are blended and applied with overlapping windowed convolution; a common calibration delay is removed equally in both ears. Smooth shared-ear level calibration follows the reference.

This is not a full personalized HRTF or a measured response of the original microphone. Sparse filter blending can color the voice. The 20 cm minimum measurement distance is not a few-centimeter ear-contact measurement. Position estimates can be wrong.

### Meta neural renderer

Official [BinauralSpeechSynthesis](https://github.com/facebookresearch/BinauralSpeechSynthesis), ICLR 2021, pretrained **3-block** release v1.1. Code is pinned to revision `81d0765ae590e8f69d3d26dc03f8a2e9c02b7d03`; download SHA-256 values are pinned in `spatial-assets.lock.json`.

It runs float32, batch 1, 1-second chunks with causal context. Only the final output head is calculated (verified equal to the upstream final output); unused intermediate heads are omitted. State dictionaries are loaded with `weights_only=True` and strict key matching. Upstream source is not modified.

The HRTF-derived approximate trajectory supplies positions at 120 Hz. SOFA positive-left azimuth is converted to Meta's positive-right y axis. Upstream mouth/ear marker offsets are accounted for. Speaker orientation is fixed to the identity quaternion; it is **not inferred**. Neural distance is clamped to at least 0.35 m; even that does not establish equivalence to the training distribution. No head tracking is available. The model can impose the acoustics of its training recordings rather than recreate the original room.

Meta code/license is kept under `work/spatial-assets/meta/LICENSE`; the upstream release is noncommercial (CC BY-NC 4.0). Use for personal/noncommercial evaluation, not as an unrestricted commercial dependency.

## Installation boundaries

- Approximately 36 MB of model/SOFA assets, plus source and a small h5py dependency.
- Existing IndexTTS PyTorch/CUDA environment is reused without installing packages into it.
- h5py 3.14.0 is installed with `--no-deps --target work/spatial-deps` for that Python interpreter.
- Assets, intermediate phrases and generated audio stay in ignored `work`/`outputs` directories.
- No GPU rental, model training, or source-audio upload. New-task translation still uses the configured DeepSeek API.
- `setup-spatial.ps1` assumes the default neighboring `asmr-next` layout and Windows uv runtime.
- Missing resources or failed integrity checks stop the new comparison; they are not silently replaced by another renderer.

## Validation

- Downloaded complete SOFA and Meta weights, and loaded both successfully on the local GPU.
- HRTF and Meta left/right sign checks pass on controlled audio.
- Optimized Meta final-head output equals upstream forward output on the same inputs.
- Multi-chunk Meta inference produces finite, correctly shaped audio.
- 2-sentence IndexTTS 2.0 and 15-second legacy-cache five-way comparisons completed.
- Short-phrase Meta peak allocated CUDA memory: approximately 778 MiB (not total process VRAM).
- Full 264.024-second / 52-phrase legacy IndexTTS 2.5 cache comparison completed. All five outputs are 48 kHz / 24-bit stereo, same length, finite, no digital clipping; joint-channel RMS matches within quantization precision. Meta peak allocated CUDA memory was approximately 912 MiB. New input uses IndexTTS 2.0; this older-cache test is not represented as a 2.0 test.
- No human listening quality judgment has been made. These tests establish operation, not superiority or faithful spatial recovery.

```powershell
..\asmr-next\.venv\Scripts\python.exe -m unittest discover -s tests -q
..\asmr-next\.asmr-dubber\runtimes\index-tts\.venv\Scripts\python.exe test_spatial_runtime.py
```

`spatial_tasks.report.json` contains estimated azimuth/distance ranges and measured GPU allocation. `comparison.json` records TTS provenance and gain factors. Model progress/errors are in `spatial_worker.log`. Individual phrase files are retained locally for diagnosis.
