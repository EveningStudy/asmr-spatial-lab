# ASMR Spatial Lab

English | [简体中文](README.zh-CN.md)

## Project status

Development is paused as of September 27, 2026. This repository is retained as an experimental record. In the owner's listening tests, RTF was preferred over the added HRTF and Meta baselines, but overall quality did not meet the original goal. This is subjective feedback, not a general benchmark. No further development is currently planned.

**New:** measured near-field HRTF and the official Meta 3-block neural renderer are available as additional baselines. Use `run --all-methods` or `compare --all-methods` for five-way comparisons. See [setup, usage, assumptions and licenses](SPATIAL_COMPARISON.md). Both use estimated, not ground-truth, positions; neither is a claim of improved listening quality.

A command-line experiment for converting Japanese binaural recordings into Chinese dubbing while attempting to preserve the speaker's voice and the original spatial impression.

**Subjective results are currently underwhelming.** This repository records experiments and implementation details; it is not a production dubbing tool or a claim of state-of-the-art spatial audio quality. Better numerical metrics do not necessarily mean greater similarity to the original recording.

## What this tests

The input is a single-speaker, two-channel Japanese recording, potentially including left/right positioning, movement, and changes in perceived distance. The experiments ask:

- Can the original binaural cues survive translation and voice-cloned Chinese synthesis?
- How much does spatial processing improve on centered dubbing?
- Does finer spectral and phase transfer sound more natural than simpler level/delay transfer?

No ground-truth 3D coordinates or head-tracking data are required. The prototype does not separate multiple speakers or complex sound effects. Its output contains only Chinese dubbing; the original Japanese speech is not mixed back in to create a spatial impression.

## How it works

```text
Japanese two-channel recording
  → Parakeet CTC 1.1B transcription and segmentation
  → DeepSeek V4 Flash translation
  → IndexTTS-2.0 voice cloning with phrase-level style references
  → Rubber Band pitch-preserving timing alignment
  → Spatial rendering: centered control / binaural cue transfer / complex RTF transfer
```

Audio processing runs locally. Only text is sent to DeepSeek for translation. All three spatial versions reuse the same synthesized Chinese speech, so TTS randomness is not mistaken for a difference between spatial algorithms.

The current backend is IndexTTS 2.0; earlier experiment records used 2.5. Cache keys distinguish the versions, so resuming an older run regenerates Chinese speech with 2.0 instead of silently reusing 2.5 clips. Existing cached clips remain on disk. The 2.0 adapter does not pass 2.5-only language or native duration parameters; timing adjustment uses Rubber Band. `compare` follows the backend recorded in the completed run.

### Baseline 1: Binaural cue transfer

Implementation: [`spatial.py`](spatial.py).

- Estimates interaural time differences (ITD) using GCC-PHAT cross-correlation, interpolation, confidence filtering, and smoothing.
- Estimates interaural level differences (ILD) across 28 frequency bands.
- Uses slow level changes and mild relative spectral changes as approximate distance cues.
- Transfers these cues to Chinese speech through frame-wise frequency-domain gains, phase delays, and overlap-add synthesis.

### Baseline 2: Time-varying complex relative transfer function (RTF) transfer

Implementation: [`spatial_rtf.py`](spatial_rtf.py).

- Compensates for bulk ITD before temporal averaging and estimates local spatial covariance between the two channels.
- Uses the principal eigenvector to estimate a complex relative transfer function, retaining finer differential spectra and frequency-dependent phase.
- Uses coherence and energy to assess confidence; low-confidence frequency bins fall back to baseline 1's cues.

Both methods use **digital signal processing (DSP)**, not pretrained neural binaural models. Neither reconstructs a complete HRTF, room impulse response, or physical distance.

### Comparison setup

The `compare` command renders cached Chinese speech through RTF, baseline 1, and the centered control. It matches whole-track joint-channel RMS levels, without normalizing the ears independently or rerunning ASR, translation, or TTS. This is level matching, not exact perceptual loudness matching.

## Experiments so far

- Short Japanese clips have completed the transcription, translation, voice cloning, and spatial rendering pipeline.
- Speech with synthetic left-to-right movement has been used for integration tests.
- Three versions have been rendered for a roughly 4-minute-24-second recording with 52 segments.
- Independent random source signals and known FIR filters have been used to test differential spectral and phase recovery. RTF achieved lower objective error for the more complex differential filtering cases.

These results establish that the pipeline runs and can transfer some cues. **They do not establish faithful reproduction of close-ear presence, performance, or spatial atmosphere.** Many frequency bins remain low-confidence on real material, and no systematic human listening evaluation has been established.

See [VALIDATION.md](VALIDATION.md) and [RTF_VALIDATION.md](RTF_VALIDATION.md) for experiment notes, currently in Chinese. Audio, cache, and output paths mentioned there refer to local experiments; those artifacts are not included in this public repository.

## Running the prototype

This is a Windows test program that depends on an existing environment, **not a standalone, ready-to-run distribution**.

Prepare the [ASMR Dubber](https://github.com/EveningStudy/asmr-dubber) source, main Python environment, Parakeet 1.1B/CrispASR installation, and isolated IndexTTS-2.0 environment and models in a neighboring `asmr-next` directory. The adapter reuses its ASR timestamp parsing, translation code, and local resources.

```text
Projects/
├─ asmr-next/
└─ asmr-spatial-lab/
```

The DeepSeek key is read from the process environment variable `DEEPSEEK_API_KEY` or the original project's local configuration. It is not copied into this project. Translation requests incur the applicable API charges.

```powershell
cd asmr-spatial-lab
.\run.ps1 doctor
.\run.ps1 run "D:\audio\sample.wav"
```

Start with a 15–60-second two-channel clip. Each run is limited to 10 minutes. Mono input can be tested with `--allow-mono`, but missing binaural information cannot be recovered.

```powershell
# Stop after translation, edit the zh fields in transcript.json, then resume
.\run.ps1 run "D:\audio\sample.wav" --out "outputs\test" --stop-after translate
.\run.ps1 resume "outputs\test"

# Create level-matched comparisons from an existing run in a new directory
.\run.ps1 compare "outputs\test"
```

Normal runs produce `zh_spatial.wav`, `zh_rtf.wav`, `zh_center.wav`, Chinese subtitles, and diagnostics. Comparison runs produce `zh_v1.wav`, `zh_rtf.wav`, and `zh_center.wav`. Audio is written as 48 kHz, 24-bit stereo; this does not imply recovery of all high-frequency detail in the source.

With another Python environment, install `requirements.txt` and run `python dub.py ...`. The ASMR Dubber resources above are still required; use `--asmr-root` for a non-default location.

```powershell
..\asmr-next\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\asmr-next\.venv\Scripts\python.exe benchmark_rtf.py
```

## Limitations

- Matching position does not mean matching breathiness, timbre, or performance. Spatial processing cannot repair differences in the synthesized voice itself.
- Distance is approximated through level and spectral changes; moving closer cannot reliably be distinguished from speaking louder.
- The methods do not recover centimeter-accurate distance, front/back or elevation coordinates, full room reverberation, or individualized ear responses.
- Chinese speech is placed within sentence windows, with spatial trajectories mapped by within-sentence progress. Words, movements, and breaths are not guaranteed to align precisely.
- There is no source separation. Output is silent outside recognized sentence windows, and original breaths and sound effects are not automatically retained.
- Strong reverberation, multiple speakers, independent signals at each ear, occlusion, and rapid movement can cause unreliable estimates or timbral distortion.
- BinauralGrad and BinauralFlow have not been implemented. Measured HRTF and Meta BinauralSpeechSynthesis are optional comparison backends, described separately above.

This public repository contains only source code, tests, and experiment documentation—not source audio, generated audio, model weights, API keys, or runtime caches. Use material for which you have the appropriate permissions, and do not present cloned output as an authentic recording of the person.
