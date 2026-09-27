# BANC integration: blocked on pretrained weights

Checked: 2026-09-12. **BANC has not been integrated or run. No BANC listening sample has been produced.**

## Verified source contract

Official repository: https://github.com/anton-jeran/MULTI-AUDIODEC

Inspected revision: `662109213869e1aaf2e4f2f969aab4c748e1eb21`.

The single-speaker implementation encodes speech and binaural room-response latents separately. The official test pipeline decodes a mono speech waveform and a two-channel impulse response, then convolves them. Reusing the estimated response with Chinese speech is an experimental adaptation, not a demonstrated cross-lingual capability of the paper.

The official default inference pair is:

- `exp/autoencoder/symAD_vctk_48000_hop300/checkpoint-200000steps.pkl`
- `exp/vocoder/AudioDec_v1_symAD_vctk_48000_hop300_clean/checkpoint-500000steps.pkl`
- Matching `config.yml` files for both models.

The alternative single-speaker release uses an encoder at 500000 steps and a decoder at 380000 steps. These pairs must not be silently mixed.

## Download checks

| Official script | Google Drive file ID | Result |
|---|---|---|
| `Single_Multi_AudioDec/download_model.sh` | `10bQADjGMwar1BbXrxysznRVg8_F9HQUh` | HTTP 404 |
| `Single_Multi_AudioDec/download_model_500.sh` | `1Ln-Q8cvi_H_X7Xp06J4-lAsf0QSSBtww` | HTTP 404 |
| `Two_Multi_AudioDec/download_model.sh` | `1qB-ZTr37r1Vf5GweAhHr_2_9sRZZRb8m` | HTTP 404 |

The first two were also checked through their canonical Drive file-view URLs, which returned HTTP 404. Some requests had transient TLS failures; the final repeated download checks returned HTTP 404 for all three IDs. This establishes that the files were not publicly downloadable through those links from this environment at the time of checking; it does not establish whether they were deleted or made private.

Additional checks:

- Official repository tree contains model configurations but no pretrained checkpoint files.
- Official GitHub releases and issues lists were empty at check time.
- All six returned public forks retained the same default single-speaker download ID.
- Hugging Face searches for `BANC`, `M3-AUDIODEC`, and `MULTI-AUDIODEC` did not locate a matching model release.
- Related AV-RIR is a different model; its weights were not substituted or represented as BANC.

Earlier discussion confirmed that download scripts existed, not that the checkpoint downloads succeeded. That distinction matters: the previously suggested ready-to-download path is currently unverified/unavailable.

## Local readiness and boundary

- Existing GPU: RTX 5070 Ti Laptop, approximately 12 GB VRAM.
- Existing isolated IndexTTS runtime includes PyTorch 2.8.0+cu128, NumPy, SciPy, soundfile, and PyYAML.
- BANC peak GPU memory and real-ASMR quality remain **unverified** without trained weights.
- No GPU rental, model training, private-audio upload, driver change, or change to the existing model environments was performed.
- Existing IndexTTS 2.0 changes and the DSP/RTF baselines are preserved. No commit or push was made for this investigation.
- The upstream project states a noncommercial research license; code and weights must be used under their respective terms.

## What is needed to continue

A working authorized download of a matching BANC encoder/decoder pair and configurations, from the author or a verifiable mirror. No random or unrelated weights should be used to produce a misleading demonstration.

Once available: load only compatible state dictionaries, verify original-Japanese reconstruction first, measure memory on short single-item inference, and then compare Chinese convolution with existing baselines using the exact same TTS waveform. Implementing an untested wrapper without the weights is not equivalent to completing this integration.
