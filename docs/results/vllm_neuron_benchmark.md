# vllm-neuron end-to-end benchmark (rc1 draft)

- Hardware: one trn2.3xlarge (1 Trainium2 device = 4 logical cores at LNC=2), public SDK 2.33 DLAMI,
  `aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0` venv (vllm 0.29.0) with the public vllm-neuron
  fork branch `jimburtoft/vllm-neuron@qwen3-asr/dflash` (eee4c7e; Qwen3-ASR as a DFlash target) on
  PYTHONPATH. Target `Qwen/Qwen3-ASR-1.7B` @ 7278e1e7, TP=4, encoder dp 4, bf16, greedy on device,
  `/v1/audio/transcriptions`, auto language ID. Draft: rc1 (s2b 16k), `num_speculative_tokens` 15.
  `encoder_cache_num_blocks` 2048 (the auto size filled up after a few requests on this branch).
- Real held-out audio (<= 30 s), 50 clips per set, one request at a time (BS=1) after 3 warm-up requests.
  Latency = client wall time per request. tau = 1 + accepted / drafts from the server's metrics.

## BS = 1

| set | clips | identical | WER plain | WER DFlash | mean ms plain | mean ms DFlash | speedup | p90 ms DFlash | tau |
|---|---|---|---|---|---|---|---|---|---|
| ami:ihm:test | 50 | 47 | 15.07 | 15.07 | 89.6 | 49.5 | 1.81x | 64.1 | 5.83 |
| common_voice_17_0:en:test | 50 | 46 | 7.55 | 7.55 | 86.7 | 49.3 | 1.76x | 65.0 | 5.64 |
| common_voice_17_0:zh-CN:test | 50 | 49 | 5.72 | 5.22 | 89.7 | 56.4 | 1.59x | 76.3 | 4.18 |
| fleurs:cmn_hans_cn:test | 50 | 49 | 9.55 | 9.60 | 141.9 | 85.5 | 1.66x | 114.3 | 3.68 |
| fleurs:en_us:test | 50 | 50 | 3.36 | 3.36 | 155.1 | 71.1 | 2.18x | 97.6 | 5.81 |
| fleurs:ja_jp:test | 50 | 48 | 4.81 | 4.93 | 179.4 | 139.8 | 1.28x | 211.0 | 2.40 |
| librispeech_asr:all:test.clean | 50 | 50 | 1.02 | 1.02 | 148.3 | 59.2 | 2.51x | 77.7 | 8.00 |
| librispeech_asr:all:test.other | 50 | 46 | 2.61 | 2.61 | 135.7 | 60.0 | 2.26x | 90.5 | 6.74 |
| multilingual_librispeech:french:test | 50 | 47 | 5.53 | 6.23 | 286.0 | 153.9 | 1.86x | 202.8 | 3.66 |
| multilingual_librispeech:german:test | 50 | 47 | 6.09 | 6.04 | 295.2 | 158.3 | 1.86x | 213.0 | 3.66 |
| multilingual_librispeech:spanish:test | 50 | 43 | 4.71 | 4.60 | 257.7 | 121.1 | 2.13x | 156.9 | 4.44 |
| voxpopuli:en:test | 50 | 46 | 6.46 | 6.63 | 144.8 | 63.3 | 2.29x | 93.6 | 6.64 |
| ALL | 600 | 568 | 5.66 | 5.74 | 167.5 | 88.9 | 1.88x | 158.3 | 4.23 |

568 / 600 transcripts are identical. The 32 differences are bf16 verify-graph numerics (a 16-token verify
pass vs 1-token decode, different reduction order), mostly one-word near-ties on hard audio (AMI, MLS);
WER over all 600 clips is 5.66 % plain vs 5.74 % DFlash (per set it moves both ways: zh-CN 5.72 -> 5.22,
MLS fr 5.53 -> 6.23). On upstream vLLM (CPU, fp32/bf16 kernels) the same draft was token-identical on
18/18 clips.

## Concurrency (LibriSpeech test-clean, 200 clips, `max_num_seqs` = C)

| C | plain RTFx | DFlash RTFx | gain | plain mean ms | DFlash mean ms | tau |
|---|---|---|---|---|---|---|
| 1 (50 clips) | -- | -- | 2.51x latency | 148.3 | 59.2 | 8.00 |
| 4 | 133.8 | 199.4 | 1.49x | 228.4 | 153.3 | 7.51 |
| 16 | 237.3 | 269.4 | 1.14x | 497.8 | 445.0 | 7.58 |

Spec decoding still helps at C=16 on this device but the gain shrinks with batch size (the verify
step costs ~16x the tokens of a decode step); at higher concurrency serve without the draft.
