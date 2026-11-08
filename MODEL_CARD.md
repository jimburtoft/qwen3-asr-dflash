---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-1.7B
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - speculative-decoding
  - dflash
  - qwen3-asr
  - draft-model
  - trainium
language:
  - en
  - de
  - fr
  - es
  - it
  - pt
  - nl
  - pl
  - zh
  - ja
  - ru
  - multilingual
datasets:
  - openslr/librispeech_asr
  - mythicinfinity/libritts_r
  - parler-tts/mls_eng
  - facebook/multilingual_librispeech
  - fsicoli/common_voice_17_0
  - facebook/voxpopuli
  - MLCommons/peoples_speech
  - edinburghcstr/ami
---

# Qwen3-ASR-1.7B DFlash draft

A [DFlash](https://arxiv.org/abs/2602.06036) block-diffusion speculative-decoding draft for
[`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B). It proposes 15 tokens per step in
one forward pass and is verified by Qwen3-ASR under greedy decoding. Trained on AWS Trainium2 with
openly licensed speech data only. Training code:
[github.com/jimburtoft/qwen3-asr-dflash](https://github.com/jimburtoft/qwen3-asr-dflash).

- **Format:** the public DFlash checkpoint format (`DFlashDraftModel`, Qwen3 layers), as read by
  vLLM's `qwen3_dflash.py`, vllm-neuron's DFlash drafter and the `z-lab/dflash` reference code.
- **Size:** 5 draft layers, hidden 2048, 272.7M parameters (BF16, 545 MB). Token embeddings and LM
  head are **not** shipped: the draft uses Qwen3-ASR's own (tied) embedding / LM head.
- **Block size 16** (`num_speculative_tokens = 15`).
- **Target features:** outputs of Qwen3-ASR text-decoder layers `[1, 7, 13, 19, 25]`
  (HF `hidden_states[l + 1]`; servers capture layers 2, 8, 14, 20, 26).
- **Mask token:** id 151705 (first embedding row past the tokenizer). Not 151669, which public Qwen3
  DFlash drafts use but which is `<|audio_start|>` in Qwen3-ASR.
- **RoPE:** draft-owned, theta 1e6 (written as `rope_theta` and `rope_parameters`). Full vocabulary,
  no `d2t`, no `mask_embedding.pt`.

## Results

Real audio, greedy decoding, held-out sets never used in training (eval speakers / meetings removed
from training, also across languages). tau = mean accepted tokens per draft step including the
target's bonus token.

### Acceptance (offline, reproducible with the released code)

64 held-out clips per set (<= 30 s; AMI long-form: 24 windows of 31-300 s), an anchor at **every**
response position, same greedy accept rule as DFlash verification in vLLM. "release" is this
checkpoint; "stage 1" is the short-clip checkpoint before long-form adaptation.

| held-out set | release (s2b 16k) | stage 1 (200k) |
| --- | --- | --- |
| **English (6 sets)** | **6.89** | 7.15 |
| **all 80 short sets** | **2.95** | 3.05 |
| **AMI long-form (31-300 s)** | **4.23** | 1.17 |
| LibriSpeech test.clean | 7.57 | 7.82 |
| VoxPopuli en test | 7.18 | 7.47 |
| LibriSpeech test.other | 7.00 | 7.35 |
| FLEURS en_us test | 6.79 | 7.12 |
| AMI ihm test | 6.20 | 6.40 |
| Common Voice 17 en test | 5.73 | 5.77 |
| VoxPopuli es test | 4.67 | 4.89 |
| MLS spanish test | 4.43 | 4.65 |
| FLEURS es_419 test | 4.35 | 4.50 |
| Common Voice 17 es test | 4.33 | 4.46 |
| VoxPopuli fr test | 4.31 | 4.41 |
| VoxPopuli de test | 4.07 | 4.21 |
| Common Voice 17 nl test | 4.06 | 4.23 |
| MLS german test | 3.99 | 4.17 |
| Common Voice 17 zh-CN test | 3.96 | 4.10 |
| VoxPopuli nl test | 3.92 | 4.06 |
| MLS french test | 3.91 | 4.01 |
| VoxPopuli it test | 3.89 | 4.09 |
| FLEURS cmn_hans_cn test | 3.86 | 3.99 |
| FLEURS it_it test | 3.81 | 3.99 |
| Common Voice 17 fr test | 3.78 | 3.93 |
| FLEURS de_de test | 3.75 | 3.89 |
| FLEURS fr_fr test | 3.73 | 3.84 |
| Common Voice 17 it test | 3.72 | 3.92 |
| MLS dutch test | 3.67 | 3.82 |
| Common Voice 17 pt test | 3.64 | 3.74 |
| Common Voice 17 de test | 3.61 | 3.73 |
| Common Voice 17 th test | 3.61 | 3.76 |
| MLS italian test | 3.48 | 3.66 |
| FLEURS pt_br test | 3.37 | 3.48 |
| Common Voice 17 pl test | 3.27 | 3.40 |
| FLEURS nl_nl test | 3.26 | 3.38 |
| MLS portuguese test | 3.24 | 3.39 |
| Common Voice 17 ru test | 3.20 | 3.26 |
| VoxPopuli pl test | 3.15 | 3.31 |
| MLS polish test | 2.96 | 3.09 |
| Common Voice 17 tr test | 2.96 | 3.04 |
| FLEURS th_th test | 2.90 | 3.00 |
| Common Voice 17 sv-SE test | 2.83 | 2.92 |
| Common Voice 17 cs test | 2.80 | 2.95 |
| VoxPopuli cs test | 2.79 | 2.89 |
| Common Voice 17 ja test | 2.78 | 2.87 |
| Common Voice 17 ar test | 2.67 | 2.79 |
| FLEURS ru_ru test | 2.66 | 2.75 |
| Common Voice 17 hu test | 2.60 | 2.73 |
| Common Voice 17 fa test | 2.58 | 2.61 |
| FLEURS pl_pl test | 2.58 | 2.66 |
| VoxPopuli hu test | 2.57 | 2.70 |
| Common Voice 17 fi test | 2.56 | 2.65 |
| FLEURS fa_ir test | 2.53 | 2.61 |
| Common Voice 17 ro test | 2.46 | 2.59 |
| FLEURS sv_se test | 2.45 | 2.48 |
| FLEURS tr_tr test | 2.42 | 2.54 |
| FLEURS cs_cz test | 2.41 | 2.49 |
| FLEURS ja_jp test | 2.39 | 2.51 |
| VoxPopuli fi test | 2.35 | 2.42 |
| FLEURS ar_eg test | 2.32 | 2.40 |
| VoxPopuli ro test | 2.32 | 2.42 |
| FLEURS ro_ro test | 2.31 | 2.39 |
| FLEURS fi_fi test | 2.25 | 2.33 |
| FLEURS hu_hu test | 2.19 | 2.26 |
| FLEURS yue_hant_hk test | 2.11 | 2.16 |
| Common Voice 17 yue test | 2.11 | 2.12 |
| Common Voice 17 zh-HK test | 2.02 | 2.06 |
| Common Voice 17 id test | 1.99 | 2.06 |
| FLEURS id_id test | 1.95 | 1.97 |
| Common Voice 17 da test | 1.82 | 1.83 |
| Common Voice 17 mk test | 1.81 | 1.84 |
| FLEURS mk_mk test | 1.73 | 1.76 |
| FLEURS ms_my test | 1.72 | 1.78 |
| Common Voice 17 el test | 1.66 | 1.68 |
| Common Voice 17 hi test | 1.65 | 1.77 |
| FLEURS hi_in test | 1.61 | 1.66 |
| FLEURS da_dk test | 1.60 | 1.62 |
| FLEURS fil_ph test | 1.54 | 1.57 |
| FLEURS el_gr test | 1.48 | 1.49 |
| Common Voice 17 vi test | 1.42 | 1.43 |
| FLEURS vi_vn test | 1.40 | 1.42 |
| FLEURS ko_kr test | 1.19 | 1.20 |
| Common Voice 17 ko test | 1.17 | 1.18 |

Acceptance follows the training data: English is highest; languages with tens of hours of training
audio (es, fr, de, it, nl, zh-CN, th) reach roughly 2.9-4.7 depending on the set; languages present only in the evaluation sets
(ko, vi, el, hi, da, id, ms, fil, mk, yue) stay at 1.2-2.1.

### End-to-end serving (vllm-neuron, Trainium2)

trn2.3xlarge (one Trainium2 device, 4 logical cores at LNC=2), public Neuron SDK 2.33 DLAMI vLLM venv
(vllm 0.29.0) with the public vllm-neuron branch
[`jimburtoft/vllm-neuron@qwen3-asr/dflash`](https://github.com/jimburtoft/vllm-neuron/tree/qwen3-asr/dflash)
(commit `eee4c7e`; makes Qwen3-ASR a DFlash target; not yet in a vllm-neuron release). TP=4, bf16,
greedy on device, `/v1/audio/transcriptions` with automatic language ID, one request at a time, 50
held-out clips per set:

| set | clips | identical | WER no draft | WER draft | mean ms no draft | mean ms draft | speedup | p90 ms draft | tau |
|---|---|---|---|---|---|---|---|---|---|
| AMI ihm test | 50 | 47 | 15.07 | 15.07 | 89.6 | 49.5 | 1.81x | 64.1 | 5.83 |
| Common Voice 17 en test | 50 | 46 | 7.55 | 7.55 | 86.7 | 49.3 | 1.76x | 65.0 | 5.64 |
| Common Voice 17 zh-CN test | 50 | 49 | 5.72 | 5.22 | 89.7 | 56.4 | 1.59x | 76.3 | 4.18 |
| FLEURS cmn_hans_cn test | 50 | 49 | 9.55 | 9.60 | 141.9 | 85.5 | 1.66x | 114.3 | 3.68 |
| FLEURS en_us test | 50 | 50 | 3.36 | 3.36 | 155.1 | 71.1 | 2.18x | 97.6 | 5.81 |
| FLEURS ja_jp test | 50 | 48 | 4.81 | 4.93 | 179.4 | 139.8 | 1.28x | 211.0 | 2.40 |
| LibriSpeech test.clean | 50 | 50 | 1.02 | 1.02 | 148.3 | 59.2 | 2.51x | 77.7 | 8.00 |
| LibriSpeech test.other | 50 | 46 | 2.61 | 2.61 | 135.7 | 60.0 | 2.26x | 90.5 | 6.74 |
| MLS french test | 50 | 47 | 5.53 | 6.23 | 286.0 | 153.9 | 1.86x | 202.8 | 3.66 |
| MLS german test | 50 | 47 | 6.09 | 6.04 | 295.2 | 158.3 | 1.86x | 213.0 | 3.66 |
| MLS spanish test | 50 | 43 | 4.71 | 4.60 | 257.7 | 121.1 | 2.13x | 156.9 | 4.44 |
| VoxPopuli en test | 50 | 46 | 6.46 | 6.63 | 144.8 | 63.3 | 2.29x | 93.6 | 6.64 |
| ALL | 600 | 568 | 5.66 | 5.74 | 167.5 | 88.9 | 1.88x | 158.3 | 4.23 |


568 / 600 transcripts are identical to the no-draft run. Two causes, neither in the draft:

- **Argmax tie-breaking (fixed on the serving branch).** vllm-neuron's on-device greedy argmax picks among
  exactly equal bf16 logits differently for a 1-token decode step and a 16-token verify step. With the fix
  (included in `jimburtoft/vllm-neuron@qwen3-asr/dflash`), a draft that is never accepted gives 600 / 600
  identical transcripts (14 differed before).
- **bf16 rounding of accepted positions (expected).** Positions computed inside a multi-token verify step after
  accepted draft tokens differ from 1-token decode by about one bf16 step (0.25-0.44 logits); when the top two
  tokens are that close the choice can flip. With the fix: 569 / 600 identical, WER 5.719 % without the draft
  vs 5.731 % with it, tau 4.25, mean latency 169.7 -> 88.6 ms.

WER uses a lowercase / punctuation normaliser (CER for Chinese and Japanese).

At TP=1 on the same branch: mean latency 335.5 -> 156.6 ms (2.14x), tau 4.26, 565 / 600 identical.
(vllm-neuron 0.29.0.1.2.0 as released is not lossless with DFlash at TP=1 -- also for the public
Qwen3-8B + z-lab DFlash pair -- because of decode-mask layout issues fixed on the branch above.)

Concurrency (LibriSpeech test-clean, 200 clips, `--max-num-seqs C`): real-time factor 133.8x -> 199.4x
at C = 4 (1.49x), 237.3x -> 269.4x at C = 16 (1.14x). The benefit shrinks as the batch grows.

## Usage

### vLLM (DFlash speculative decoding)

```bash
vllm serve Qwen/Qwen3-ASR-1.7B \
  --speculative-config '{"method": "dflash", "model": "jburtoft/Qwen3-ASR-1.7B-DFlash", "num_speculative_tokens": 15}'
```

- **Upstream vLLM (>= 0.33):** `Qwen3ASRForConditionalGeneration` exposes the auxiliary hidden states
  DFlash needs. Checked with the `vllm/vllm-openai-cpu:v0.33.0` image on 18 real-audio clips: greedy
  output identical with and without the draft. On the **CPU** platform the acceptance is low (1.3)
  because vLLM's CPU `rms_norm` kernel ignores the per-layer rows of DFlash's grouped context-key norm;
  fixed in [vllm-project/vllm#66168](https://github.com/vllm-project/vllm/pull/66168) (with the fix:
  4.76, still lossless). GPU builds were not tested.
- **Trainium (vllm-neuron):** use the branch above (TP=4 shown; add
  `"encoder_neuron_config": {"encoder_cache_num_blocks": 2048, ...}` to `--additional-config` on that
  branch, see the code repository's `eval/vllm_neuron_serve_bench.py` for the full command).

### Reference implementation (PyTorch)

The code repository contains `model/dflash.py` (the draft), `model/release.py` (load / save) and
`eval/acceptance_eval.py`. The format matches the public `z-lab/dflash` reference modelling code
(unit test in the repository).

## Training

Full reproduction steps are in the [code repository](https://github.com/jimburtoft/qwen3-asr-dflash).

- **Data:** 273k clips (~680 h, 1-30 s) after removing 8,676 clips whose speaker appears in any
  evaluation set: English 66 % (LibriSpeech, LibriTTS-R, MLS en, Common Voice 17 en, VoxPopuli en, The
  People's Speech clean with CC BY / public-domain sources only, AMI ihm) and 34 % other languages
  (MLS de / fr / es / it / pt / nl / pl; VoxPopuli de / fr / es / it / nl / pl / ro / hu / cs / fi;
  Common Voice 17 in 18 languages). Plus 4,299 long-form clips (178.6 h, 31-300 s, concatenated
  consecutive training utterances). FLEURS: evaluation only.
- **Labels:** Qwen3-ASR's own greedy outputs for the real audio (`language X<asr_text>...`, automatic
  language ID, as served), and its hidden states at the five conditioning layers computed on the fly.
- **Objective (paper recipe):** random anchors in the response, block of 16 = anchor + 15 masks,
  bidirectional within the block and to target features before the anchor, position weights
  exp(-(k-1)/7).
- **Schedule:** stage 1, 200k steps of 4 clips x 16 blocks (AdamW 6e-4 cosine, bf16 autocast, fp32
  master weights, random init); stage 2 long-form adaptation, 20k + 16k steps (lr 2e-4, then 1e-4),
  one long-form item every second step. Stage 2 raised long-form tau from 1.17 to 4.23 at a cost of
  0.26 on short English clips.
- **Compute:** one AWS Trainium2 trn2.3xlarge with PyTorch on Neuron (public SDK 2.33 DLAMI
  torch-neuronx venv): ~2 h short labels, ~5 h long-form labels, ~9 h stage 1, ~7 h stage 2.

## Limitations

- Trained for transcription with the public chat template; context / hotword system prompts and other
  tasks were not trained.
- Low acceptance for languages without training data (see above) and for Cantonese and Korean.
- Block size 16 only.

## License and attribution

Apache-2.0 (weights and code). The draft is derived from the outputs and hidden states of
Qwen3-ASR-1.7B (Apache-2.0, Qwen team, Alibaba) and trained on the datasets listed above; see
`NOTICE` for the attributions and `ATTRIBUTION_peoples_speech.csv` for The People's Speech sources.
Users must not attempt to identify speakers in Common Voice, LibriSpeech or Multilingual LibriSpeech.
No audio or speaker identifiers are distributed with this model.

## Citation

```bibtex
@article{chen2026dflash,
  title   = {DFlash: Block Diffusion for Flash Speculative Decoding},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
