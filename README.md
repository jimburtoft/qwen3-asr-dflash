# qwen3-asr-dflash

Training code for an openly licensed [DFlash](https://arxiv.org/abs/2602.06036) speculative-decoding
draft for [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), trained on AWS
Trainium2. The released draft (weights + model card) is on Hugging Face:
[`jburtoft/Qwen3-ASR-1.7B-DFlash`](https://huggingface.co/jburtoft/Qwen3-ASR-1.7B-DFlash); see
`MODEL_CARD.md` for results and usage.

All code is Apache-2.0 (`LICENSE`, `NOTICE`). It is adapted from the public
[`jimburtoft/voxtral-dflash`](https://github.com/jimburtoft/voxtral-dflash) training code. Only public
sources and openly licensed data are used; datasets, licenses, pinned revisions and filters are
documented in `data/LICENSES.md`. The release format is fixed in `CONTRACT.md`.

## Layout

| dir | contents |
|---|---|
| `model/` | `dflash.py` draft model, `config.py` release config, `release.py` save/load, `qwen3_asr_target.py` static-shape Qwen3-ASR target (audio encoder in 8-chunk windows + 28-layer Qwen3 decoder; batched left-padded decode; chunked prefill), `device.py` device selection / compile backend (Trainium, CUDA, CPU) |
| `train/` | `dflash_objective.py` (anchors, block mask, weighted loss, acceptance), `data.py` (label/audio loading, batch packing), queue trainer + feature producers (short and long-form), launch scripts, CPU reference trainer |
| `labels/` | target-generated labels (greedy transcripts of real audio, short and long-form) and the HF-equivalence checks |
| `data/` | corpus build (`prep/`), license / overlap audits (`scripts/`), filters, `LICENSES.md` |
| `eval/` | offline acceptance evaluator, upstream vLLM and vllm-neuron loader / serving checks, serving benchmark |
| `tests/` | CPU unit tests |
| `tools/` | text-decoder export (loader tests), format-test checkpoint writer |
| `docs/` | label WER tables, loader checks, acceptance and serving results |

## Environments

- **CPU** (model code, tests, HF-equivalence checks, corpus build): `pip install -r requirements-cpu.txt`,
  then `python -m pytest -q tests`.
- **Trainium2** (labels, features, training): a trn2.3xlarge launched from the public **AWS Deep
  Learning AMI Neuron, SDK 2.33** (multi-framework, or `Deep Learning AMI Neuron torch-neuronx 2.14.0`).
  Use its torch-neuronx venv as is and add the extra packages on top (does not touch torch):

  ```bash
  source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate   # torch 2.14 + torch-neuronx 2.14.3
  pip install -r requirements-neuron.txt
  ```

  Plain PyTorch (`torch.device("neuron")`, `torch.compile(backend="neuron")`); no container, no custom
  build. One trn2.3xlarge (4 logical cores at LNC=2) was used for everything. Add swap before compiling.
  Every script takes `--device auto|neuron|cuda|cpu`.

Paths: `$Q3DF_CORPUS` (corpus, default `~/q3df_corpus`) and `$Q3DF_WORK` (labels, eval items, runs,
default `~/q3df_work`); see `train/_env.sh`.

## Reproducing the release

Times are for one trn2.3xlarge unless noted.

### 1. Corpus (CPU, ~5 h on 32 vCPU, ~75 GB)

```bash
cd data/prep
WORKERS=8 bash build_all.sh                  # every eval set first, then every train set (resumable)
python manifest_and_sanity.py                # manifests + sanity checks
python ../scripts/eval_speaker_exclude.py    # cross-language eval-speaker exclusion list
python ami_longform.py --n 60                # eval: 30 s - 5 min AMI test windows
python build_longform.py --workers 4         # long-form TRAIN set from the train shards only
python longform_sanity.py
```

Train: 281,801 clips / 700.9 h (1-30 s; English 66 %, 30 other language sets 34 %), of which 8,676
are excluded because their speaker appears in an eval set of another language. Eval: 34,248 clips in
80 held-out sets. Long-form train: 4,299 clips / 178.6 h (31-300 s).

### 2. Labels (~2 h short + ~5 h long-form)

The target's own greedy transcripts of the real audio, with the public chat template and automatic
language identification (what a server does by default):

```bash
sh labels/run_labels_all.sh                       # short eval + train, one process per core
for r in 0 1 2 3; do NEURON_RT_VISIBLE_CORES=$r python labels/generate_labels_longform.py \
  --root $Q3DF_CORPUS/shards/train_longform --out $Q3DF_WORK/labels/train_longform --rank $r --world 4 & done; wait
python labels/generate_labels_longform.py --root $Q3DF_CORPUS/shards/eval/ami_longform --out $Q3DF_WORK/labels/eval_longform
python labels/label_wer.py "$Q3DF_WORK/labels/train/*.labels.parquet"     # sanity
```

`labels/check_static_target_cpu.py` and `labels/check_batched_cpu.py` check the static target against
HF transformers on CPU (prompt ids, features, logits and greedy tokens identical in fp32).

### 3. Held-out eval items (~30 min)

```bash
for r in 0 1 2 3; do NEURON_RT_VISIBLE_CORES=$r python train/feature_producer.py \
  --labels "$Q3DF_WORK/labels/eval/*.labels.parquet" --audio-root $Q3DF_CORPUS/shards --rank $r --world 4 \
  --batch 4 --eval-out $Q3DF_WORK/eval_batches/b4_per16 --eval-per-set 16 & done; wait
python train/feature_producer_longform.py --labels $Q3DF_WORK/labels/eval_longform \
  --audio-root $Q3DF_CORPUS/shards/eval/ami_longform --queue /dev/shm/unused \
  --eval-out $Q3DF_WORK/eval_batches/longform_eval --eval-n 24 --n-blocks 64
```

(`--eval-per-set 64` into `b4_per64` for the final offline numbers.)

### 4. Stage 1: short clips (~9 h)

Cores 0-2 run frozen-target feature producers, core 3 trains the draft:

```bash
EVAL=$Q3DF_WORK/eval_batches/b4_per16 sh train/launch_queue_run.sh full_s1 200000 \
  --compute bf16 --lr 6e-4 --warmup-frac 0.04 --eval-every 10000 --save-every 10000 --log-every 500
```

Batch = 4 clips x 16 random anchor blocks of 16 tokens, paper loss weights (gamma 7), AdamW
(0.9, 0.95), cosine, clip 1.0, bf16 autocast with fp32 master weights, random init (copying target
decoder layers into the draft was worse in a pilot). 0.155 s/step.

### 5. Stage 2: long-form adaptation (~7 h)

```bash
N_SHORT=1 N_LONG=2 LONG_EVERY=2 sh train/launch_stage2.sh s2a 20000 $Q3DF_WORK/runs/full_s1/step0200000/train_state.pt \
  --compute bf16 --lr 2e-4 --warmup-frac 0.02 --eval-every 2000 --save-every 2000
N_SHORT=1 N_LONG=2 LONG_EVERY=2 sh train/launch_stage2.sh s2b 16000 $Q3DF_WORK/runs/s2a/step0020000/train_state.pt \
  --compute bf16 --lr 1e-4 --warmup-frac 0.01 --eval-every 2000 --save-every 2000
```

The release is `s2b/step0016000/release/`. Without stage 2 the draft gets tau 1.17 on long-form audio.

### 6. Checks

```bash
python eval/acceptance_eval.py --ckpt $Q3DF_WORK/runs/s2b/step0016000/train_state.pt \
  --eval-dirs $Q3DF_WORK/eval_batches/b4_per64
python eval/vllm_upstream_asr_check.py --help       # upstream vLLM (CPU image), real audio, lossless check
python eval/vllm_neuron_serve_bench.py --help       # end-to-end serving on vllm-neuron
python train/parity_draft_step.py --item <one queue item .pt>   # accelerator vs CPU fp32 draft step
```

## Results (summary; details in `MODEL_CARD.md` and `docs/`)

- Offline held-out tau (block 16, 64 clips per set): English 6.89 (LibriSpeech test-clean 7.57),
  80 short sets 2.95, AMI long-form 4.23.
- vllm-neuron, trn2.3xlarge TP=4, BS=1, 600 held-out clips: mean latency 167.5 -> 88.9 ms (1.88x),
  served tau 4.23; LibriSpeech test-clean 2.51x.
- Upstream vLLM v0.33 (CPU image): loads with the real Qwen3-ASR target, greedy output identical on
  18/18 real-audio clips.

## Notes

- `CONTRACT.md` lists every format value (mask token 151705 -- not 151669, which is `<|audio_start|>`
  in Qwen3-ASR; target layers [1, 7, 13, 19, 25]; block 16; RoPE 1e6; full vocabulary; no
  `mask_embedding.pt`).
- Static-shape design choices: encoder and decoder as separate graphs; causal attention with right
  padding (training features) or left padding with per-row RoPE shift (batched greedy labels) instead of
  a key-padding mask; chunked prefill so one graph serves every long-form prompt length.
