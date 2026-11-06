# Loader checks (format-test checkpoint)

Checkpoint: an untrained random-init draft written by `tools/make_format_test_ckpt.py` (release format).
Target for loader checks: the Qwen3-ASR-1.7B text decoder exported as `Qwen3ForCausalLM`
(`tools/extract_text_decoder.py`). Text prompt, no audio: these are format checks, not acceptance
measurements.

## Upstream vLLM v0.33.0 (`vllm/vllm-openai-cpu:v0.33.0-x86_64`, CPU) -- PASS

`eval/vllm_upstream_load_check.py`, K = 15:

- all 58 checkpoint tensors consumed by `DFlashQwen3Model.load_weights`; only `mask_embedding` stays
  zero (expected: no `mask_embedding.pt` shipped, so vLLM uses `embed_tokens[mask_token_id]`)
- resolved: `dflash_config` = {mask_token_id 151705, target_layer_ids [1, 7, 13, 19, 25]}, block_size 16,
  `rope_parameters.rope_theta` 1e6, rotary base used by the layer 1e6, 5 layers
- greedy spec output == plain output (lossless)

## vllm-neuron 0.29.0.1.2.0 (SDK 2.33 DLAMI venv, trn2.3xlarge, LNC=2, TP=1, max_num_seqs=1) -- loads; verify path not lossless (also for the public reference pair)

`eval/vllm_neuron_load_check.py`, K = 15, `on_device_sampling_config.all_greedy`:

- `DFlashConfig.from_configs` resolves: block 16, mask 151705, target_layer_ids [1, 7, 13, 19, 25],
  rope_theta 1e6, hidden 2048, inter 6144, 5 layers, 16 / 8 heads, vocab 151936, eps 1e-6
- runner: `Using DFlash auxiliary layers: (2, 8, 14, 20, 26)` (the +1 capture convention)
- draft compiles, warms up and runs
- greedy spec output != plain output, **and the same happens for the public reference pair**
  `Qwen/Qwen3-8B` + `z-lab/Qwen3-8B-DFlash-b16` under identical settings (spec output degenerates
  into one repeated token after a few steps). The issue is in the vllm-neuron DFlash serving path
  at this configuration (TP=1, trn2.3xlarge), not in the checkpoint format. vllm-neuron's
  documented validated DFlash setup is TP=8 on trn2.48xlarge with a Llama-3.1-8B target.
- CPU mode (`VLLM_NEURON_CPU_MODE=1`) cannot run it: the torch fallback for block-KV attention
  rejects GQA with a per-head block table.

Raw logs were kept with the run.

## Upstream vLLM v0.33.0 with the real target `Qwen/Qwen3-ASR-1.7B` on real audio (stage-1 200k draft) -- PASS

`eval/vllm_upstream_asr_check.py`, CPU image `vllm/vllm-openai-cpu:v0.33.0-x86_64`, K = 15, greedy,
18 held-out clips (6 LibriSpeech test-clean, 6 AMI ihm test, 6 MLS German test), max_num_seqs 1:

- upstream `Qwen3ASRForConditionalGeneration` implements the aux-hidden-state (EAGLE3) interface, so
  DFlash runs against the full multimodal target (no text-decoder export needed)
- resolved: mask 151705, target_layer_ids [1, 7, 13, 19, 25], block 16, RoPE 1e6
- **greedy spec output == plain output on 18 / 18 clips**
- acceptance (vLLM spec-decode metrics, mean accepted + 1 per draft step):
  - stock CPU image: **1.32**. vLLM's CPU path in `DFlashQwen3Model._normalize_context_k` runs one
    grouped `ops.rms_norm` over the stacked `[L, head_dim]` k_norm weights, and the CPU kernel applies
    only layer 0's weight to every layer, so context keys of draft layers 1-4 are mis-normalised
    (lossless, but most proposals are rejected).
  - with a per-layer k_norm loop (`--patch-knorm 1`, the intended computation): **4.70**
    (accepted per position: 130, 108, 82, 60, 50, 41, 34, 28, 24, 16, 11, 8, 8, 6, 5 of 165 drafts).
  - GPU builds use a different kernel path and were not tested here (GPU use is out of scope).

## vLLM CPU fix (vllm-project/vllm#66168)

Root cause of the 1.32: `csrc/cpu/layernorm.cpp` `rms_norm` ignored the rows of a 2D weight (always row 0),
while the CUDA kernel selects the row by the outermost input index; DFlash's fused context-key norm passes a
stacked `[L, head_dim]` weight. Fix: same 2D-weight semantics and shape checks in the CPU kernel; the existing
`tests/kernels/core/test_batched_weight_rms_norm.py` now runs on CPU (v0.33.0: 7 failed; fix: 10 passed, 3
skipped). Real audio, same 18 clips, stock model code, built from the PR branch with `docker/Dockerfile.cpu`:
**18/18 identical, tau 4.76** (accepted per position 133, 112, 84, 61, 54, 46, 32, 25, 18, 15, 11, 9, 8, 4, 1 of 163 drafts).
