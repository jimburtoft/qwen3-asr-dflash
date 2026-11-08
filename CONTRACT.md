# Qwen3-ASR-1.7B DFlash draft: release format contract

Status: **frozen** (v1).

This is everything a loader / serving integration needs to run the draft against
[`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B). The format is the public DFlash
checkpoint format (z-lab/dflash, vLLM `qwen3_dflash.py`, vllm-neuron `DFlashDraftModel`).

## Target

| | |
|---|---|
| target model | `Qwen/Qwen3-ASR-1.7B` (rev `7278e1e70fe206f11671096ffdd38061171dd6e5`); identical weights to `Qwen/Qwen3-ASR-1.7B-hf` (rev `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`), checked tensor by tensor |
| text decoder | Qwen3, 28 layers, hidden 2048, 16 q / 8 kv heads, head_dim 128, QK-norm, rope_theta 1e6, vocab 151936, tied embedding / LM head |
| target positions | 0..L-1 over the whole prompt (audio tokens included). The declared interleaved mRoPE gives every token the same index in all three sections, so 1-D RoPE is exact |
| prompt | public chat template: `<\|im_start\|>system\n{context}<\|im_end\|>\n<\|im_start\|>user\n<\|audio_start\|>` + `<\|audio_pad\|>` x N + `<\|audio_end\|><\|im_end\|>\n<\|im_start\|>assistant\n`; N = audio tokens (13 per full second / 100 mel frames). Default = auto language: the target itself writes `language X<asr_text>` then the transcript |
| stop tokens | `<\|im_end\|>` 151645, `<\|endoftext\|>` 151643 |

## Files

```
config.json          Qwen3 config + DFlash fields (below)
model.safetensors    bf16; 58 tensors, 272.7 M parameters
README.md            model card
```

- No `embed_tokens` and no `lm_head` in the checkpoint: the draft uses the **target's** token
  embedding and LM head (tied in Qwen3-ASR, one tensor `[151936, 2048]`).
- No `mask_embedding.pt` (the mask slot uses the target embedding row of `mask_token_id`; see below).
- No `d2t` / `draft_vocab_size`: the draft predicts over the full 151936 vocabulary.

Tensor names (draft-local layer indices 0..4):

```
fc.weight                               [2048, 10240]   (5 target features x 2048 -> 2048)
hidden_norm.weight                      [2048]
norm.weight                             [2048]
layers.{i}.input_layernorm.weight       [2048]
layers.{i}.post_attention_layernorm.weight [2048]
layers.{i}.self_attn.q_proj.weight      [2048, 2048]
layers.{i}.self_attn.k_proj.weight      [1024, 2048]
layers.{i}.self_attn.v_proj.weight      [1024, 2048]
layers.{i}.self_attn.o_proj.weight      [2048, 2048]
layers.{i}.self_attn.q_norm.weight      [128]
layers.{i}.self_attn.k_norm.weight      [128]
layers.{i}.mlp.gate_proj.weight         [6144, 2048]
layers.{i}.mlp.up_proj.weight           [6144, 2048]
layers.{i}.mlp.down_proj.weight         [2048, 6144]
```

## config.json fields that matter

| field | value | note |
|---|---|---|
| `architectures` | `["DFlashDraftModel"]` | |
| `model_type` | `qwen3` | |
| `block_size` | **16** | serve with `num_speculative_tokens = 15` (train and serve block sizes must match) |
| `dflash_config.mask_token_id` | **151705** | first embedding row past the tokenizer (last token is 151704 `<asr_text>`); never produced by tokenization. **Not** 151669: that is the z-lab Qwen3 drafts' mask id but is `<\|audio_start\|>` in Qwen3-ASR |
| `dflash_config.target_layer_ids` | **[1, 7, 13, 19, 25]** | HF output-hidden-state convention: feature for id `l` = output of decoder layer `l` = HF `hidden_states[l + 1]` (residual stream, before the final norm). z-lab rule for 28 layers. vllm-neuron requests capture layers `l + 1` = [2, 8, 14, 20, 26] |
| `num_target_layers` | 28 | |
| `hidden_size` / `intermediate_size` | 2048 / 6144 | |
| `num_hidden_layers` | 5 | |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 8 / 128 | |
| `rms_norm_eps` | 1e-6 | |
| `rope_theta` and `rope_parameters.rope_theta` | **1000000.0** (both written) | draft-owned RoPE; also the fallback default of vLLM / vllm-neuron, so no loader can silently resolve a different base |
| `vocab_size` | 151936 | |
| `tie_word_embeddings` | false | draft has no head of its own (shared with the target) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | 151643 / 151645 / 151643 | |

## Draft semantics (what the trainer optimises)

- Context: target features (5 x 2048, concatenated in `target_layer_ids` order) of every verified
  token **before** the anchor, projected by `hidden_norm(fc(.))` and injected as K/V into every draft
  layer (each layer's own `k_proj` / `k_norm` / `v_proj`), RoPE at the tokens' absolute positions.
  Prompt tokens (audio included) are context too.
- Block: `[anchor, mask x 15]` at positions `anchor_pos .. anchor_pos + 15`; embeddings from the target
  embedding table; bidirectional attention within the block plus all context.
- Output: `norm(.)` then the target LM head; slots 1..15 are the 15 proposals.

## Verified loaders (untrained format-test checkpoint)

| loader | result |
|---|---|
| this repo `model/release.py` | strict load, 58 tensors |
| upstream vLLM v0.33.0 (CPU image), target = Qwen3-ASR text decoder as Qwen3ForCausalLM | loads all 58 tensors; resolved block 16, mask 151705, target layers [1,7,13,19,25], RoPE base 1e6; greedy spec == plain |
| vllm-neuron 0.29.0.1.2.0 (trn2.3xlarge, TP=1), target = same text decoder | loads; resolved block 16, mask 151705, target layers [1,7,13,19,25], RoPE 1e6, aux capture layers (2, 8, 14, 20, 26). Greedy spec != plain, but the public reference pair Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16 shows the same at this config, so it is a serving-path issue, not a format issue (`docs/loader_checks.md`) |

## Checkpoints

| repo | contents |
|---|---|
| [`jburtoft/Qwen3-ASR-1.7B-DFlash`](https://huggingface.co/jburtoft/Qwen3-ASR-1.7B-DFlash) | the trained draft (this contract, v1) |

## Verified with the trained draft

| loader | result |
|---|---|
| upstream vLLM v0.33.0 (CPU image), target = `Qwen/Qwen3-ASR-1.7B` (multimodal, real audio) | greedy spec == plain on 18 / 18 clips (see `docs/loader_checks.md` for the CPU k_norm note) |
| vllm-neuron, branch `jimburtoft/vllm-neuron@qwen3-asr/dflash`, trn2.3xlarge TP=4 | loads and serves; 568 / 600 held-out transcripts identical to no-draft (argmax tie-breaking + bf16 rounding in the verify step, not the draft; with the tie-break fix 569 / 600; see `docs/results/vllm_neuron_benchmark.md`), served tau 4.23 |
