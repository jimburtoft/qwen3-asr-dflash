# SPDX-License-Identifier: Apache-2.0
"""End-to-end Qwen3-ASR-1.7B (+ optional DFlash) on vllm-neuron through the server's
``/v1/audio/transcriptions`` endpoint, real audio, one request at a time (BS=1) or ``--concurrency C``.

Starts ``vllm serve`` with the public vllm-neuron Qwen3-ASR settings (DFlash needs a vllm-neuron build
where Qwen3-ASR is a speculative target; see the README), sends each clip as a WAV file,
records wall latency, the transcript, and the server's spec-decode counters (Prometheus
``/metrics``) before/after, so tau (accepted tokens per draft + 1) is measured by the server.

Usage (vllm-neuron venv; needs ``pyarrow`` and ``requests``):
    python eval/vllm_neuron_serve_bench.py --clips "$Q3DF_CORPUS/shards/eval/*/*/shard-*.parquet" \
        --out run.json [--draft <draft repo or dir> --k 15] [--per-set 50] [--min-dur 0 --max-dur 30]
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import wave

import numpy as np
import pyarrow.parquet as pq
import requests

TARGET = "Qwen/Qwen3-ASR-1.7B"
REV = "7278e1e70fe206f11671096ffdd38061171dd6e5"


def wav_bytes(pcm16: bytes) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16)
    return b.getvalue()


def load_clips(pattern, per_set, min_dur, max_dur):
    by = {}
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f, columns=["clip_id", "dataset", "config", "split", "duration_s", "ref_text", "audio_pcm16"])
        for r in t.to_pylist():
            if min_dur <= r["duration_s"] <= max_dur:
                by.setdefault((r["dataset"], r["config"], r["split"]), []).append(r)
    out = []
    for k in sorted(by):
        rs = sorted(by[k], key=lambda r: r["clip_id"])
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(rs))[:per_set]
        out.extend(rs[i] for i in sorted(idx))
    return out


def metrics(base):
    txt = requests.get(base + "/metrics", timeout=30).text
    vals = {}
    for name in ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens",
                 "vllm:spec_decode_num_draft_tokens", "vllm:generation_tokens"):
        tot = 0.0
        for m in re.finditer(r"^" + re.escape(name) + r"(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", txt, re.M):
            tot += float(m.group(1))
        vals[name] = tot
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--per-set", type=int, default=50)
    ap.add_argument("--min-dur", type=float, default=0.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--draft", default="")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--enc-buckets", default="4")
    ap.add_argument("--enc-dp", type=int, default=0, help="encoder dp_size (default = tp)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--revision", default=REV)
    ap.add_argument("--draft-revision", default="")
    ap.add_argument("--extra-neuron-config", default="{}", help="JSON merged into neuron_config (e.g. debug_logits_dir)")
    ap.add_argument("--no-device-sampling", action="store_true", help="drop on_device_sampling_config (host sampling)")
    ap.add_argument("--clip-ids", default="", help="comma list: only these clips (sent in this order)")
    ap.add_argument("--logits-dir", default="", help="debug_logits_dir; dumps of request i are moved to <dir>/<i>/")
    ap.add_argument("--block-size", type=int, default=0, help="KV cache block size (0 = server default)")
    ap.add_argument("--max-tokens", type=int, default=0, help="max_completion_tokens for each request (0 = server default)")
    ap.add_argument("--enc-cache-blocks", type=int, default=2048,
                    help="encoder_cache_num_blocks (workaround: the auto size filled up after a few requests)")
    ap.add_argument("--port", type=int, default=8711)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log", default="")
    args = ap.parse_args()

    enc_b = [int(x) for x in args.enc_buckets.split(",")]
    seqb = sorted({1, args.max_num_seqs})
    add = {"neuron_config": {"quantization": "bf16", "num_batched_tokens_buckets": [512, args.max_model_len],
                             "num_seqs_buckets": seqb, "on_device_sampling_config": {"all_greedy": True}},
           "encoder_neuron_config": {"num_encoder_input_buckets": enc_b, "encoder_attention_block_size": 128,
                                     "dp_size": args.enc_dp or args.tp,
                                     "encoder_cache_num_blocks": args.enc_cache_blocks}}
    add["neuron_config"].update(json.loads(args.extra_neuron_config))
    if args.logits_dir:
        add["neuron_config"]["debug_logits_dir"] = args.logits_dir
    if args.no_device_sampling:
        add["neuron_config"].pop("on_device_sampling_config", None)
    cmd = ["vllm", "serve", TARGET, "--revision", args.revision,
           "--tensor-parallel-size", str(args.tp), "--dtype", "bfloat16", "--max-model-len", str(args.max_model_len),
           "--max-num-seqs", str(args.max_num_seqs), "--max-num-batched-tokens", str(args.max_model_len),
           "--limit-mm-per-prompt", json.dumps({"audio": 1}), "--no-enable-prefix-caching",
           "--mm-processor-cache-gb", "0", "--no-enable-log-requests", "--port", str(args.port),
           "--additional-config", json.dumps(add)]
    if args.block_size:
        cmd += ["--block-size", str(args.block_size)]
    if args.draft:
        sc = {"method": "dflash", "model": args.draft, "num_speculative_tokens": args.k}
        if args.draft_revision:
            sc["revision"] = args.draft_revision
        cmd += ["--speculative-config", json.dumps(sc)]
    # trn2.3xlarge has no EFA; the plugin's EFA CPU-affinity probe is an optimisation only.
    env = dict(os.environ, VLLM_RPC_TIMEOUT="100000", NEURON_SKIP_EFA_AFFINITY="1")
    log = open(args.server_log or (args.out + ".server.log"), "w")
    print("CMD " + " ".join(cmd), flush=True)
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = f"http://127.0.0.1:{args.port}"
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"server exited with {srv.returncode}; see log")
            try:
                if requests.get(base + "/health", timeout=5).status_code == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > 7200:
                raise TimeoutError("server start")
            time.sleep(10)
        startup_s = time.time() - t0
        clips = load_clips(args.clips, args.per_set, args.min_dur, args.max_dur)
        if args.clip_ids:
            order = args.clip_ids.split(",")
            byid = {c["clip_id"]: c for c in clips}
            clips = [byid[k] for k in order if k in byid]

        def transcribe(c):
            files = {"file": ("a.wav", wav_bytes(c["audio_pcm16"]), "audio/wav")}
            data = {"model": TARGET, "temperature": "0.0", "response_format": "json"}   # auto language ID
            if args.max_tokens:
                data["max_completion_tokens"] = str(args.max_tokens)
            t = time.perf_counter()
            r = requests.post(base + "/v1/audio/transcriptions", files=files, data=data, timeout=1800)
            dt = time.perf_counter() - t
            r.raise_for_status()
            return r.json().get("text", ""), dt

        for c in clips[: args.warmup]:
            transcribe(c)

        def move_dumps(i):
            if not args.logits_dir:
                return
            import glob as _g
            import shutil
            dst = os.path.join(args.logits_dir, str(i))
            os.makedirs(dst, exist_ok=True)
            for f in _g.glob(os.path.join(args.logits_dir, "step_*")):
                shutil.move(f, os.path.join(dst, os.path.basename(f)))
        move_dumps("warmup")
        m0 = metrics(base)
        res = []
        if args.concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor
            tw = time.perf_counter()
            with ThreadPoolExecutor(args.concurrency) as ex:
                outs = list(ex.map(transcribe, clips))
            wall = time.perf_counter() - tw
            m1 = metrics(base)
            lat = np.array([o[1] for o in outs])
            audio_s = sum(c["duration_s"] for c in clips)
            summ = {"n": len(clips), "concurrency": args.concurrency, "draft": args.draft, "wall_s": round(wall, 2),
                    "rtfx": round(audio_s / wall, 1), "mean_ms": round(1000 * lat.mean(), 1),
                    "p90_ms": round(1000 * float(np.percentile(lat, 90)), 1)}
            d = m1["vllm:spec_decode_num_drafts"] - m0["vllm:spec_decode_num_drafts"]
            if d > 0:
                summ["tau"] = round(1 + (m1["vllm:spec_decode_num_accepted_tokens"] - m0["vllm:spec_decode_num_accepted_tokens"]) / d, 4)
            print("SUMMARY " + json.dumps(summ), flush=True)
            json.dump({"summary": summ, "cmd": cmd, "texts": [o[0] for o in outs],
                       "clip_ids": [c["clip_id"] for c in clips]}, open(args.out, "w"))
            return
        for c in clips:
            mb = metrics(base)
            text, dt = transcribe(c)
            time.sleep(0.5 if args.logits_dir else 0)
            move_dumps(len(res))
            ma = metrics(base)
            res.append({"clip_id": c["clip_id"], "dataset": c["dataset"], "config": c["config"], "split": c["split"],
                        "duration_s": c["duration_s"], "latency_s": dt, "text": text, "ref_text": c["ref_text"],
                        "gen_tokens": ma["vllm:generation_tokens"] - mb["vllm:generation_tokens"],
                        "drafts": ma["vllm:spec_decode_num_drafts"] - mb["vllm:spec_decode_num_drafts"],
                        "accepted": ma["vllm:spec_decode_num_accepted_tokens"] - mb["vllm:spec_decode_num_accepted_tokens"]})
        m1 = metrics(base)
        lat = np.array([r["latency_s"] for r in res])
        summ = {"n": len(res), "draft": args.draft, "k": args.k, "startup_s": round(startup_s, 1),
                "mean_ms": round(1000 * lat.mean(), 1), "median_ms": round(1000 * float(np.median(lat)), 1),
                "p90_ms": round(1000 * float(np.percentile(lat, 90)), 1)}
        d = m1["vllm:spec_decode_num_drafts"] - m0["vllm:spec_decode_num_drafts"]
        if d > 0:
            a = m1["vllm:spec_decode_num_accepted_tokens"] - m0["vllm:spec_decode_num_accepted_tokens"]
            summ["accepted_per_draft"] = round(a / d, 4)
            summ["tau"] = round(1 + a / d, 4)
        print("SUMMARY " + json.dumps(summ), flush=True)
        json.dump({"summary": summ, "cmd": cmd, "clips": res}, open(args.out, "w"))
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=120)
        except Exception:
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    main()
