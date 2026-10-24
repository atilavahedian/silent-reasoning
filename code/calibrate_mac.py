"""Phase 0 calibration: GPT-2 124M training throughput and memory per precision, plus a Coconut-forward parity check
(accelerator fp32 vs CPU float64) on a tiny random model. Trains nothing; downloads nothing.

    PYTHONPATH=common python3 code/calibrate_mac.py            # Mac, 20 steps
    PYTHONPATH=common python3 code/calibrate_mac.py --steps 2  # CPU smoke
    PYTHONPATH=common python3 code/calibrate_mac.py --real     # D4 parity

Writes results/calibration_<device>.json in the experiment folder (or --out).

--real (registration D4) loads openai-community/gpt2 (the one mode that may download), builds one real ProsQA batch
from data/prosqa_valid.json with latent.data (coconut arm, 16 examples, pad_latent_to_max, stages 3 and 6, i.e. 3
and 6 latents per example), runs one coconut-mode forward on the accelerator in fp32 and on the CPU in float64, and
reports the max abs and max relative error of the logits (attended positions) and of the loss. It writes
results/parity_real_<device>.json and skips the throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "..", "..", "common"))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from latent.data import Collator, add_latent_tokens, build_stage_dataset, load_examples, tokenize_examples  # noqa
from latent.model import LatentLM, load_base  # noqa: E402
from latent.train import autocast_ctx, resolve_device  # noqa: E402

GPT2_SHAPE = dict(n_layer=12, n_embd=768, n_head=12, vocab_size=50260, n_positions=1024)
SHAPES = {"gpt2": GPT2_SHAPE,                                                   # 124M
          "gpt2-medium": dict(n_layer=24, n_embd=1024, n_head=16, vocab_size=50260, n_positions=1024)}   # 355M (7.7)


def build_gpt2(shape: str = "gpt2"):
    """Mirrors run.py AutoModelForCausalLM.from_pretrained("openai-community/<shape>") from the local HF cache only;
    falls back to a random GPT2Config of identical shape (vocab 50260 = 50257 + the three latent tokens). shape
    "gpt2-medium" is the 355M plain-depth control's memory probe (A13.2 (b))."""
    cfg = SHAPES[shape]
    try:
        m = GPT2LMHeadModel.from_pretrained(f"openai-community/{shape}", local_files_only=True)
        m.resize_token_embeddings(cfg["vocab_size"])
        return m, f"openai-community/{shape} (local cache)"
    except Exception:
        torch.manual_seed(0)
        return GPT2LMHeadModel(GPT2Config(**cfg)), f"random GPT2Config(n_layer={cfg['n_layer']}, n_embd={cfg['n_embd']}, n_head={cfg['n_head']})"


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _reset_peak(device):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device):
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def _latent_batch(batch, seq, n_latent, vocab, latent_id, device, gen):
    ids = torch.randint(0, vocab - 3, (batch, seq), generator=gen)
    if n_latent:
        start = int(seq * 0.75) - n_latent
        ids[:, start:start + n_latent] = latent_id
    return ids.to(device)


def bench(model, device, precision, steps, batch, seq, n_latent=0, warmup=1):
    """Ours. Times `steps` forward+backward+AdamW steps of LatentLM (mode coconut; n_latent latents per row inside
    the sequence, 0 = plain causal LM) and returns tokens/s and peak memory."""
    latent_id = model.config.vocab_size - 1
    lm = LatentLM(model, latent_id, latent_id - 2, latent_id - 1, 50256).to(device).train()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4, weight_decay=0.01)
    gen = torch.Generator().manual_seed(0)
    am = torch.ones(batch, seq, dtype=torch.long, device=device)
    pos = torch.arange(seq, device=device).unsqueeze(0).expand(batch, seq)
    scaler = torch.amp.GradScaler("cuda") if (precision == "fp16" and device.startswith("cuda")) else None
    times = []
    _reset_peak(device)
    for i in range(warmup + steps):
        ids = _latent_batch(batch, seq, n_latent, model.config.vocab_size, latent_id, device, gen)
        labels = ids.masked_fill(ids == latent_id, -100)
        _sync(device)
        t = time.perf_counter()
        with autocast_ctx(precision, device):
            loss = lm(ids, am, labels, pos, mode="coconut").loss
        (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t)
    sec = sum(times) / len(times)
    return {"precision": precision, "n_latent": n_latent, "steps": steps, "batch": batch, "seq": seq,
            "sec_per_step": sec, "tokens_per_s": batch * seq / sec, "peak_mem_mb": _peak_mb(device),
            "final_loss": float(loss.detach())}


def parity(device, n_latent=3):
    """Ours. One coconut-mode forward with n_latent latents (ragged 0..n_latent across the batch, reference
    left-padding) of a tiny random GPT-2 on `device` in fp32 vs CPU float64. Returns max |logit diff|."""
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=64, n_head=4, vocab_size=96, n_positions=128)
    base = GPT2LMHeadModel(cfg).eval()
    latent, start, end = 95, 93, 94
    rows, gen = [], torch.Generator().manual_seed(0)
    for i, n in enumerate(range(n_latent + 1)):
        q = torch.randint(3, 90, (6 + 2 * i,), generator=gen).tolist()
        tail = torch.randint(3, 90, (5 + i,), generator=gen).tolist()
        rows.append(q + [start] + [latent] * n + [end] + tail)
    first = max(r.index(latent) if latent in r else 0 for r in rows)
    padded, pos = [], []
    for r in rows:
        k = first - r.index(latent) if latent in r else 0
        padded.append([0] * k + r)
        pos.append([0] * k + list(range(len(r))))
    T = max(len(r) for r in padded)
    ids = torch.tensor([r + [0] * (T - len(r)) for r in padded])
    am = torch.tensor([[0] * (first - r.index(latent) if latent in r else 0) + [1] * len(r) + [0] * (T - len(p))
                       for r, p in zip(rows, padded)])
    pos = torch.tensor([p + [0] * (T - len(p)) for p in pos])
    out = {}
    for dev, dt in ((device, torch.float32), ("cpu", torch.float64)):
        m = GPT2LMHeadModel(cfg).eval()
        m.load_state_dict(base.state_dict())
        lm = LatentLM(m.to(dev, dt), latent, start, end, 0)
        with torch.no_grad():
            out[dev if dt == torch.float32 else "ref"] = lm(ids.to(dev), am.to(dev), None, pos.to(dev)).logits
    mask = am.bool()
    diff = (out[device].double().cpu() - out["ref"])[mask].abs().max().item()
    return {"device": device, "dtype": "float32", "reference": "cpu float64", "n_latent_max": n_latent,
            "max_abs_logit_diff": diff}


def real_parity(base, tokenizer, examples, device, stage, batch=16, c_thought=1, max_latent_stage=6):
    """Ours (registration D4). One coconut-mode forward on a real batch (latent.data stage dataset + Collator, as
    reference get_cot_latent_dataset + MyCollator) on `device` in fp32 vs the CPU in float64; returns max abs / max
    relative logit error over attended positions and the loss error."""
    import copy
    import random

    ids = add_latent_tokens(tokenizer)
    tok = tokenize_examples(examples[:batch], tokenizer)
    recs = build_stage_dataset(tok, stage=stage, arm="coconut", c_thought=c_thought, max_latent_stage=max_latent_stage,
                               pad_latent_to_max=True, uniform_prob=0.0, rng=random.Random(0), **ids)
    b = Collator(tokenizer.pad_token_id, ids["latent_id"])(recs)
    out = {}
    for dev, dt in ((device, torch.float32), ("cpu", torch.float64)):
        m = copy.deepcopy(base).to(dev, dt).eval()
        lm = LatentLM(m, ids["latent_id"], ids["start_id"], ids["end_id"], tokenizer.eos_token_id)
        with torch.no_grad():
            o = lm(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["labels"].to(dev),
                   b["position_ids"].to(dev), mode="coconut")
        out["acc" if dt == torch.float32 else "ref"] = (o.logits.double().cpu(), o.loss.double().cpu())
        del m, lm
    mask = b["attention_mask"].bool()
    la, lr = out["acc"][0][mask], out["ref"][0][mask]
    err = (la - lr).abs()
    n_lat = int((b["input_ids"] == ids["latent_id"]).sum(1).max())
    return {"device": device, "dtype": "float32", "reference": "cpu float64", "stage": stage, "n_latent": n_lat,
            "batch": int(b["input_ids"].shape[0]), "seq": int(b["input_ids"].shape[1]),
            "max_abs_logit_diff": float(err.max()),
            "max_rel_logit_diff": float((err / lr.abs().clamp_min(1e-6)).max()),
            "loss": float(out["ref"][1]), "loss_abs_diff": float((out["acc"][1] - out["ref"][1]).abs()),
            "loss_rel_diff": float((out["acc"][1] - out["ref"][1]).abs() / out["ref"][1].abs())}


def main_real(args, device):
    """Ours. The --real mode (see the module docstring)."""
    base, tokenizer = load_base("openai-community/gpt2", torch.float32, "cpu", resize=True)
    data = args.data or os.path.join(EXP, "..", "..", "data", "prosqa_valid.json")
    examples = load_examples(data, max_size=args.batch)
    rows = []
    for stage in (3, 6):
        r = real_parity(base, tokenizer, examples, device, stage, batch=args.batch)
        rows.append(r)
        print(f"real parity stage {stage} ({r['n_latent']} latents, batch {r['batch']}, seq {r['seq']}) {device} fp32 "
              f"vs cpu float64: max abs logit diff {r['max_abs_logit_diff']:.3e}, max rel "
              f"{r['max_rel_logit_diff']:.3e}, loss {r['loss']:.6f} abs diff {r['loss_abs_diff']:.3e}", flush=True)
    result = {"device": device, "model": "openai-community/gpt2", "data": os.path.abspath(data),
              "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform(),
              "parity": rows, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = args.out or os.path.join(EXP, "results", f"parity_real_{device.split(':')[0]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)
    return result


def main(argv=None):
    """Ours. CLI entry point."""
    ap = argparse.ArgumentParser(description="GPT-2 124M throughput + coconut parity calibration")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=None, help="default 16 on cuda / mps; 2 on cpu (batch 16 x seq 320 "
                                                             "fp32 needs about 14 GB of host memory)")
    ap.add_argument("--warmup", type=int, default=None, help="untimed steps; default 1 on cuda / mps, 0 on cpu")
    ap.add_argument("--seq", type=int, default=320)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="gpt2", help="model shape to probe (gpt2-medium: 355M)")
    ap.add_argument("--latents", type=int, default=6, help="extra coconut-mode fp32 row with this many latents "
                                                           "(the ProsQA final stage); 0 skips it")
    ap.add_argument("--out", default=None)
    ap.add_argument("--real", action="store_true", help="real GPT-2 124M + one ProsQA batch parity only (D4)")
    ap.add_argument("--data", default=None, help="--real: ProsQA file (default data/prosqa_valid.json)")
    args = ap.parse_args(argv)
    device = resolve_device(args.device)
    cpu = device == "cpu"
    if args.batch is None:
        args.batch = 2 if cpu and not args.real else 16
        if cpu and not args.real:
            print("cpu: batch 2 (pass --batch 16 for the registered shape if the host has about 14 GB free)")
    warmup = args.warmup if args.warmup is not None else (0 if cpu else 1)
    if args.real:
        return main_real(args, device)
    model, source = build_gpt2(args.shape)
    precisions = ["fp32"] + (["fp16", "bf16"] if device.startswith("cuda") or device == "mps" else [])
    rows = []
    for prec in precisions:
        try:
            r = bench(model, device, prec, args.steps, args.batch, args.seq, warmup=warmup)
        except Exception as e:  # e.g. bf16 autocast unsupported on this device
            r = {"precision": prec, "error": repr(e)}
        rows.append(r)
        print(json.dumps(r), flush=True)
    if args.latents:
        r = bench(model, device, "fp32", args.steps, args.batch, args.seq, n_latent=args.latents, warmup=warmup)
        rows.append(r)
        print(json.dumps(r), flush=True)
    par = parity(device)
    print(f"parity {device} fp32 vs cpu float64: max abs logit diff {par['max_abs_logit_diff']:.3e}")
    result = {"device": device, "model": source, "torch": torch.__version__, "transformers": transformers.__version__,
              "platform": platform.platform(), "machine": platform.machine(), "bench": rows, "parity": par,
              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = args.out or os.path.join(EXP, "results", f"calibration_{device.split(':')[0]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", out)
    return result


if __name__ == "__main__":
    main()
