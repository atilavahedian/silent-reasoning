"""One-batch cross-GPU parity (critic amendment A3 item 5): loss and gradient norm of one fixed training batch, fp32,
TF32 off, from a fixed checkpoint, written as a reference on one GPU type and compared on another.

    PYTHONPATH=common python3 code/parity_check.py \
        --config configs/smoke_prosqa.json --checkpoint CKPT \
        (--write-reference ref.json | --compare ref.json --out result.json) [--set key=value ...] \
        [--stage 6] [--batch 16] [--loss-rtol 1e-5] [--grad-rtol 1e-4] [--precision fp32|fp32-tf32]

The batch: the first `--batch` examples of the config's val_path (data/prosqa_valid.json for ProsQA), built with
latent.data at `--stage` (6 = every step latent on ProsQA), collated as in training, model in train mode with dropout
disabled (eval-mode dropout, train-mode graph) so the number is deterministic. Records loss, the global L2 norm of
all parameter gradients, the checkpoint sha256, GPU name, torch/CUDA versions and the TF32 flags. --compare exits 1
when |loss - ref| / |ref| > loss-rtol or |gnorm - ref| / ref > grad-rtol (A3: 1e-5 and 1e-4).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common"))
sys.path.insert(0, HERE)


def measure(cfg, checkpoint: str, stage: int, batch: int, precision: str = "fp32") -> dict:
    """precision: "fp32" (strict: TF32 off, "highest"; the A3 item 5 parity) or "fp32-tf32" (the oracle's registered
    flag set, A4; used by the A4 item 3 diagnostic that compares the two on one batch). Flags via train.set_numerics."""
    import torch
    from latent.data import Collator, add_latent_tokens, build_stage_dataset, load_examples, tokenize_examples
    from latent.train import build_model, resolve_device, set_numerics, set_seed, sha256_file

    if precision not in ("fp32", "fp32-tf32"):
        raise SystemExit(f"--precision must be fp32 or fp32-tf32, not {precision!r}")
    set_numerics(precision)
    set_seed(0)
    device = resolve_device(cfg.device)
    lm, tokenizer, ids = build_model(cfg, load_path=checkpoint)
    lm.to(device)
    lm.eval()                       # deterministic: no dropout
    examples = load_examples(cfg.val_path, batch)
    tok = tokenize_examples(examples, tokenizer)
    recs = build_stage_dataset(tok, stage=stage, arm=cfg.arm, c_thought=cfg.c_thought,
                               max_latent_stage=cfg.max_latent_stage, pad_latent_to_max=cfg.pad_latent_to_max,
                               uniform_prob=0.0, rng=random.Random(0), **ids)
    b = Collator(tokenizer.pad_token_id, ids["latent_id"])(recs)
    b = {k: v.to(device) for k, v in b.items() if k != "idx"}
    mode = "coconut" if cfg.arm in ("coconut", "coconut_no_curriculum") else "pause"
    lm.zero_grad(set_to_none=True)
    out = lm(**b, mode=mode)
    out.loss.backward()
    sq = sum(float(p.grad.double().pow(2).sum()) for p in lm.parameters() if p.grad is not None)
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else device
    return {"loss": float(out.loss.detach().double()), "grad_norm": sq ** 0.5, "checkpoint": checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None, "stage": stage,
            "batch": int(b["input_ids"].shape[0]), "seq": int(b["input_ids"].shape[1]), "arm": cfg.arm,
            "val_path": os.path.relpath(cfg.val_path, REPO), "device": device, "gpu": gpu,
            "torch": torch.__version__, "cuda": torch.version.cuda, "precision": precision,
            "tf32": {"cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                     "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                     "float32_matmul_precision": torch.get_float32_matmul_precision()}}


def compare(ref: dict, got: dict, loss_rtol: float, grad_rtol: float) -> dict:
    lr = abs(got["loss"] - ref["loss"]) / max(abs(ref["loss"]), 1e-30)
    gr = abs(got["grad_norm"] - ref["grad_norm"]) / max(abs(ref["grad_norm"]), 1e-30)
    same_ckpt = ref.get("checkpoint_sha256") == got.get("checkpoint_sha256")
    ok = lr <= loss_rtol and gr <= grad_rtol and same_ckpt
    return {"ok": ok, "loss_rel_diff": lr, "grad_norm_rel_diff": gr, "loss_rtol": loss_rtol, "grad_rtol": grad_rtol,
            "same_checkpoint": same_ckpt, "reference": ref, "measured": got}


def main(argv=None) -> int:
    from run_train import load_config
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-reference")
    g.add_argument("--compare")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--stage", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--loss-rtol", type=float, default=1e-5)
    ap.add_argument("--grad-rtol", type=float, default=1e-4)
    ap.add_argument("--precision", choices=("fp32", "fp32-tf32"), default="fp32",
                    help="fp32 = strict (A3 item 5 parity); fp32-tf32 = the oracle flags (A4 item 3 diagnostic)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    got = measure(cfg, a.checkpoint, a.stage, a.batch, a.precision)
    if a.write_reference:
        with open(a.write_reference, "w") as f:
            json.dump(got, f, indent=2)
        print(json.dumps(got))
        return 0
    with open(a.compare) as f:
        ref = json.load(f)
    res = compare(ref, got, a.loss_rtol, a.grad_rtol)
    with open(a.out or "parity_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("reference", "measured")}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
