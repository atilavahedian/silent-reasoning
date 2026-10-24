"""Train one arm from a JSON config: the CLI counterpart of the reference's `torchrun run.py args/<task>.yaml`.

Usage (from anywhere; relative paths in the config resolve against the repo root):
    PYTHONPATH=common python3 code/run_train.py \
        configs/smoke_prosqa.json [--set key=value ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "common"))

from latent.train import TrainConfig, train  # noqa: E402

PATH_KEYS = ("train_path", "val_path", "save_dir", "load_model_path")


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def load_config(path: str, overrides=()) -> TrainConfig:
    """Mirrors run.py's yaml loading into Config: reads the JSON, applies --set overrides, requires an explicit
    max_new_tokens, resolves relative paths against the repo root, and builds a TrainConfig."""
    with open(path) as f:
        d = json.load(f)
    for ov in overrides:
        k, _, v = ov.partition("=")
        d[k] = _parse_value(v)
    if d.get("max_new_tokens") is None:
        raise ValueError(f"{path}: max_new_tokens must be stated explicitly (128 ProsQA / ProntoQA, 64 GSM; D3)")
    for k in PATH_KEYS:
        if d.get(k) and not os.path.isabs(d[k]):
            d[k] = os.path.join(REPO, d[k])
    return TrainConfig.from_dict(d)


def set_tf32(enabled: bool) -> dict:
    """Ours. Sets TF32 for fp32 matmuls and cuDNN explicitly (legacy flags, matmul precision, and the torch >= 2.9
    fp32_precision flags where they exist) and returns the read-back state. Used by run_eval.py; training takes its
    flags from latent.train (registration A3.2, D18), so main() does not call this."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    want = "tf32" if enabled else "ieee"
    for mod in (getattr(torch.backends.cuda, "matmul", None), getattr(torch.backends, "cudnn", None)):
        if mod is not None and hasattr(mod, "fp32_precision"):
            try:
                mod.fp32_precision = want
            except Exception as e:  # pragma: no cover - torch version dependent
                print(f"warning: could not set fp32_precision={want} on {mod}: {e}")
    state = {"tf32_requested": bool(enabled),
             "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
             "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "torch": torch.__version__, "cuda": torch.version.cuda}
    for name, mod in (("cuda.matmul.fp32_precision", getattr(torch.backends.cuda, "matmul", None)),
                      ("cudnn.fp32_precision", getattr(torch.backends, "cudnn", None))):
        if mod is not None and hasattr(mod, "fp32_precision"):
            state[name] = str(mod.fp32_precision)
    return state


def main(argv=None):
    """Mirrors run.py main()'s argument handling (one config file) and calls latent.train.train."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config field (JSON-parsed value), e.g. --set seed=1 --set device=cpu")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    print("Config:", json.dumps(cfg.__dict__))
    out = train(cfg)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    return out


if __name__ == "__main__":
    main()
