"""Readout and faithfulness report for one LatentLM checkpoint (plan 6.5, registration 7.9): R1 logit lens, R2 linear
probes, F1 counterfactual patching, F2 scrambled drop, F3 probe-direction edits, coverage and load-bearing fraction.

Usage (from anywhere):
    PYTHONPATH=common python3 code/run_probes.py \
        --checkpoint runs/prosqa_coconut/ckpt_best_final.pt \
        --config configs/prosqa_coconut.json \
        --split test --task prosqa --out results/probes/prosqa_coconut_s0/

--split is a split name (data/<task>_<split>.json under the repo root) or a path. A relative --out resolves against
the experiment folder. Writes probe_report.json, probe_report.md, latents.npz (gitignored inside --out) and index.json. Probes
are k-fold over the examples of --split, so every decoded score and every edit uses a probe trained on disjoint
examples. Positive control: --task reach --synthetic (trains the tiny from-scratch model; no checkpoint).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "common"))

from latent.probe.edit import ALPHA_GRID, MIN_EFFECT, SIGMA_K  # noqa: E402


def read_config(path: str) -> dict:
    """The training config JSON ("_" keys are comments), as run_train.load_config reads it."""
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def load_model(cfg: dict, checkpoint: str, device: str, model_path: str | None = None):
    """Mirrors run.py / train.build_model for a latent arm through model.py's public API: load_base(resize=True)
    (adds the three latent tokens, resizes, initialises from "<<"), then the checkpoint's state dict
    (keys optionally prefixed "base_causallm.", as train.load_state accepts), wrapped in LatentLM."""
    import torch

    from latent.model import LatentLM, load_base

    model, tok = load_base(model_path or cfg["model_id"], torch.float32, "cpu", resize=True)
    sd = torch.load(checkpoint, map_location="cpu")
    if any(k.startswith("base_causallm.") for k in sd):
        sd = {k[len("base_causallm."):]: v for k, v in sd.items() if k.startswith("base_causallm.")}
    print("load", checkpoint, model.load_state_dict(sd, strict=False))
    ids = {k: tok.convert_tokens_to_ids(t) for k, t in
           (("latent_id", "<|latent|>"), ("start_id", "<|start-latent|>"), ("end_id", "<|end-latent|>"))}
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], tok.eos_token_id).to(device).eval()
    return lm, tok


def load_r3(loader: str | None, path: str, lm, tok, task: str):
    """Resolve "module:function" and call it as fn(path, lm, tokenizer, task) -> StateReadout (linear.py)."""
    import importlib

    if not loader:
        raise SystemExit("--r3 needs --r3-loader MODULE:FUNCTION (the C2 decoder's loader)")
    mod, _, fn = loader.partition(":")
    return getattr(importlib.import_module(mod), fn)(path, lm, tok, task)


def resolve_split(task: str, split: str) -> str:
    if os.path.exists(split):
        return split
    return os.path.join(REPO, "data", f"{task}_{split}.json")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", help="state dict (.pt) of the latent arm")
    ap.add_argument("--config", help="the training config JSON of that run")
    ap.add_argument("--split", default="test", help="split name or JSON path")
    ap.add_argument("--task", required=True, choices=["prosqa", "gsm", "reach"])
    ap.add_argument("--out", required=True, help="output dir; relative paths resolve against the experiment folder")
    ap.add_argument("--synthetic", action="store_true", help="run the T3 positive control instead (task reach)")
    ap.add_argument("--model-path", default=None, help="local base model dir instead of the config's model_id")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--n-latent", type=int, default=None, help="default max_latent_stage * c_thought")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default the config's (128 ProsQA, 64 GSM)")
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHA_GRID))
    ap.add_argument("--min-effect", type=float, default=MIN_EFFECT)
    ap.add_argument("--sigma-k", type=float, default=SIGMA_K)
    ap.add_argument("--downstream", choices=["frozen", "recompute"], default="frozen")
    ap.add_argument("--lb-intervention", choices=["F3", "F1"], default="F3")
    ap.add_argument("--pair-rules", default=None, help="comma list (default all rules of the task)")
    ap.add_argument("--edit-kinds", default=None, help="comma list (default all kinds of the task)")
    ap.add_argument("--skip", default="", help="comma list of R1,F1,F2,F3 to skip")
    ap.add_argument("--r3", default=None, help="saved R3 decoder (the C2 decoder) passed to --r3-loader")
    ap.add_argument("--r3-loader", default=None, metavar="MODULE:FUNCTION",
                    help="fn(path, lm, tokenizer, task) -> latent.probe.linear.StateReadout")
    ap.add_argument("--decoded-by", choices=["R2", "R3"], default="R2", help="readout that defines decoded steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-steps", type=int, default=400)
    return ap.parse_args(argv)


def main(argv=None):
    """Load the checkpoint, run latent.probe.report.run_readout on the split, write probe_report.json / probe_report.md."""
    a = parse_args(argv)
    from latent.data import load_examples
    from latent.probe.report import run_readout, to_markdown, write_report

    out = a.out if os.path.isabs(a.out) else os.path.join(EXP, a.out)
    split_list = lambda s: tuple(x for x in (s or "").split(",") if x)  # noqa: E731
    kw = dict(k_folds=a.k_folds, l2=a.l2, alphas=tuple(float(x) for x in a.alphas.split(",")),
              min_effect=a.min_effect, sigma_k=a.sigma_k, downstream=a.downstream, lb_intervention=a.lb_intervention,
              pair_rules=split_list(a.pair_rules) or None, edit_kinds=split_list(a.edit_kinds) or None,
              skip=split_list(a.skip), seed=a.seed, batch_size=a.batch_size)
    if a.device == "auto":
        import torch
        mps = getattr(torch.backends, "mps", None)
        a.device = "cuda" if torch.cuda.is_available() else ("mps" if mps is not None and mps.is_available() else "cpu")
    if a.synthetic:
        from latent.probe.synthetic import run_positive_control
        rep = run_positive_control(out, steps=a.synthetic_steps, seed=a.seed, device=a.device,
                                   readout_kw={k: v for k, v in kw.items() if k != "batch_size"})
        print(to_markdown(rep).split("## R2")[0])
        return rep
    if not (a.checkpoint and a.config):
        raise SystemExit("--checkpoint and --config are required (or --synthetic)")
    cfg = read_config(a.config)
    c = int(cfg.get("c_thought", 1))
    n_latent = a.n_latent if a.n_latent is not None else int(cfg.get("max_latent_stage", 6)) * c
    mnt = a.max_new_tokens or cfg.get("max_new_tokens") or (64 if a.task == "gsm" else 128)
    split = resolve_split(a.task, a.split)
    examples = load_examples(split, max_size=a.max_examples)
    lm, tok = load_model(cfg, a.checkpoint, a.device, a.model_path)
    meta = {"checkpoint": a.checkpoint, "config": a.config, "split": split, "arm": cfg.get("arm")}
    r3 = load_r3(a.r3_loader, a.r3, lm, tok, a.task) if a.r3 else None
    rep = run_readout(lm, tok, examples, task=a.task, n_latent=n_latent, c_thought=c, device=a.device,
                      max_new_tokens=mnt, out_dir=out, meta=meta, r3=r3, decoded_by=a.decoded_by, **kw)
    paths = write_report(rep, out)
    print(json.dumps({"report": paths, **{k: rep["load_bearing"][k] for k in
                                          ("coverage", "load_bearing_fraction", "label")}}))
    return rep


if __name__ == "__main__":
    main()
