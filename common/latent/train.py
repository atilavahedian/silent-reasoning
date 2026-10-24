"""Training loop mirroring facebookresearch/coconut (MIT) run.py main(), single process, no wandb.

Kept from the reference: scheduled_stage = 0 for cot / no_cot else epoch // epochs_per_stage; the stage dataset is
rebuilt and shuffled every epoch; AdamW(lr, weight_decay) with torch defaults otherwise, re-created at the start of
every epoch when reset_optimizer (run.py does this every epoch, not only on a stage change; optimizer_reset="stage"
gives the INTERFACE.md wording); loss / grad_accum with an optimizer step every grad_accum micro-batches and on the
last one (4 GPUs x 32 in the reference = 32 x 4 accumulation here); validation loss on the stage dataset of the
validation set; validation generation accuracy on get_question_latent_dataset prompts; best checkpoint when accuracy
strictly beats the best so far; every-epoch checkpoint unless save_only_improve; `resume` skips the first epochs
(and their stages; the stage arithmetic stays epoch // epochs_per_stage); an existing run directory is resumed from
its latest checkpoint, ignoring `resume`, as run.py does.

Checkpoints (registration 3.3 / D6; the reference keeps checkpoint_<epoch> files and selection happens afterwards):
ckpt_best_final.pt = best validation accuracy among FINAL-stage epochs (scheduled stage >= max_latent_stage for the
curriculum arms; every epoch for cot, no_cot and coconut_no_curriculum), strict >, so ties go to the earliest epoch:
the selected checkpoint. ckpt_best.pt = best over all epochs (the labelled secondary read). ckpt_last.pt = the latest
epoch (not written under save_only_improve). Both "best" trackers start at -1, so the first eligible epoch is always
saved (the reference starts at 0 and never saves a 0-accuracy run). log.jsonl records every epoch's stage, whether it
is final-stage, and which files it wrote.

Phase 2 cards (docs/cards/; registration A3.4), all off by default (the Phase 1 path is unchanged):
* C1b `latent_band: [a, b]`: LatentLM runs blocks [a, b) at latent positions (model.py); training FLOPs charge the
  latent positions layer_fraction x 2 N_ne (accounting.py); the per-epoch validation logs the measured wall-clock per
  latent step and per emitted token.
* C2 `state_sup`, `state_lambda` (0.1), `state_null` ("none" | "shuffled"): loss = CE + lambda L_state from
  aux_losses.StateDecoder on the fed-back latents, targets from latent.state_targets (training split only; shuffled =
  a fixed seeded derangement within step-count classes); no state loss at stage 0 or beyond the stage's latents;
  validation loss stays CE only; the decoder is saved beside each checkpoint as state_decoder_{last,best,best_final}.pth
  and is never used by generate / evaluate. The run-start target statistics (A3.4: the fraction of records whose last
  supervised target contains the answer) go to config.json ("_state_targets") and state_targets.json.
* C3 `dep_loss`, `dep_margin` (0.2), `dep_beta` (1.0), `dep_null` ("none" | "foreign"): a second forward pass per
  batch in mode "scrambled" with a bank from a per-batch, per-epoch seeded derangement within latent-count classes,
  latents detached; loss = CE_true + beta mean_i max(0, m - (CE_scr(i) - CE_true(i))); C3-foreign (A3.4) adds a third
  pass and hinges CE_B against CE_A. No hinge at stage 0. Every extra pass is charged to the training FLOPs (logged
  per pass kind; the arm is "training-heavy").
Registration A3.2 / A3.6: D18 (TF32 off and "highest" matmul precision for fp32 runs, asserted every epoch, logged in
every row; bf16 rows log the flags too), D17 (optimizer state saved beside the latest checkpoint as optimizer_last.pth
for arms without a per-epoch reset, restored on resume), D19 (every resume logged with epoch, source, sha256 and host
in ckpt_state.json and in the first log.jsonl row after it; runs_index_row.csv in save_dir: run, seed, device, GPU
type, resumes, final epoch, "fragmented" beyond 3 resumes; ckpt_last.pt sha256 checked against ckpt_state.json).
"""
from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import socket
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from latent import aux_losses as AX
from latent import state_targets as ST
from latent.accounting import flops_train_per_sequence, model_dims
from latent.data import Collator, add_latent_tokens, build_stage_dataset, load_examples, tokenize_examples
from latent.eval import evaluate, summarize
from latent.model import LatentLM, init_latent_embeddings, load_base

NO_MARKER_ARMS = ("cot", "no_cot")


@dataclass
class TrainConfig:
    """Mirrors the reference yaml (args/*.yaml) fields: coconut/cot/no_cot flags become `arm`,
    batch_size_training becomes batch_size, gradient_accumulation_steps becomes grad_accum, bf16 becomes precision,
    save_path/name becomes save_dir. Fields after max_train_examples are ours."""
    name: str
    arm: str
    model_id: str = "openai-community/gpt2"
    load_model_path: str | None = None
    train_path: str = "data/prosqa_train.json"
    val_path: str = "data/prosqa_valid.json"
    c_thought: int = 1
    epochs_per_stage: int = 5
    max_latent_stage: int = 6
    pad_latent_to_max: bool = True
    uniform_prob: float = 0.0
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    grad_accum: int = 4
    seed: int = 0
    resume: int = 0
    reset_optimizer: bool = True
    save_only_improve: bool = False
    precision: str = "fp32"          # "fp32" (strict) | "fp32-tf32" (oracle, A4) | "bf16" | "fp16" (autocast)
    device: str = "auto"
    save_dir: str = "runs/default"
    max_new_tokens: int | None = None   # required (registration D3): 128 ProsQA / ProntoQA, 64 GSM
    eval_batch_size: int = 32
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    optimizer_reset: str = "epoch"   # "epoch" (reference run.py; every oracle config) | "stage" (diagnostic only)
    load_model_epoch: int | None = None    # names the initial checkpoint (registration D15), logged
    load_model_sha256: str | None = None   # verified before loading when set
    auto_resume: bool = True
    log_every: int = 50
    latent_band: list | None = None    # C1b: [a, b] blocks run at latent positions; None = full depth
    state_sup: bool = False            # C2
    state_lambda: float = 0.1
    state_null: str = "none"           # "none" | "shuffled"
    dep_loss: bool = False             # C3
    dep_margin: float = 0.2
    dep_beta: float = 1.0
    dep_null: str = "none"             # "none" | "foreign"

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """Ours. Builds a config from a JSON dict; keys starting with "_" are comments and are dropped."""
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**d)


def resolve_device(name: str) -> str:
    """Ours (the reference is CUDA-only). "auto" picks cuda, then mps, then cpu."""
    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Mirrors utils.set_seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Registered numerics (registration A3.2 and amendment A4, D18): "fp32" = strict (TF32 off, "highest"), used by the
# precision-pair run; "fp32-tf32" = fp32 storage and accumulation with TF32 matmuls on ("high"), the oracle's
# registered setting; bf16 / fp16 = autocast with fp32 master weights (screens). bf16 runs "on top of the oracle's
# TF32 setting" (A4 item 4; amendment A10.5): its registered flags are the fp32-tf32 set, set at process start and
# asserted per epoch like the oracle's.
REGISTERED_FLAGS = {"fp32": (False, False, "highest"), "fp32-tf32": (True, True, "high"),
                    "bf16": (True, True, "high")}


def set_numerics(precision: str) -> dict:
    """Ours (registration A3.2, D18; amendments A4, A10.5). Sets the registered TF32 flags for fp32, fp32-tf32 and
    bf16 before the model is built; every precision: the read-back flags (logged in every row)."""
    if precision in REGISTERED_FLAGS:
        mm, cd, mp = REGISTERED_FLAGS[precision]
        torch.backends.cuda.matmul.allow_tf32 = mm
        torch.backends.cudnn.allow_tf32 = cd
        torch.set_float32_matmul_precision(mp)
    return numerics_state(precision)


def numerics_state(precision: str) -> dict:
    """Ours (D18). The three flags as torch reports them; raises if a fp32, fp32-tf32 or bf16 run's flags differ from
    the registered values (such a run would be "precision-mismatched" and enter no cell)."""
    st = {"cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
          "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
          "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if precision in REGISTERED_FLAGS:
        want = REGISTERED_FLAGS[precision]
        got = (st["cuda.matmul.allow_tf32"], st["cudnn.allow_tf32"], st["float32_matmul_precision"])
        if got != want:
            raise RuntimeError(f"{precision} run with flags {st}, registered {want} (registration A3.2/A4, D18)")
    return st


def autocast_ctx(precision: str, device: str):
    """Ours (the reference casts the whole model with `bf16: True`). fp32: no autocast; bf16 / fp16: torch.autocast
    on the device type."""
    if precision in ("fp32", "fp32-tf32"):
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)


def make_scaler(precision: str, device: str):
    """Ours. A GradScaler for fp16 where torch supports one on the device; None otherwise."""
    if precision != "fp16":
        return None
    try:
        return torch.amp.GradScaler(device.split(":")[0])
    except Exception as e:  # pragma: no cover - device dependent
        print(f"warning: no GradScaler on {device} ({e}); fp16 runs unscaled")
        return None


def load_state(base_model, path: str, map_location="cpu"):
    """Mirrors run.py's checkpoint loading (load_state_dict(strict=False)); accepts a plain causal-LM state dict
    or a Coconut / LatentLM one (keys "base_causallm.*"; the reference's "embedding.*" alias is dropped)."""
    sd = torch.load(path, map_location=map_location)
    if any(k.startswith("base_causallm.") for k in sd):
        sd = {k[len("base_causallm."):]: v for k, v in sd.items() if k.startswith("base_causallm.")}
    return base_model.load_state_dict(sd, strict=False)


def _ckpt_vocab(path):
    sd = torch.load(path, map_location="cpu")
    for k, v in sd.items():
        if k.endswith("wte.weight") or k.endswith("embed_tokens.weight"):
            return v.shape[0]
    return None


def build_model(cfg: TrainConfig, model=None, tokenizer=None, load_path: str | None = None):
    """Mirrors run.py model setup: from_pretrained, pad = eos, three latent tokens, a plain (un-resized) checkpoint
    is loaded before the resize, embeddings resized and initialised from "<<" for non-CoT arms, then a resized
    (Coconut / resume) checkpoint is loaded. `model` / `tokenizer` inject a prebuilt base (tests)."""
    if model is None:
        model, tokenizer = load_base(cfg.model_id, torch.float32, "cpu", resize=False)
    ids = add_latent_tokens(tokenizer)
    pending = load_path
    if pending and _ckpt_vocab(pending) == model.get_input_embeddings().weight.shape[0]:
        print("load", pending, load_state(model, pending))
        pending = None
    if cfg.arm not in NO_MARKER_ARMS:
        init_latent_embeddings(model, tokenizer, ids)
    if pending:
        print("load", pending, load_state(model, pending))
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], tokenizer.eos_token_id,
                  latent_band=getattr(cfg, "latent_band", None))
    return lm, tokenizer, ids


def default_max_new_tokens(cfg: TrainConfig) -> int:
    """Mirrors run.py: 64 if "gsm" in val_path else 128 (the value every config must state explicitly)."""
    return 64 if "gsm" in cfg.val_path else 128


def is_final_stage(cfg: TrainConfig, stage: int, max_steps_train: int | None = None) -> bool:
    """Ours (the reference keeps every checkpoint and selects later); registration 3.3 and amendment A2 (D16): the
    selection pool. cot, no_cot and coconut_no_curriculum: every epoch. Curriculum arms: the regime in which every
    language step of every training example is replaced, i.e. scheduled stage >= min(max_latent_stage + 1,
    max_steps_train), where max_steps_train is the largest step count in the training set. Under the reference
    semantics a stage k <= max_latent_stage still keeps steps beyond the k-th as text (GSM epochs 9-11 at stage 3),
    so those epochs are not final; past the cap every step is removed. With max_steps_train unknown (None) the
    threshold is max_latent_stage + 1, the conservative choice."""
    if cfg.arm in NO_MARKER_ARMS + ("coconut_no_curriculum",):
        return True
    threshold = cfg.max_latent_stage + 1
    if max_steps_train is not None:
        threshold = min(threshold, max_steps_train)
    return stage >= threshold


def sha256_file(path: str) -> str:
    """Ours. Hex SHA-256 of a file (checkpoint pinning, registration D15)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scheduled_stage(cfg: TrainConfig, epoch: int) -> int:
    """Mirrors run.py: 0 if cot / no_cot else epoch // epochs_per_stage (the dataset builder caps and pins it)."""
    return 0 if cfg.arm in NO_MARKER_ARMS else epoch // cfg.epochs_per_stage


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != "idx"}


def _save(lm, path):
    torch.save(lm.base_causallm.state_dict(), path)


DECODER_FILES = {"ckpt_last.pt": "state_decoder_last.pth", "ckpt_best.pt": "state_decoder_best.pth",
                 "ckpt_best_final.pt": "state_decoder_best_final.pth"}
OPTIMIZER_FILE = "optimizer_last.pth"
LATENT_FEEDBACK_ARMS = ("coconut", "coconut_no_curriculum")
STATE_NULL_SEED = 7_000_003      # C2-shuffled: fixed derangement seed (+ cfg.seed)
DECODER_INIT_SEED = 9_000_011    # C2 decoder initialisation (+ cfg.seed), independent of the training streams


def check_phase2(cfg: TrainConfig) -> None:
    """Ours. The Phase 2 fields need a latent-feedback arm and registered values."""
    used = cfg.latent_band is not None or cfg.state_sup or cfg.dep_loss
    if used and cfg.arm not in LATENT_FEEDBACK_ARMS:
        raise ValueError(f"latent_band / state_sup / dep_loss need a latent-feedback arm {LATENT_FEEDBACK_ARMS}, "
                         f"not {cfg.arm!r}")
    if cfg.state_null not in ("none", "shuffled"):
        raise ValueError(f"state_null must be 'none' or 'shuffled', not {cfg.state_null!r}")
    if cfg.dep_null not in ("none", "foreign"):
        raise ValueError(f"dep_null must be 'none' or 'foreign', not {cfg.dep_null!r} (C3-placebo was withdrawn, A3.4)")
    if cfg.latent_band is not None and len(cfg.latent_band) != 2:
        raise ValueError("latent_band must be [a, b]")


def gpu_type(device: str) -> str:
    """Ours (D19): the accelerator's name for the runs index."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    return device


def _write_runs_index_row(cfg, device, resumes, final_epoch, status):
    """D19: one row per run in save_dir/runs_index_row.csv (rewritten every epoch; the launcher aggregates)."""
    row = {"run": cfg.name, "seed": cfg.seed, "device": device, "gpu_type": gpu_type(device),
           "resumes": len(resumes), "final_epoch": final_epoch, "fragmented": len(resumes) > 3, "status": status}
    with open(os.path.join(cfg.save_dir, "runs_index_row.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


def _row_latents(input_ids, latent_id):
    return (input_ids == latent_id).sum(1).tolist()


def _aux_step(cfg, lm, batch, idx, train_mode, stage, epoch, step, decoder, targets):
    """One training micro-batch with the Phase 2 losses (C2 and / or C3). Returns (out, total_loss, logs, passes)
    where passes counts the forward passes over the batch (1 + scrambled passes) for the FLOP log."""
    out = lm(**batch, mode=train_mode, return_latents=True)
    total, logs, passes = out.loss, {}, 1
    has_lat = stage > 0 and any(x.shape[0] > 0 for x in out.latents)
    if cfg.state_sup and has_lat:
        ls, n_sup, units = AX.state_loss(decoder, out.latents, idx, targets, c_thought=cfg.c_thought,
                                         emb=lm.embedding.weight)
        if ls is not None:
            total = total + cfg.state_lambda * ls
            logs["state_loss"] = float(ls.detach())
            tg = targets[idx[0]]
            S = max(targets[i]["n_symbols"] for i in idx) if tg["kind"] == "prosqa" else 0
            logs["_state_flops_fwd"] = n_sup * decoder.flops_per_target(tg["kind"], lm.lm_head.weight.shape[0], S)
        logs["state_latents"] = n_sup
    if cfg.dep_loss and has_lat:
        counts = [x.shape[0] for x in out.latents]
        ce_true, ntok = AX.per_example_ce(out.logits, batch["labels"])
        pi_a, va = AX.batch_derangement(counts, AX.train_derangement_seed(cfg.seed, epoch, step, 0))
        out_a = lm(**batch, mode="scrambled", scramble_bank=AX.build_bank(out.latents, pi_a, va))
        ce_a, _ = AX.per_example_ce(out_a.logits, batch["labels"])
        passes += 1
        if cfg.dep_null == "foreign":
            pi_b, vb = AX.batch_derangement(counts, AX.train_derangement_seed(cfg.seed, epoch, step, 1), avoid=pi_a)
            out_b = lm(**batch, mode="scrambled", scramble_bank=AX.build_bank(out.latents, pi_b, vb))
            ce_b, _ = AX.per_example_ce(out_b.logits, batch["labels"])
            passes += 1
            valid = torch.tensor([a and b for a, b in zip(va, vb)], device=ntok.device) & (ntok > 0)
            ref, alt = ce_a, ce_b
        else:
            valid = torch.tensor(va, device=ntok.device) & (ntok > 0)
            ref, alt = ce_true, ce_a
        h = AX.hinge(ref, alt, valid, cfg.dep_margin)
        if h is not None:
            total = total + cfg.dep_beta * h
            gap = (alt - ref).detach()[valid]
            logs.update(dep_hinge=float(h.detach()), dep_gap=float(gap.mean()),
                        dep_active=float((gap < cfg.dep_margin).float().mean()), dep_rows=int(valid.sum()))
    return out, total, logs, passes


def _decoder_seed_init(cfg, d_model, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DECODER_INIT_SEED + cfg.seed)
        return AX.StateDecoder(d_model).to(device)


def train(cfg: TrainConfig, *, model=None, tokenizer=None) -> dict:
    """Mirrors run.py main() epoch loop (see the module docstring). Writes save_dir/config.json, log.jsonl (one row
    per epoch), ckpt_last.pt, ckpt_best.pt, ckpt_best_final.pt and ckpt_state.json. Returns best_acc, best_epoch,
    best_final_acc, best_final_epoch, rows."""
    if cfg.max_new_tokens is None:
        raise ValueError(f"max_new_tokens must be set explicitly (reference rule for this val_path: "
                         f"{default_max_new_tokens(cfg)}; registration D3)")
    if cfg.max_new_tokens != default_max_new_tokens(cfg):
        print(f"warning: max_new_tokens {cfg.max_new_tokens} differs from the reference rule "
              f"({default_max_new_tokens(cfg)}); not an oracle configuration")
    check_phase2(cfg)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)
    numerics = set_numerics(cfg.precision)   # D18: before the model is built
    set_seed(cfg.seed)
    state_path = os.path.join(cfg.save_dir, "ckpt_state.json")
    start_epoch, load_path = cfg.resume, cfg.load_model_path
    best = {"all": (-1.0, None), "final": (-1.0, None)}   # (accuracy, 1-based epoch)
    resumes, resume_now, st = [], None, {}
    if cfg.auto_resume and os.path.exists(state_path):
        with open(state_path) as f:
            st = json.load(f)
        start_epoch, load_path = st["latest_epoch"], st["latest_path"]
        best = {k: tuple(v) for k, v in st["best"].items()}
        got = sha256_file(load_path)
        if st.get("latest_sha256") and got != st["latest_sha256"]:
            raise ValueError(f"resume: {load_path} sha256 {got} != ckpt_state.json {st['latest_sha256']} (A3.6)")
        resumes = list(st.get("resumes", []))
        resume_now = {"after_epoch": start_epoch, "source": os.path.basename(load_path), "sha256": got,
                      "host": socket.gethostname(), "device": device, "gpu_type": gpu_type(device),
                      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        resumes.append(resume_now)
        with open(state_path, "w") as f:          # D19: the record survives an epoch that saves nothing
            json.dump({**st, "resumes": resumes}, f)
        print(f"found previous run; resuming after epoch {start_epoch} from {load_path} (`resume` ignored); "
              f"resume {len(resumes)}")
    else:
        if cfg.resume and not cfg.load_model_path:
            print(f"warning: skipping the first {cfg.resume} epochs without loading a checkpoint")
        if cfg.load_model_path and cfg.load_model_sha256:
            got = sha256_file(cfg.load_model_path)
            if got != cfg.load_model_sha256:
                raise ValueError(f"{cfg.load_model_path}: sha256 {got} != configured {cfg.load_model_sha256}")
        elif cfg.load_model_path:
            print(f"warning: {cfg.load_model_path} loaded without a pinned sha256 (registration D15)")
    cfg_dump = asdict(cfg)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=2)

    lm, tokenizer, ids = build_model(cfg, model, tokenizer, load_path)
    lm.to(device)
    dims = model_dims(lm)
    lf = lm.layer_fraction
    collator = Collator(tokenizer.pad_token_id, ids["latent_id"])
    train_examples = load_examples(cfg.train_path, cfg.max_train_examples)
    tok_train = tokenize_examples(train_examples, tokenizer)
    val_examples = load_examples(cfg.val_path, cfg.max_val_examples)
    tok_val = tokenize_examples(val_examples, tokenizer)
    mnt = cfg.max_new_tokens
    train_mode = "coconut" if cfg.arm in LATENT_FEEDBACK_ARMS else "pause"
    stage_kw = dict(arm=cfg.arm, c_thought=cfg.c_thought, max_latent_stage=cfg.max_latent_stage,
                    pad_latent_to_max=cfg.pad_latent_to_max, **ids)
    scaler = make_scaler(cfg.precision, device)

    decoder, targets = None, None
    if cfg.state_sup:                        # C2: targets from the training split only (A3.4)
        own = ST.build_targets(train_examples, tokenizer)
        targets = own
        if cfg.state_null == "shuffled":
            donors = ST.shuffled_donors({i: t["n_steps"] for i, t in own.items()}, STATE_NULL_SEED + cfg.seed)
            targets = {i: own[donors[i]] for i in own}
        stats = ST.target_stats(train_examples, own, cfg.max_latent_stage,
                                used=targets if cfg.state_null == "shuffled" else None)
        stats.update(state_null=cfg.state_null, lambda_=cfg.state_lambda)
        print("state targets:", json.dumps(stats), flush=True)
        cfg_dump["_state_targets"] = stats
        with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
            json.dump(cfg_dump, f, indent=2)
        with open(os.path.join(cfg.save_dir, "state_targets.json"), "w") as f:
            json.dump(stats, f, indent=2)
        decoder = _decoder_seed_init(cfg, dims.d, device)
        dec_path = os.path.join(cfg.save_dir, DECODER_FILES.get(os.path.basename(load_path or ""), ""))
        if resume_now is not None and os.path.isfile(dec_path):
            decoder.load_state_dict(torch.load(dec_path, map_location=device))
    params = list(lm.parameters()) + (list(decoder.parameters()) if decoder is not None else [])

    def new_opt():
        return torch.optim.AdamW(params if decoder is not None else lm.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    optimizer = None if cfg.reset_optimizer else new_opt()
    if optimizer is not None and resume_now is not None and st.get("optimizer_path"):   # D17
        opath = os.path.join(cfg.save_dir, st["optimizer_path"])
        if st.get("optimizer_sha256") and sha256_file(opath) != st["optimizer_sha256"]:
            raise ValueError(f"resume: {opath} does not match ckpt_state.json (A3.6, D17)")
        optimizer.load_state_dict(torch.load(opath, map_location=device))
        resume_now["optimizer_restored"] = True
    prev_stage, rows = None, []
    log_path = os.path.join(cfg.save_dir, "log.jsonl")

    max_steps_train = max((len(ex["steps_tokenized"]) for ex in tok_train), default=0)  # D16
    extras = cfg.state_sup or cfg.dep_loss

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()
        numerics = numerics_state(cfg.precision)          # D18: asserted every epoch
        stage = scheduled_stage(cfg, epoch)
        if cfg.reset_optimizer and (cfg.optimizer_reset == "epoch" or stage != prev_stage or optimizer is None):
            optimizer = new_opt()
        prev_stage = stage
        rng = random.Random(cfg.seed * 1_000_003 + epoch)   # per-epoch data order and uniform_prob draws
        torch.manual_seed(cfg.seed * 1_000_003 + epoch)      # per-epoch dropout stream (resumable)
        recs = build_stage_dataset(tok_train, stage=stage, uniform_prob=cfg.uniform_prob, rng=rng, **stage_kw)
        rng.shuffle(recs)
        chunks = [recs[i:i + cfg.batch_size] for i in range(0, len(recs), cfg.batch_size)]

        lm.train()
        if decoder is not None:
            decoder.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, tokens, padded, flops, opt_steps = 0.0, 0, 0, 0.0, 0
        aux_sum, aux_n, flops_extra, flops_dec = {}, {}, 0.0, 0.0
        for step, chunk in enumerate(chunks):
            cb = collator(chunk)
            batch = _to(cb, device)
            passes = 1
            with autocast_ctx(cfg.precision, device):
                if extras:
                    out, total, logs, passes = _aux_step(cfg, lm, batch, cb["idx"].tolist(), train_mode, stage,
                                                         epoch, step, decoder, targets)
                    flops_dec += 3.0 * logs.pop("_state_flops_fwd", 0.0)
                    for k, v in logs.items():
                        aux_sum[k] = aux_sum.get(k, 0.0) + v
                        aux_n[k] = aux_n.get(k, 0) + 1
                else:
                    out = lm(**batch, mode=train_mode)
                    total = out.loss
            loss = total / cfg.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            if (step + 1) % cfg.grad_accum == 0 or step == len(chunks) - 1:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1
            loss_sum += float(out.loss.detach())
            lengths = batch["attention_mask"].sum(1).tolist()
            tokens += int(sum(lengths))
            padded += int(batch["attention_mask"].numel())
            if lf == 1.0:
                f_seq = sum(flops_train_per_sequence(dims, int(T)) for T in lengths)
            else:
                f_seq = sum(flops_train_per_sequence(dims, int(T), n, lf)
                            for T, n in zip(lengths, _row_latents(batch["input_ids"], ids["latent_id"])))
            flops += f_seq * passes
            flops_extra += f_seq * (passes - 1)
            if cfg.log_every and step % cfg.log_every == 0:
                print(f"epoch {epoch + 1}/{cfg.num_epochs} stage {stage} batch {step}/{len(chunks)} "
                      f"loss {float(out.loss.detach()):.4f}", flush=True)
        flops += flops_dec
        train_seconds = time.time() - t0

        saved, latest = [], None
        if not cfg.save_only_improve:
            latest = os.path.join(cfg.save_dir, "ckpt_last.pt")
            _save(lm, latest)
            saved.append("ckpt_last.pt")
            if decoder is not None:
                torch.save(decoder.state_dict(), os.path.join(cfg.save_dir, DECODER_FILES["ckpt_last.pt"]))

        lm.eval()
        val_recs = build_stage_dataset(tok_val, stage=stage, uniform_prob=cfg.uniform_prob,
                                       rng=random.Random(cfg.seed + 7), **stage_kw)
        vl = []
        with torch.no_grad(), autocast_ctx(cfg.precision, device):
            for i in range(0, len(val_recs), cfg.batch_size):
                vl.append(float(lm(**_to(collator(val_recs[i:i + cfg.batch_size]), device), mode=train_mode).loss))
        timing = [] if cfg.arm in LATENT_FEEDBACK_ARMS else None
        with autocast_ctx(cfg.precision, device):
            res = evaluate(lm, tokenizer, val_examples, arm=cfg.arm, n_latent=0, batch_size=cfg.eval_batch_size,
                           device=device, max_new_tokens=mnt,
                           stage_spec=dict(stage=stage, c_thought=cfg.c_thought,
                                           max_latent_stage=cfg.max_latent_stage,
                                           pad_latent_to_max=cfg.pad_latent_to_max),
                           **({"timing_out": timing} if timing is not None else {}))
        summ = summarize(res, n_boot=1000)
        acc = summ.get("accuracy", 0.0)
        final = is_final_stage(cfg, stage, max_steps_train)
        for key, fname, eligible in (("all", "ckpt_best.pt", True), ("final", "ckpt_best_final.pt", final)):
            if eligible and acc > best[key][0]:
                best[key] = (acc, epoch + 1)
                _save(lm, os.path.join(cfg.save_dir, fname))
                saved.append(fname)
                if decoder is not None:
                    torch.save(decoder.state_dict(), os.path.join(cfg.save_dir, DECODER_FILES[fname]))
                if cfg.save_only_improve and latest is None:
                    latest = os.path.join(cfg.save_dir, fname)
        if latest:
            state = {"latest_epoch": epoch + 1, "latest_path": latest, "best": best,
                     "latest_sha256": sha256_file(latest), "resumes": resumes}
            if optimizer is not None and not cfg.reset_optimizer:            # D17
                torch.save(optimizer.state_dict(), os.path.join(cfg.save_dir, OPTIMIZER_FILE))
                state.update(optimizer_path=OPTIMIZER_FILE,
                             optimizer_sha256=sha256_file(os.path.join(cfg.save_dir, OPTIMIZER_FILE)))
            with open(state_path, "w") as f:
                json.dump(state, f)
        n_lat = res[0].latent_steps if res and cfg.arm in LATENT_FEEDBACK_ARMS else 0
        row = {"epoch": epoch + 1, "epoch_index": epoch, "stage": stage, "final_stage": final,
               "n_latent_eval": n_lat, "saved": saved, "train_loss": loss_sum / max(1, len(chunks)),
               "val_loss": (sum(vl) / len(vl)) if vl else math.nan, "val_acc": acc,
               "val_cot_em": summ.get("cot_em", 0.0), "val_emitted_mean": summ.get("emitted_tokens_mean", 0.0),
               "tokens": tokens, "padded_tokens": padded, "flops": flops, "opt_steps": opt_steps,
               "n_train": len(recs), "seconds": time.time() - t0, "train_seconds": train_seconds,
               "best_acc": best["all"][0], "best_final_acc": best["final"][0], "max_new_tokens": mnt,
               "device": device, "precision": cfg.precision, "numerics": numerics, "resumes": len(resumes)}
        if resume_now is not None and epoch == start_epoch:
            row["resumed_from"] = resume_now
        if extras or lf != 1.0:
            row.update(layer_fraction=lf, flops_extra_passes=flops_extra, flops_state_decoder=flops_dec,
                       flops_primary_pass=flops - flops_extra - flops_dec,
                       **{f"train_{k}": aux_sum[k] / aux_n[k] for k in aux_sum})
        if timing:
            ms = lambda kind: [t[f"{kind}_ms_mean"] for t in timing if t and t.get(f"{kind}_ms_mean") is not None]
            for kind in ("latent", "token"):
                xs = ms(kind)
                row[f"val_{kind}_step_ms"] = (sum(xs) / len(xs)) if xs else None
        rows.append(row)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        _write_runs_index_row(cfg, device, resumes, epoch + 1, "running" if epoch + 1 < cfg.num_epochs else "done")
        print(json.dumps(row), flush=True)
    return {"best_acc": best["all"][0], "best_epoch": best["all"][1], "best_final_acc": best["final"][0],
            "best_final_epoch": best["final"][1], "save_dir": cfg.save_dir, "rows": rows}


__all__ = ["TrainConfig", "train", "build_model", "load_state", "resolve_device", "set_seed", "scheduled_stage",
           "autocast_ctx", "default_max_new_tokens", "is_final_stage", "sha256_file", "set_numerics",
           "numerics_state", "check_phase2"]
