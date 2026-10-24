"""Hour caps for the window's train jobs from measured throughput (PLAN.md amendment A4: the oracle caps are
re-estimated from the prep VM's fp32-tf32 numbers before the first oracle admission).

    python3 code/estimate_hours.py --smoke results/prep/prep-main [--jobs configs/jobs/*.json] [--sample 60000]
    python3 code/estimate_hours.py --tps 42000 --step-s 0.045 --val-s-per-ex 0.05 ...      (manual override)

Method (ours). For every run of every train job: load the config, apply the run's --set overrides, tokenize the
training data with the repo's GPT-2 tokenizer (common/babylm/recipes/sprint1_port/gpt2_tokenizer, the 50257-token
vocabulary of openai-community/gpt2) exactly as latent.data.tokenize_examples does, build each stage's dataset with
latent.data.build_stage_dataset, shuffle, cut batches of batch_size and pad them with latent.data.Collator; this
gives the padded tokens (attention_mask.numel(), the quantity train.py logs as padded_tokens) and the number of
sequential latent steps per epoch. Seconds per epoch:

    train = padded_tokens / tps + n_batches * n_latent_steps * step_s          (+ C3: x pass_factor)
    val   = n_val * val_s_per_ex * (max_new_tokens / 128)

with tps (padded tokens per second of a plain forward+backward+step, fp32-tf32), step_s (seconds one extra latent
step costs a batch of 32 beyond its share of padded tokens) and val_s_per_ex read from the prep smoke's log.jsonl
rows (smoke_prosqa_cot_gpu stage 0: tps; smoke_prosqa_gpu epoch 2 at stage 1: step_s; the validation residual of
both: val_s_per_ex). A large training set is sampled (--sample) and scaled by the record count. The job cap is
1.3 x the sum of its runs + 0.5 h (boot, data, parity check, checkpoint sync), rounded up to the half hour.
The rates are GPT-2 124M rates; a run on another base (the 355M plain-depth control, model_id gpt2-medium) divides tps
and multiplies step_s and val_s_per_ex by its per-token FLOP ratio to 124M (model_cost_factor: 2 (N_ne + V d) +
2 n_L d T at T = 512; 2.86 for gpt2-medium; no budget for the better GPU utilisation of the wider model, the
conservative side). Nothing here enters a cell: it sets max_hours and est_hours only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "common"))

from latent.data import Collator, build_stage_dataset, load_examples  # noqa: E402

TOK_DIR = ROOT / "common" / "babylm" / "recipes" / "sprint1_port" / "gpt2_tokenizer"   # GPT-2's 50257-token vocabulary
SPECIAL = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
PASS_FACTOR = {"foreign": 3.0, "none": 2.0}      # C3: true pass + scrambled pass(es) per batch (train.py logs them)
BASE_MODEL = "openai-community/gpt2"
MODEL_SHAPES = {"openai-community/gpt2": (12, 768), "openai-community/gpt2-medium": (24, 1024),
                "openai-community/gpt2-large": (36, 1280)}           # (n_layer, d_model)


def model_cost_factor(model_id: str | None, T: int = 512, V: int = 50260) -> float:
    """Per-token training FLOPs of model_id over GPT-2 124M's (latent.accounting's 2 (N_ne + V d) + 2 n_L d_a T, with
    N_ne = n_L (12 d^2 + 13 d) + 2 d for the GPT-2 block); the measured rates are 124M rates."""
    def per_token(n_l: int, d: int) -> float:
        n_ne = n_l * (12 * d * d + 13 * d) + 2 * d
        return 2.0 * (n_ne + V * d) + 2.0 * n_l * d * T
    model_id = model_id or BASE_MODEL
    if model_id not in MODEL_SHAPES:
        raise SystemExit(f"estimate_hours: no shape for model_id {model_id!r}; add it to MODEL_SHAPES")
    return per_token(*MODEL_SHAPES[model_id]) / per_token(*MODEL_SHAPES[BASE_MODEL])


def tokenizer():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(TOK_DIR))
    for t in SPECIAL:
        tok.add_tokens([t])
    ids = {"start_id": tok.convert_tokens_to_ids(SPECIAL[0]), "end_id": tok.convert_tokens_to_ids(SPECIAL[1]),
           "latent_id": tok.convert_tokens_to_ids(SPECIAL[2])}
    return tok, ids


def tokenize_fast(examples, tok) -> list[dict]:
    """tokenize_examples with batched encode calls (same pieces: q+"\\n", each step+"\\n", "### "+answer, eos)."""
    qs = tok([e.question + "\n" for e in examples], add_special_tokens=True)["input_ids"]
    flat = [s + "\n" for e in examples for s in e.steps]
    st = tok(flat, add_special_tokens=False)["input_ids"] if flat else []
    an = tok(["### " + e.answer for e in examples], add_special_tokens=False)["input_ids"]
    out, k = [], 0
    for i, e in enumerate(examples):
        n = len(e.steps)
        out.append({"question_tokenized": qs[i], "steps_tokenized": st[k:k + n],
                    "answer_tokenized": an[i] + [tok.eos_token_id], "idx": i})
        k += n
    return out


def epoch_shape(tok_train, cfg, stage, ids, pad_id, rng) -> tuple[int, int, int]:
    """(padded_tokens, n_batches, latent_steps_per_sequence) for one epoch at `stage`."""
    arm = cfg["arm"]
    recs = build_stage_dataset(tok_train, stage=stage, arm=arm, c_thought=cfg.get("c_thought", 1),
                               max_latent_stage=cfg.get("max_latent_stage", 0),
                               pad_latent_to_max=cfg.get("pad_latent_to_max", True), uniform_prob=0.0,
                               latent_id=ids["latent_id"], start_id=ids["start_id"], end_id=ids["end_id"], rng=rng)
    rng.shuffle(recs)
    bs = cfg.get("batch_size", 32)
    coll = Collator(pad_id, ids["latent_id"])
    padded, n_lat = 0, 0
    for i in range(0, len(recs), bs):
        cb = coll(recs[i:i + bs])
        padded += int(cb["attention_mask"].numel())
        n_lat = max(n_lat, int((cb["input_ids"] == ids["latent_id"]).sum(1).max()))
    feedback = arm in ("coconut", "coconut_no_curriculum")      # pause: one pass, no sequential steps
    return padded, math.ceil(len(recs) / bs), (n_lat if feedback else 0)


def scheduled_stage(cfg, epoch):
    return 0 if cfg["arm"] in ("cot", "no_cot") else epoch // cfg.get("epochs_per_stage", 5)


def estimate_run(cfg, rates, tok, ids, sample, seed=0):
    mf = model_cost_factor(cfg.get("model_id"))
    rates = {"tps": rates["tps"] / mf, "step_s": rates["step_s"] * mf, "val_s_per_ex": rates["val_s_per_ex"] * mf}
    train = load_examples(str(ROOT / cfg["train_path"]), cfg.get("max_train_examples"))
    n_total = len(train)
    if sample and n_total > sample:
        random.Random(seed).shuffle(train)
        train = train[:sample]
    scale = n_total / len(train)
    tok_train = tokenize_fast(train, tok)
    n_val = len(load_examples(str(ROOT / cfg["val_path"]), cfg.get("max_val_examples")))
    start = int(cfg.get("resume", 0) or 0)
    shapes, total_train_s, total_val_s = {}, 0.0, 0.0
    pf = PASS_FACTOR.get(cfg.get("dep_null", "none"), 2.0) if cfg.get("dep_loss") else 1.0
    for epoch in range(start, cfg["num_epochs"]):
        st = scheduled_stage(cfg, epoch)
        if st not in shapes:
            shapes[st] = epoch_shape(tok_train, cfg, st, ids, tok.pad_token_id or tok.eos_token_id,
                                     random.Random(seed * 1_000_003 + epoch))
        padded, nb, nlat = shapes[st]
        train_s = (padded * scale / rates["tps"] + nb * scale * nlat * rates["step_s"]) * pf
        total_train_s += train_s
        total_val_s += n_val * rates["val_s_per_ex"] * (cfg.get("max_new_tokens", 128) / 128.0) * (1 + 0.5 * nlat)
    epochs = cfg["num_epochs"] - start
    return {"epochs": epochs, "n_train": n_total, "n_val": n_val, "train_h": total_train_s / 3600,
            "val_h": total_val_s / 3600, "hours": (total_train_s + total_val_s) / 3600,
            "stages": {s: {"padded_tokens_per_epoch": int(v[0] * scale), "batches": int(v[1] * scale),
                           "latent_steps": v[2]} for s, v in shapes.items()}, "pass_factor": pf,
            "model_id": cfg.get("model_id") or BASE_MODEL, "model_cost_factor": round(mf, 4)}


def rates_from_smoke(prep_dir: Path) -> dict:
    """tps, step_s and val_s_per_ex from the prep smoke logs (fp32-tf32 on the prep VM's GPU)."""
    def rows(name):
        for cand in (prep_dir / f"{name}_bucket" / "log.jsonl", prep_dir / name / "log.jsonl",
                     prep_dir / "results_prep" / name / "log.jsonl"):
            if cand.exists():
                return [json.loads(l) for l in cand.read_text().splitlines() if l.strip()]
        raise SystemExit(f"no log.jsonl for {name} under {prep_dir}")
    cot = rows("smoke_prosqa_cot_gpu")
    coco = rows("smoke_prosqa_gpu")
    # stage-0 rows of the cot smoke give the plain pass rate; the second epoch of a resumed run excludes warm-up
    r0 = cot[-1]
    tps = r0["padded_tokens"] / r0["train_seconds"]
    e2 = [r for r in coco if r["stage"] == 1][-1]
    nb = math.ceil(e2["n_train"] / 32)
    nlat = max(1, e2.get("n_latent_eval") or 1)
    step_s = max(0.0, (e2["train_seconds"] - e2["padded_tokens"] / tps) / (nb * nlat))
    vals = [(r["seconds"] - r["train_seconds"]) / 100.0 for r in cot + coco]        # 100 validation examples
    val_s = sum(vals) / len(vals)
    return {"tps": tps, "step_s": step_s, "val_s_per_ex": val_s,
            "source": {"cot_row": {k: r0[k] for k in ("epoch", "padded_tokens", "train_seconds", "seconds")},
                       "coconut_stage1_row": {k: e2[k] for k in ("epoch", "stage", "padded_tokens", "train_seconds",
                                                                 "seconds", "n_latent_eval")},
                       "precision": r0.get("precision"), "device": r0.get("device")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", help="prep output dir holding the smoke runs' log.jsonl")
    ap.add_argument("--tps", type=float, help="override: padded tokens/s of a plain fp32-tf32 pass")
    ap.add_argument("--step-s", type=float, help="override: seconds per extra latent step per batch of 32")
    ap.add_argument("--val-s-per-ex", type=float, help="override: validation seconds per example (128 new tokens)")
    ap.add_argument("--jobs", nargs="*", default=sorted(glob.glob(str(EXP / "configs" / "jobs" / "*.json"))))
    ap.add_argument("--sample", type=int, default=60000)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--overhead-h", type=float, default=0.5)
    ap.add_argument("--out", default=str(EXP / "results" / "prep" / "hour_estimates.json"))
    a = ap.parse_args(argv)
    if a.smoke:
        rates = rates_from_smoke(Path(a.smoke))
    else:
        if None in (a.tps, a.step_s, a.val_s_per_ex):
            ap.error("give --smoke or all of --tps --step-s --val-s-per-ex")
        rates = {"tps": a.tps, "step_s": a.step_s, "val_s_per_ex": a.val_s_per_ex, "source": "manual"}
    tok, ids = tokenizer()
    report = {"rates": rates, "margin": a.margin, "overhead_h": a.overhead_h, "jobs": {}}
    cache: dict[str, dict] = {}
    for jp in a.jobs:
        job = json.loads(Path(jp).read_text())
        if job.get("kind") != "train":
            continue
        runs, total = [], 0.0
        for r in job["runs"]:
            cfg = json.loads((EXP / r["config"]).read_text())
            cfg.update(r.get("set") or {})
            key = json.dumps({k: cfg.get(k) for k in ("model_id", "train_path", "val_path", "arm", "c_thought",
                                                         "epochs_per_stage",
                                                         "max_latent_stage", "num_epochs", "resume", "batch_size",
                                                         "max_new_tokens", "dep_loss", "dep_null",
                                                         "max_train_examples", "max_val_examples")}, sort_keys=True)
            if key not in cache:
                cache[key] = estimate_run(cfg, rates, tok, ids, a.sample)
            est = cache[key]
            runs.append({"run": r["run"], "config": r["config"], **est})
            total += est["hours"]
        cap = math.ceil((a.margin * total + a.overhead_h) * 2) / 2
        report["jobs"][job["name"]] = {"project_alias": job.get("project_alias"), "gpu": job.get("gpu"),
                                       "current_max_hours": job.get("max_hours"), "sum_est_hours": total,
                                       "proposed_max_hours": cap, "runs": runs}
        flag = "" if cap <= job.get("max_hours", 0) else "   <-- raise"
        print(f"{job['name']:<20} {job.get('project_alias'):<7} sum {total:6.2f} h  cap now {job.get('max_hours')}"
              f"  proposed {cap}{flag}")
        for x in runs:
            print(f"    {x['run']:<28} {x['hours']:6.2f} h  (train {x['train_h']:.2f}, val {x['val_h']:.2f}; "
                  f"{x['epochs']} epochs, {x['n_train']} records)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"rates: {json.dumps({k: v for k, v in rates.items() if k != 'source'})}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
