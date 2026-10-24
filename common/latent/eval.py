"""Generation accuracy and per-example cost, mirroring the validation-accuracy block of facebookresearch/coconut
(MIT) run.py, batched, with the interventions of the silent-reasoning plan (scrambled latents, zero latents, step
sweep).

Answer scoring is the reference's: decode prompt + generated tokens with skip_special_tokens=True, take
`text.split("#")[-1].replace(",", "").strip()` and compare it with `answer.replace(",", "").strip()`; the CoT
exact match uses `("\\n".join(text.split("\\n")[1:])).split("#")[0].strip()` against "\\n".join(steps).
"""
from __future__ import annotations

import inspect
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from latent.accounting import inference_record, model_dims
from latent.data import (Collator, answer_extract, answer_normalize, build_stage_dataset, cot_extract,
                         tokenize_examples)
from latent.model import LatentLM

LATENT_ARMS = ("coconut", "coconut_no_curriculum")
INTERVENTIONS = (None, "scrambled", "zero_latent")
STEP_SWEEP = (0, 1, 2, 4, 8)
COST_AXES = ("prefill_tokens", "latent_steps", "emitted_tokens", "sequential_passes", "flops", "reference_flops")


@dataclass
class ExampleResult:
    """Ours: one scored example (the reference keeps only running Cor / CoT / Total counters)."""
    idx: int
    correct: bool
    pred: str
    gold: str
    text: str
    prefill_tokens: int
    latent_steps: int
    emitted_tokens: int
    sequential_passes: int
    flops: float
    cot_match: bool = False          # reference "cot_em"
    latents: object = field(default=None, repr=False)   # [n_latent, d] cpu tensor when return_latents
    reference_flops: float = 0.0     # registration 5.5 (c) / A6.4: the reference's own cost, secondary accounting


def _ids(tokenizer):
    return {k: tokenizer.convert_tokens_to_ids(t) for k, t in
            (("start_id", "<|start-latent|>"), ("end_id", "<|end-latent|>"), ("latent_id", "<|latent|>"))}


def gen_mode(arm: str) -> str:
    """Mirrors run.py: the Coconut wrapper (latent feedback) for coconut arms, the plain causal LM otherwise."""
    return "coconut" if arm in LATENT_ARMS else "pause"


def build_prompts(tok_examples, tokenizer, *, arm: str, n_latent: int | None = None, stage_spec: dict | None = None):
    """Mirrors get_question_latent_dataset. With stage_spec ({stage, c_thought, max_latent_stage,
    pad_latent_to_max}) the reference's per-stage prompt is built for `arm`; otherwise every example gets exactly
    n_latent <|latent|> tokens between the markers (cot / no_cot: question only)."""
    ids = _ids(tokenizer)
    if stage_spec is not None:
        kw = dict(stage_spec)
        arm_b = arm
    else:
        n = int(n_latent or 0)
        kw = dict(stage=n, c_thought=1, max_latent_stage=n, pad_latent_to_max=True)
        arm_b = arm if arm in ("cot", "no_cot") else "coconut"
    return build_stage_dataset(tok_examples, arm=arm_b, uniform_prob=0.0, rng=random.Random(0),
                               for_generation=True, **kw, **ids)


def _record(dims, arm, q_len, prompt_len, latent_steps, emitted, n_latent_tokens=0, layer_fraction=1.0):
    """Registration 5.3 through accounting.inference_record: the v2 signature (dims, arm, q_len, L, E'; for pause L
    is the number of pause tokens) when the installed accounting has it; otherwise the v1 signature with p = prompt
    minus the fed-back latents (same totals). layer_fraction (C1b) is passed only when it is not 1."""
    params = inspect.signature(inference_record).parameters
    if "arm" in params:
        if arm == "pause":
            return inference_record(dims, "pause", q_len, n_latent_tokens, emitted)
        acc_arm = arm if arm in ("cot", "no_cot") else "coconut"
        if layer_fraction != 1.0 and latent_steps:
            return inference_record(dims, acc_arm, q_len, latent_steps, emitted, layer_fraction=layer_fraction)
        return inference_record(dims, acc_arm, q_len, latent_steps, emitted)
    return inference_record(dims, prompt_len - latent_steps, latent_steps, emitted)


def _batches(records, batch_size):
    for i in range(0, len(records), batch_size):
        yield records[i:i + batch_size]


def _derangement(n, rng):
    """Sattolo's algorithm: a uniformly random single n-cycle, hence no fixed points for n >= 2."""
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        p[i], p[j] = p[j], p[i]
    return p


def scramble_donors(latent_counts: dict, seed: int = 0) -> dict:
    """Ours (no reference). Maps each idx to a donor idx with the same latent count, never itself (a seeded
    derangement within each latent-count class). Raises if a class with latents has a single member."""
    rng = random.Random(seed)
    classes = {}
    for idx in sorted(latent_counts):
        classes.setdefault(latent_counts[idx], []).append(idx)
    donor = {}
    for n, members in classes.items():
        if n == 0:
            donor.update({m: m for m in members})
            continue
        if len(members) < 2:
            raise ValueError(f"cannot scramble: only one example has {n} latents")
        perm = _derangement(len(members), rng)
        donor.update({members[k]: members[perm[k]] for k in range(len(members))})
    return donor


def evaluate(model, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str,
             intervention: str | None = None, max_new_tokens: int, return_latents: bool = False,
             stage_spec: dict | None = None, scramble_seed: int = 0, timing_out: list | None = None
             ) -> list[ExampleResult]:
    """Mirrors the run.py "val generation accuracy" loop (get_question_latent_dataset prompts, Coconut.generate,
    answer extraction), batched through LatentLM.generate with per-example accounting from
    accounting.inference_record (registration 5.3 position table; pause tokens count 0 latent steps).
    max_new_tokens has no default: 128 for ProsQA / ProntoQA, 64 for GSM (reference run.py, registration D3).

    intervention: None; "zero_latent" (0 latents, markers kept); "scrambled" (a first clean pass collects every
    example's fed-back vectors, then each example is regenerated with the vectors of a donor of the same latent
    count, see scramble_donors). stage_spec reproduces the reference's per-stage prompt (used by train.py).
    A C1b model (LatentLM with latent_band) is charged its layer_fraction per latent step; timing_out, when a list,
    receives one LatentLM.generate timing dict per batch of the scored pass (C1b A3.4: wall-clock per step)."""
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention {intervention!r}")
    ids = _ids(tokenizer)
    lm = model if isinstance(model, LatentLM) else LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"],
                                                            tokenizer.eos_token_id)
    was_training = lm.training
    lm.eval()
    if intervention == "zero_latent":
        n_latent, stage_spec = 0, (None if stage_spec is None else {**stage_spec, "stage": 0})
    mode = gen_mode(arm)
    if intervention == "scrambled" and mode != "coconut":
        raise ValueError("the scrambled intervention needs a latent-feedback arm")
    by_idx = {ex.idx: ex for ex in examples}
    tok_ex = tokenize_examples(examples, tokenizer)
    q_len = {s["idx"]: len(s["question_tokenized"]) for s in tok_ex}
    records = build_prompts(tok_ex, tokenizer, arm=arm, n_latent=n_latent, stage_spec=stage_spec)
    collator = Collator(tokenizer.pad_token_id, ids["latent_id"])
    dims = model_dims(lm.base_causallm)

    def run(gen_mode_, bank_of=None, want_latents=False, timed=False):
        outs = {}
        for chunk in _batches(records, batch_size):
            b = collator(chunk)
            bank = None
            if bank_of is not None:
                vecs = [bank_of[r["idx"]] for r in chunk]
                n_max = max(1, max(v.shape[0] for v in vecs))
                bank = torch.zeros(len(vecs), n_max, vecs[0].shape[-1], dtype=vecs[0].dtype)
                for i, v in enumerate(vecs):
                    bank[i, :v.shape[0]] = v
                bank = bank.to(device)
            g = lm.generate(b["input_ids"].to(device), b["attention_mask"].to(device), b["position_ids"].to(device),
                            max_new_tokens=max_new_tokens, mode=gen_mode_, scramble_bank=bank,
                            return_latents=want_latents, **({"time_steps": True} if timed else {}))
            if timed:
                timing_out.append(g.timing)
            for i, r in enumerate(chunk):
                outs[r["idx"]] = (r, g.tokens[i], g.emitted_tokens[i], g.latent_steps[i], g.sequential_passes[i],
                                  g.latents[i].float().cpu() if want_latents else None)
        return outs

    if intervention == "scrambled":
        clean = run("coconut", want_latents=True)
        donors = scramble_donors({k: v[5].shape[0] for k, v in clean.items()}, seed=scramble_seed)
        outs = run("scrambled", bank_of={k: clean[donors[k]][5] for k in clean}, want_latents=return_latents,
                   timed=timing_out is not None)
    else:
        outs = run(mode, want_latents=return_latents, timed=timing_out is not None)
    lf = getattr(lm, "layer_fraction", 1.0)

    results = []
    for r in records:
        rec, gen, emitted, steps, passes, lat = outs[r["idx"]]
        ex = by_idx[r["idx"]]
        text = tokenizer.decode(rec["input_ids"] + gen, skip_special_tokens=True)
        pred, gold = answer_extract(text), answer_normalize(ex.answer)
        acc = _record(dims, arm, q_len[r["idx"]], len(rec["input_ids"]), steps, emitted,
                      rec["input_ids"].count(ids["latent_id"]), layer_fraction=lf)
        results.append(ExampleResult(idx=ex.idx, correct=pred == gold, pred=pred, gold=gold, text=text,
                                     prefill_tokens=acc.prefill_tokens, latent_steps=acc.latent_steps,
                                     emitted_tokens=acc.emitted_tokens, sequential_passes=passes, flops=acc.flops,
                                     cot_match=cot_extract(text) == "\n".join(ex.steps), latents=lat,
                                     reference_flops=getattr(acc, "reference_flops", 0.0)))
    lm.train(was_training)
    return results


def _boot_ci(values: np.ndarray, n_boot: int, seed: int, chunk: int = 1000):
    n = len(values)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        means[s:s + k] = values[rng.integers(0, n, size=(k, n))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(results, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Ours (the reference prints only Cor/Total). Accuracy, CoT exact match, mean and median of every cost axis,
    and a seeded percentile-bootstrap 95% CI over examples (n_boot draws) for accuracy and each cost mean."""
    if not results:
        return {"n": 0}
    out = {"n": len(results)}
    acc = np.array([float(r.correct) for r in results])
    out["accuracy"] = float(acc.mean())
    out["accuracy_ci95"] = _boot_ci(acc, n_boot, seed)
    out["cot_em"] = float(np.mean([float(r.cot_match) for r in results]))
    for axis in COST_AXES:
        v = np.array([float(getattr(r, axis)) for r in results])
        out[f"{axis}_mean"] = float(v.mean())
        out[f"{axis}_median"] = float(np.median(v))
        out[f"{axis}_mean_ci95"] = _boot_ci(v, n_boot, seed)  # same draws: paired across axes
    return out


__all__ = ["ExampleResult", "evaluate", "summarize", "scramble_donors", "build_prompts", "gen_mode", "STEP_SWEEP",
           "INTERVENTIONS"]
