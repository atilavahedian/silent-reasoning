"""F3 probe-direction edits and the load-bearing fraction (plan 6.5 headline; registration 7.9).

Edit. For example i, latent step t and an edit spec (tasks.edit_specs: symbol s, sign, predicted option), take the
out-of-fold R2 probe for "s in frontier" (never trained on example i), its unit direction u = w_raw / ||w_raw|| in
raw latent space and its scale sigma_u (std of the projection h . u over that probe's training latents), and set
    h_t' = h_t + sign * alpha * sigma_u * u           (only step t changes; patch.delta_bank)
for alpha in a registered grid (default 1, 2, 4, 8 probe-projection standard deviations). The probe logit then
moves by exactly sign * alpha * sigma_u * ||w_raw|| (tested). Control: the same norm along a seeded random unit
direction (paired, same example and step). Outcome per example: moved = [answer option == predicted option].
Effect(t, alpha) = mean(moved_edit - moved_control) with a paired example bootstrap; examples whose probe for s
was not trained are not editable at t and are left out.

Load-bearing (defaults are parameters of `load_bearing`):
  * a step t is DECODED iff the registered readout (R2 by default) scores above its shuffled-label baseline by
    `sigma_k` = 2 binomial sigma over examples (linear.py);
  * a step t is LOAD-BEARING iff some alpha in the grid has effect >= `min_effect` (default 0.05, five points of
    answers moved beyond the random-direction control) AND a paired bootstrap interval excluding 0 at level
    1 - 0.05 / len(grid) (Bonferroni over the grid; `bonferroni=False` uses 0.95);
  * coverage = sum_t w_t decoded_t / sum_t w_t, with w_t = examples having step t;
  * load_bearing_fraction (registration 7.9, the headline) = sum_t w_t decoded_t lb_t / sum_t w_t decoded_t;
  * load_bearing_over_all_steps (the operator's brief: decoded AND moved, over all steps) = sum w decoded lb / sum w;
  * label: "explained" iff coverage >= 0.5 and load_bearing_fraction >= 0.5; "partly explained" iff both >= 0.25;
    else "unexplained".
"""
from __future__ import annotations

import numpy as np
import torch

from latent.probe.patch import BankRunner, delta_bank
from latent.probe.stats import bootstrap_mean_ci
from latent.probe.tasks import PRIMARY_EDIT, answer_option, edit_specs

ALPHA_GRID = (1.0, 2.0, 4.0, 8.0)
MIN_EFFECT = 0.05
SIGMA_K = 2.0


def edit_delta(u: np.ndarray, scale: float, alpha: float, sign: float = 1.0) -> np.ndarray:
    """sign * alpha * scale * u / ||u|| (u renormalised defensively)."""
    u = np.asarray(u, dtype=np.float64)
    return (sign * alpha * scale * u / np.linalg.norm(u)).astype(np.float32)


def random_unit(d: int, seed) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(d)
    return (v / np.linalg.norm(v)).astype(np.float32)


def run_edits(lm, tokenizer, examples, ext, probes, *, n_latent: int, alphas=ALPHA_GRID, kinds=None,
              batch_size: int = 32, device: str = "cpu", max_new_tokens: int = 128, downstream: str = "frozen",
              seed: int = 0, bonferroni: bool = True) -> dict:
    """F3 over every latent step, alpha and edit kind. ext must be the Extraction of `examples` (same order)."""
    if [r["idx"] for r in ext.records] != [e.idx for e in examples]:
        raise ValueError("ext.records and examples are not in the same order")
    task = ext.task
    kinds = tuple(kinds) if kinds else (PRIMARY_EDIT[task],)
    level = 1 - 0.05 / len(alphas) if bonferroni else 0.95
    runner = BankRunner(lm, tokenizer, examples, n_latent=n_latent, batch_size=batch_size, device=device,
                        max_new_tokens=max_new_tokens)
    own = [torch.as_tensor(ext.latents[i, :n_latent]) for i in range(len(examples))]
    base = [answer_option(p, task) for p in runner.run([(e.idx, o, n_latent) for e, o in zip(examples, own)])]
    specs = [{s["kind"]: s for s in edit_specs(r)} for r in ext.records]
    out = {"task": task, "downstream": downstream, "alphas": list(alphas), "kinds": list(kinds), "ci_level": level,
           "per_kind": {}}
    for kind in kinds:
        rows = []
        for t in range(n_latent):
            todo = []
            for i, e in enumerate(examples):
                sp = specs[i].get(kind)
                if sp is None or ext.n_latent[i] <= t:
                    continue
                dirn = probes.direction(i, sp["name"])
                if dirn is not None:
                    todo.append((i, e, sp, dirn))
            row = {"step": t, "n_editable": len(todo), "per_alpha": []}
            if not todo:
                rows.append(row)
                continue
            base_hit = np.array([base[i] == sp["predicted"] for i, _, sp, _ in todo], float)
            row["base_rate"] = float(base_hit.mean())
            for a in alphas:
                jobs_e, jobs_c = [], []
                for i, e, sp, (u, sc) in todo:
                    de = edit_delta(u, sc, a, sp["sign"])
                    dc = random_unit(len(u), (seed, int(e.idx), t)) * float(np.linalg.norm(de))
                    jobs_e.append((e.idx, delta_bank(own[i], t, de), t))
                    jobs_c.append((e.idx, delta_bank(own[i], t, dc), t))
                pe = runner.run(jobs_e, downstream=downstream)
                pc = runner.run(jobs_c, downstream=downstream)
                moved = np.array([answer_option(p, task) == sp["predicted"] for p, (_, _, sp, _) in zip(pe, todo)], float)
                ctrl = np.array([answer_option(p, task) == sp["predicted"] for p, (_, _, sp, _) in zip(pc, todo)], float)
                d, lo, hi = bootstrap_mean_ci(moved - ctrl, seed=seed, level=level)
                row["per_alpha"].append({"alpha": a, "edit_rate": float(moved.mean()), "control_rate": float(ctrl.mean()),
                                         "effect": d, "effect_ci": [lo, hi]})
            rows.append(row)
        out["per_kind"][kind] = rows
    return out


def step_load_bearing(f3_rows, *, min_effect: float = MIN_EFFECT) -> dict:
    """{step: (load_bearing bool, best alpha or None, effect)} from one edit kind's rows (CI level already applied)."""
    res = {}
    for row in f3_rows:
        best = None
        for pa in row.get("per_alpha", []):
            ok = pa["effect"] >= min_effect and pa["effect_ci"][0] > 0
            if ok and (best is None or pa["effect"] > best["effect"]):
                best = pa
        res[row["step"]] = (best is not None, None if best is None else best["alpha"],
                            None if best is None else best["effect"])
    return res


def f1_load_bearing(f1_out, *, min_effect: float = MIN_EFFECT) -> dict:
    """{step: (load_bearing, None, effect)} from run_patching: effect = follow rate minus null-bank follow rate."""
    res = {}
    for row in f1_out.get("per_step", []):
        if row["step"] == "all":
            continue
        ok = row["effect_vs_null"] >= min_effect and row["effect_ci95"][0] > 0
        res[row["step"]] = (bool(ok), None, row["effect_vs_null"])
    return res


def load_bearing(decoded: dict, weights: dict, lb: dict) -> dict:
    """Coverage, load-bearing fraction (both definitions) and the registration 7.9 label from per-step flags.
    decoded / lb: {step: bool} (lb may hold tuples from step_load_bearing); weights: {step: n_examples}."""
    steps = sorted(weights)
    w = np.array([float(weights[s]) for s in steps])
    dec = np.array([bool(decoded.get(s, False)) for s in steps])
    lbv = np.array([bool(lb[s][0] if isinstance(lb.get(s), tuple) else lb.get(s, False)) for s in steps])
    tot = w.sum()
    cov = float((w * dec).sum() / tot) if tot else 0.0
    lbd = float((w * dec * lbv).sum() / (w * dec).sum()) if (w * dec).sum() else 0.0
    lba = float((w * dec * lbv).sum() / tot) if tot else 0.0
    if cov >= 0.5 and lbd >= 0.5:
        label = "explained"
    elif cov >= 0.25 and lbd >= 0.25:
        label = "partly explained"
    else:
        label = "unexplained"
    return {"coverage": cov, "load_bearing_fraction": lbd, "load_bearing_over_all_steps": lba, "label": label,
            "per_step": [{"step": s, "weight": int(weights[s]), "decoded": bool(d), "load_bearing": bool(x)}
                         for s, d, x in zip(steps, dec, lbv)]}


__all__ = ["ALPHA_GRID", "MIN_EFFECT", "SIGMA_K", "edit_delta", "random_unit", "run_edits", "step_load_bearing",
           "f1_load_bearing", "load_bearing"]
