"""One JSON + Markdown readout report per checkpoint: R1, R2 (and R3 when a decoder is given), F1, F2, F3, coverage
and load-bearing fraction with CIs (plan 6.5, registration 7.9). `run_readout` is the whole pipeline; the CLI
code/run_probes.py and probe/synthetic.py call it."""
from __future__ import annotations

import json
import os
import time

import torch

from latent.probe.edit import ALPHA_GRID, MIN_EFFECT, SIGMA_K, f1_load_bearing, load_bearing, run_edits, \
    step_load_bearing
from latent.probe.extract import extract, save_extraction
from latent.probe.linear import logit_lens, score_readout, train_frontier_probes, train_gsm_probes
from latent.probe.patch import run_patching, scrambled_drop
from latent.probe.tasks import EDIT_KINDS, PAIR_RULES, PRIMARY_EDIT

REPORT_FIELDS = ("R1", "R2", "F1", "F2", "F3", "load_bearing")

DEFINITIONS = {
    "frontier": "F_t = concepts reachable from the root in at most t hops (root excluded); latent j carries "
                "t = j // c_thought + 1 (card C2)",
    "R2": "per-symbol-name L2 logistic probes on the standardised latent, pooled over steps, k-fold over examples; "
          "GSM: multinomial probe for the first token of step t's result",
    "decoded": "step score > shuffled-label score + sigma_k * sqrt(s (1 - s) / n_examples)",
    "F1": "A's latent at step t replaced by B's (minimal pair, same graph); follows / stays / breaks by answer option",
    "F3": "h_t + sign * alpha * sigma_u * u (u the unit out-of-fold probe direction) vs a same-norm random direction",
    "load_bearing_step": "some alpha with effect >= min_effect and a paired bootstrap interval (Bonferroni over the "
                         "alpha grid) excluding 0",
    "load_bearing_fraction": "decoded and load-bearing steps / decoded steps (registration 7.9), example-weighted",
    "load_bearing_over_all_steps": "decoded and load-bearing steps / all steps (lead's brief), example-weighted",
    "label": "explained iff coverage >= 0.5 and load_bearing_fraction >= 0.5; partly explained iff both >= 0.25; "
             "else unexplained",
}


def _strip(obj):
    """Drop per-example indicator lists (kept in the JSON) for the Markdown view."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in ("moved", "control")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def run_readout(lm, tokenizer, examples, *, task: str, n_latent: int, c_thought: int = 1, batch_size: int = 32,
                device: str = "cpu", max_new_tokens: int = 128, k_folds: int = 5, l2: float = 1e-2,
                alphas=ALPHA_GRID, min_effect: float = MIN_EFFECT, sigma_k: float = SIGMA_K, pair_rules=None,
                edit_kinds=None, downstream: str = "frozen", lb_intervention: str = "F3", r3=None,
                decoded_by: str = "R2", seed: int = 0, out_dir: str | None = None, meta: dict | None = None,
                skip=()) -> dict:
    """Run every readout and intervention on `examples` and assemble the report dict (fields REPORT_FIELDS)."""
    t0 = time.time()
    lm.eval()
    common = dict(batch_size=batch_size, device=device, max_new_tokens=max_new_tokens)
    ext = extract(lm, tokenizer, examples, n_latent=n_latent, c_thought=c_thought, task=task, **common)
    rep = {"meta": {**(meta or {}), "task": task, "n_examples": len(examples), "n_latent": n_latent,
                    "c_thought": c_thought, "max_new_tokens": max_new_tokens, "seed": seed},
           "thresholds": {"k_folds": k_folds, "l2": l2, "alphas": list(alphas), "min_effect": min_effect,
                          "sigma_k": sigma_k, "bonferroni_over_alphas": True, "downstream": downstream,
                          "lb_intervention": lb_intervention, "decoded_by": decoded_by},
           "definitions": DEFINITIONS, "accuracy": ext.info.get("accuracy")}
    if out_dir:
        rep["meta"]["latents"] = save_extraction(ext, out_dir)
    rep["R1"] = logit_lens(lm, tokenizer, ext, seed=seed) if "R1" not in skip else {"skipped": True}

    graph = task in ("prosqa", "reach")
    probes = None
    if graph:
        probes = train_frontier_probes(ext, k=k_folds, l2=l2, seed=seed, sigma_k=sigma_k)
        rep["R2"] = probes.metrics
    else:
        rep["R2"] = train_gsm_probes(ext, k=k_folds, l2=l2, seed=seed, sigma_k=sigma_k)
    rep["R3"] = score_readout(r3, ext, seed=seed, sigma_k=sigma_k) if r3 is not None else {
        "skipped_reason": "no R3 decoder given (the C2 decoder plugs in via linear.StateReadout)"}

    own = [torch.as_tensor(ext.latents[i, :n_latent]) for i in range(len(examples))]
    if graph and "F1" not in skip:
        rules = tuple(pair_rules) if pair_rules else PAIR_RULES[task]
        rep["F1"] = {"rules": list(rules), "primary_rule": rules[0], "by_rule": {
            r: run_patching(lm, tokenizer, examples, task=task, rule=r, n_latent=n_latent, own_latents=own,
                            downstream=downstream, seed=seed, **common) for r in rules}}
    else:
        rep["F1"] = {"skipped_reason": "minimal pairs are defined for prosqa-format tasks only" if not graph
                     else "skipped"}
    rep["F2"] = scrambled_drop(lm, tokenizer, examples, n_latent=n_latent, seed=seed, **common) \
        if "F2" not in skip else {"skipped": True}
    if graph and "F3" not in skip:
        kinds = tuple(edit_kinds) if edit_kinds else EDIT_KINDS[task]
        rep["F3"] = run_edits(lm, tokenizer, examples, ext, probes, n_latent=n_latent, alphas=alphas, kinds=kinds,
                              downstream=downstream, seed=seed, **common)
        rep["F3"]["primary_kind"] = PRIMARY_EDIT[task]
    else:
        rep["F3"] = {"skipped_reason": "frontier edits are defined for prosqa-format tasks only (no registered "
                                       "predicted answer for a GSM intermediate-result edit)" if not graph
                     else "skipped"}

    readout = rep["R3"] if decoded_by == "R3" and r3 is not None else rep["R2"]
    decoded = {s["step"]: s["decoded"] for s in readout["per_step"]}
    weights = {s["step"]: s["n_examples"] for s in readout["per_step"]}
    lb = None
    if lb_intervention == "F3" and "per_kind" in rep["F3"]:
        lb = step_load_bearing(rep["F3"]["per_kind"][rep["F3"]["primary_kind"]], min_effect=min_effect)
    elif lb_intervention == "F1" and "by_rule" in rep["F1"]:
        lb = f1_load_bearing(rep["F1"]["by_rule"][rep["F1"]["primary_rule"]], min_effect=min_effect)
    if lb is None:
        res = load_bearing(decoded, weights, {})
        res.update(load_bearing_fraction=None, load_bearing_over_all_steps=None, label="unexplained",
                   note=f"no {lb_intervention} intervention for task {task}: load-bearing is not measurable here")
    else:
        res = load_bearing(decoded, weights, lb)
        for row in res["per_step"]:
            v = lb.get(row["step"])
            row["alpha"], row["effect"] = (v[1], v[2]) if v else (None, None)
    secondary = {}
    if graph and "per_kind" in rep["F3"] and "by_rule" in rep["F1"]:
        other = "F1" if lb_intervention == "F3" else "F3"
        lb2 = (f1_load_bearing(rep["F1"]["by_rule"][rep["F1"]["primary_rule"]], min_effect=min_effect)
               if other == "F1" else
               step_load_bearing(rep["F3"]["per_kind"][rep["F3"]["primary_kind"]], min_effect=min_effect))
        s2 = load_bearing(decoded, weights, lb2)
        secondary = {"intervention": other, **{k: s2[k] for k in ("coverage", "load_bearing_fraction",
                                                                  "load_bearing_over_all_steps", "label")}}
    res["secondary"] = secondary
    res["intervention"], res["readout"] = lb_intervention, decoded_by
    rep["load_bearing"] = res
    rep["meta"]["seconds"] = time.time() - t0
    return rep


def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return f"[{_fmt(x[0], nd)}, {_fmt(x[1], nd)}]"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def to_markdown(rep: dict) -> str:
    """Human-readable summary of a report dict."""
    m, lb = rep["meta"], rep["load_bearing"]
    L = [f"# Readout report: {m.get('checkpoint', '?')}", "",
         f"task {m['task']}, split {m.get('split', '?')}, {m['n_examples']} examples, {m['n_latent']} latents "
         f"(c_thought {m['c_thought']}), accuracy {_fmt(rep.get('accuracy'))}", "",
         "## Headline", "",
         f"- coverage (decoded by {lb['readout']}): {_fmt(lb['coverage'])}",
         f"- load-bearing fraction ({lb['intervention']}, among decoded steps): {_fmt(lb['load_bearing_fraction'])}",
         f"- load-bearing over all steps: {_fmt(lb['load_bearing_over_all_steps'])}",
         f"- label: **{lb['label']}**"]
    if lb.get("secondary"):
        s = lb["secondary"]
        L.append(f"- secondary ({s['intervention']}): coverage {_fmt(s['coverage'])}, load-bearing fraction "
                 f"{_fmt(s['load_bearing_fraction'])}, label {s['label']}")
    if lb.get("note"):
        L.append(f"- note: {lb['note']}")
    L += ["", "| step | n | decoded | load-bearing | alpha | effect |", "|---|---|---|---|---|---|"]
    L += [f"| {r['step']} | {r['weight']} | {r['decoded']} | {r['load_bearing']} | {_fmt(r.get('alpha'))} | "
          f"{_fmt(r.get('effect'))} |" for r in lb["per_step"]]
    r2 = rep["R2"]
    L += ["", f"## R2 linear probes (metric {r2['metric']})", ""]
    if r2["metric"] == "f1":
        L += ["| step | n | F1 | CI95 | shuffled F1 | always-in F1 | decoded |", "|---|---|---|---|---|---|---|"]
        L += [f"| {s['step']} | {s['n_examples']} | {_fmt(s['f1'])} | {_fmt(s['f1_ci95'])} | {_fmt(s['shuffled_f1'])}"
              f" | {_fmt(s['chance_f1_always_in'])} | {s['decoded']} |" for s in r2["per_step"]]
    else:
        L += ["| step | n | accuracy | CI95 | shuffled | majority | decoded |", "|---|---|---|---|---|---|---|"]
        L += [f"| {s['step']} | {s['n_examples']} | {_fmt(s['accuracy'])} | {_fmt(s['accuracy_ci95'])} | "
              f"{_fmt(s['shuffled_accuracy'])} | {_fmt(s.get('chance_majority'))} | {s['decoded']} |"
              for s in r2["per_step"]]
    if "per_step" in rep.get("R1", {}):
        L += ["", f"## R1 logit lens (top-{rep['R1']['k']}; reported, never counts)", "",
              "| step | next-step entity in top-k | following-step entity | frontier tokens in top-k |",
              "|---|---|---|---|"]
        L += [f"| {s['step']} | {_fmt(s['next_step_entity_topk'])} | {_fmt(s['following_step_entity_topk'])} | "
              f"{_fmt(s['frontier_tokens_in_topk'])} |" for s in rep["R1"]["per_step"]]
    f2 = rep["F2"]
    if "drop" in f2:
        L += ["", "## F2 scrambled drop", "",
              f"clean {_fmt(f2['clean_accuracy'])} {_fmt(f2['clean_accuracy_ci95'])}, scrambled "
              f"{_fmt(f2['scrambled_accuracy'])} {_fmt(f2['scrambled_accuracy_ci95'])}, drop {_fmt(f2['drop'])} "
              f"{_fmt(f2['drop_ci95'])} (paired)"]
    if "by_rule" in rep["F1"]:
        L += ["", f"## F1 counterfactual patching (downstream {rep['thresholds']['downstream']})"]
        for rule, f1 in rep["F1"]["by_rule"].items():
            L += ["", f"rule `{rule}`: {f1['n_pairs']} pairs of {f1['n_records']} records", ""]
            if not f1["per_step"]:
                L.append(f"- {f1.get('skipped_reason', 'no pairs')}")
                continue
            L += ["| step | follows | CI95 | stays | breaks | effect vs null |", "|---|---|---|---|---|---|"]
            L += [f"| {s['step']} | {_fmt(s['follows']['rate'])} | {_fmt(s['follows']['ci95'])} | "
                  f"{_fmt(s['stays']['rate'])} | {_fmt(s['breaks']['rate'])} | {_fmt(s['effect_vs_null'])} |"
                  for s in f1["per_step"]]
    if "per_kind" in rep["F3"]:
        f3 = rep["F3"]
        L += ["", f"## F3 probe-direction edits (alphas {f3['alphas']}, CI level {_fmt(f3['ci_level'])})"]
        for kind, rows in f3["per_kind"].items():
            L += ["", f"edit `{kind}`", "", "| step | editable | alpha | edit rate | control rate | effect | CI |",
                  "|---|---|---|---|---|---|---|"]
            for r in rows:
                for pa in r.get("per_alpha", []):
                    L.append(f"| {r['step']} | {r['n_editable']} | {pa['alpha']} | {_fmt(pa['edit_rate'])} | "
                             f"{_fmt(pa['control_rate'])} | {_fmt(pa['effect'])} | {_fmt(pa['effect_ci'])} |")
    L += ["", "## Definitions", ""] + [f"- {k}: {v}" for k, v in rep["definitions"].items()]
    L += ["", "## Thresholds", "", "```", json.dumps(rep["thresholds"]), "```", ""]
    return "\n".join(L)


def write_report(rep: dict, out_dir: str) -> dict:
    """probe_report.json (everything, including per-example indicators) and probe_report.md under out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    pj, pm = os.path.join(out_dir, "probe_report.json"), os.path.join(out_dir, "probe_report.md")
    with open(pj, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    with open(pm, "w") as f:
        f.write(to_markdown(_strip(rep)))
    return {"json": pj, "md": pm}


__all__ = ["REPORT_FIELDS", "DEFINITIONS", "run_readout", "to_markdown", "write_report"]
