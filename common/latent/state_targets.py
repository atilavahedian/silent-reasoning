"""C2 state targets: pure functions of a training record (docs/cards/C2_verifiable_state_supervision.md, "Mechanism",
"Targets per task" and "Null"; registration A3.4 C2 with the critic's correction of the card's leak answer).

* ProsQA / ProsQA-hard: the frontier F_t = the record's nodes reachable from `root` in at most t "is a" hops
  (breadth-first search over `edges`), the root itself excluded (it is the queried entity, at distance 0, not a concept
  reached), for t = 1 .. len(steps). The decoder head is a multi-label sigmoid over the record's symbols
  (`idx_to_symbol`), padded to the batch's largest symbol count and masked (padding and the root's slot).
* GSM8k-Aug: the numeric result of step t = the text after the last "=" of the t-th "<<expr=c>>" step string,
  tokenised with the model's tokenizer (`encode(c, add_special_tokens=False)`); a result longer than 4 tokens, an
  empty result or a step that does not parse has no target (masked).
* Which latent carries step t: latent t (1-based) for c_thought 1; the second latent of pair t for c_thought 2; in
  general latent t * c_thought. Steps beyond the current stage's latents have no target; stage 0 has none
  (`supervised_steps`).
* Targets come from the record's steps and edges only, from the training split only; the answer field is never read
  to build a target. As the critic corrected (A3.4), state targets do contain the answer in most records (the last GSM
  step's result equals the final answer in about 95% of records; the ProsQA frontier at the proof depth contains the
  target): `answer_in_last_target` reports the fraction per run, and attribution rests on the C2-shuffled null.
* C2-shuffled: `shuffled_donors` gives a fixed seeded derangement of the training set within step-count classes
  (example i is supervised with record pi(i)'s targets, pi(i) != i); a class with a single member is merged into the
  class with the nearest step count (ties to the smaller count), the only case where donor and record differ in step
  count (the donor's own step count then bounds its targets).
"""
from __future__ import annotations

import random
import re
from collections import deque

GSM_STEP = re.compile(r"^<<(?P<expr>.*)=(?P<res>[^=]*)>>$")
MAX_RESULT_TOKENS = 4
KINDS = ("prosqa", "gsm")


def _get(rec, key, default=None):
    if isinstance(rec, dict):
        return rec.get(key, default)
    if key in ("question", "answer", "steps", "idx"):
        return getattr(rec, key)
    return rec.meta.get(key, default)


def task_kind(rec) -> str:
    """"prosqa" for a record with edges, root and idx_to_symbol (ProsQA format), else "gsm"."""
    if all(_get(rec, k) is not None for k in ("edges", "root", "idx_to_symbol")):
        return "prosqa"
    return "gsm"


def bfs_distances(n: int, edges, root: int) -> dict[int, int]:
    """Hop distance from root over the directed edges (u, v) = "u is a v" (mirrors latent.prosqa_gen.bfs_dist)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))
    dist, q = {int(root): 0}, deque([int(root)])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def prosqa_frontiers(edges, root: int, n_symbols: int, n_steps: int) -> list[list[int]]:
    """C2 card: F_t for t = 1..n_steps, each the sorted node indices v != root with 1 <= dist(root, v) <= t."""
    dist = bfs_distances(n_symbols, edges, root)
    return [sorted(v for v, d in dist.items() if v != int(root) and d <= t) for t in range(1, n_steps + 1)]


def gsm_step_result(step: str) -> str | None:
    """The text after the last "=" of a "<<expr=c>>" step, stripped; None if the step does not have that form."""
    m = GSM_STEP.match(step.strip())
    if not m:
        return None
    res = m.group("res").strip()
    return res if res else None


def gsm_result_tokens(result: str | None, tokenizer, max_tokens: int = MAX_RESULT_TOKENS) -> list[int] | None:
    """The result tokenised with the model's tokenizer (no special tokens); None when absent or longer than
    max_tokens (the card's 4 parallel heads)."""
    if result is None:
        return None
    ids = list(tokenizer.encode(result, add_special_tokens=False))
    return ids if 0 < len(ids) <= max_tokens else None


def record_targets(rec, tokenizer, max_tokens: int = MAX_RESULT_TOKENS) -> dict:
    """All per-step targets of one record (a pure function of the record and the tokenizer).
    prosqa: {kind, n_steps, n_symbols, root, symbols, frontiers, symbol_tokens (encode(" " + name) per symbol)}.
    gsm: {kind, n_steps, results (str | None per step), tokens (list[int] | None per step)}."""
    steps = list(_get(rec, "steps"))
    if task_kind(rec) == "prosqa":
        names = list(_get(rec, "idx_to_symbol"))
        root = int(_get(rec, "root"))
        return {"kind": "prosqa", "n_steps": len(steps), "n_symbols": len(names), "root": root, "symbols": names,
                "frontiers": prosqa_frontiers(_get(rec, "edges"), root, len(names), len(steps)),
                "symbol_tokens": [list(tokenizer.encode(" " + s, add_special_tokens=False)) for s in names]}
    results = [gsm_step_result(s) for s in steps]
    return {"kind": "gsm", "n_steps": len(steps), "results": results,
            "tokens": [gsm_result_tokens(r, tokenizer, max_tokens) for r in results]}


def build_targets(examples, tokenizer, max_tokens: int = MAX_RESULT_TOKENS) -> dict:
    """idx -> record_targets for a training set; raises if the set mixes ProsQA and GSM records."""
    out = {_get(ex, "idx", i): record_targets(ex, tokenizer, max_tokens) for i, ex in enumerate(examples)}
    kinds = {t["kind"] for t in out.values()}
    if len(kinds) > 1:
        raise ValueError(f"training set mixes target kinds {sorted(kinds)}")
    return out


def supervised_steps(n_steps: int, n_latent: int, c_thought: int) -> list[tuple[int, int]]:
    """C2 card "Steps beyond the number of latents at the current stage have no target": the (step t, latent index)
    pairs, t 1-based and the latent index 0-based (t * c_thought - 1), for t <= min(n_steps, n_latent // c_thought).
    Stage 0 (n_latent 0) gives none."""
    c = max(1, int(c_thought))
    return [(t, t * c - 1) for t in range(1, min(int(n_steps), int(n_latent) // c) + 1)]


def _norm(s: str) -> str:
    return s.replace(",", "").strip()


def answer_in_last_target(tgt: dict, answer: str, n_latent_steps: int) -> bool | None:
    """Whether the last supervised target (step min(n_steps, n_latent_steps)) contains `answer`: the GSM result
    equals the normalised answer, or the ProsQA frontier holds the symbol the answer names ("<root> is a <target>.",
    compared by name, so a C2-shuffled donor's frontier can be checked against the record's own answer). None if
    nothing is supervised. Reporting only (registration A3.4 C2): the answer is never used to build a target."""
    t = min(tgt["n_steps"], int(n_latent_steps))
    if t <= 0:
        return None
    if tgt["kind"] == "gsm":
        r = tgt["results"][t - 1]
        return r is not None and _norm(r) == _norm(answer)
    name = answer.strip().rstrip(".").rsplit(" ", 1)[-1]
    return name in {tgt["symbols"][v] for v in tgt["frontiers"][t - 1]}


def target_stats(examples, targets: dict, n_latent_steps: int, used: dict | None = None) -> dict:
    """Run-start statistics (A3.4 C2): records, supervised steps at the final regime, masked GSM steps, and the
    fraction of records whose last supervised target contains the answer, for the own targets and (C2-shuffled) for
    the targets actually used against the record's own answer."""
    own, use, n_rec, n_sup, n_masked = [], [], 0, 0, 0
    for i, ex in enumerate(examples):
        idx = _get(ex, "idx", i)
        tgt = targets[idx]
        n_rec += 1
        t = min(tgt["n_steps"], n_latent_steps)
        n_sup += t
        if tgt["kind"] == "gsm":
            n_masked += sum(1 for k in tgt["tokens"][:t] if k is None)
        a = answer_in_last_target(tgt, _get(ex, "answer"), n_latent_steps)
        if a is not None:
            own.append(a)
        if used is not None:
            b = answer_in_last_target(used[idx], _get(ex, "answer"), n_latent_steps)
            if b is not None:
                use.append(b)
    out = {"records": n_rec, "kind": next(iter(targets.values()))["kind"] if targets else None,
           "final_regime_latent_steps": int(n_latent_steps), "supervised_steps_final_regime": n_sup,
           "gsm_masked_steps_final_regime": n_masked,
           "answer_in_last_target_frac": (sum(own) / len(own)) if own else None}
    if used is not None:
        out["answer_in_last_used_target_frac"] = (sum(use) / len(use)) if use else None
    return out


def _derangement(n: int, rng: random.Random) -> list[int]:
    """Sattolo's algorithm (as latent.eval._derangement): a single n-cycle, no fixed point for n >= 2."""
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        p[i], p[j] = p[j], p[i]
    return p


def shuffled_donors(step_counts: dict, seed: int) -> dict:
    """C2-shuffled null: idx -> donor idx, a fixed seeded derangement within step-count classes (pi(i) != i);
    singleton classes are merged into the nearest class (ties to the smaller step count)."""
    classes: dict[int, list] = {}
    for idx in sorted(step_counts):
        classes.setdefault(int(step_counts[idx]), []).append(idx)
    for n in sorted(k for k, v in classes.items() if len(v) == 1):
        if len(classes) == 1 or len(classes.get(n, [])) != 1:
            continue
        others = [k for k in classes if k != n]
        nearest = min(others, key=lambda k: (abs(k - n), k))
        classes[nearest] = sorted(classes[nearest] + classes.pop(n))
    rng = random.Random(seed)
    donor = {}
    for n in sorted(classes):
        members = classes[n]
        if len(members) < 2:
            raise ValueError("cannot derange a training set with a single record")
        perm = _derangement(len(members), rng)
        donor.update({members[k]: members[perm[k]] for k in range(len(members))})
    assert all(donor[i] != i for i in donor)
    return donor


__all__ = ["task_kind", "bfs_distances", "prosqa_frontiers", "gsm_step_result", "gsm_result_tokens", "record_targets",
           "build_targets", "supervised_steps", "answer_in_last_target", "target_stats", "shuffled_donors",
           "MAX_RESULT_TOKENS", "GSM_STEP"]
