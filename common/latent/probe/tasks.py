"""Task adapters for the faithfulness interventions: answer-option parsing, minimal pairs (F1) and edit specs (F3).

Tasks: "prosqa" (Coconut ProsQA format: question = edge sentences + "Is <root> a <x> or <y>?", answer
"<root> is a <target>."), "reach" (the synthetic positive control of probe/synthetic.py: edge list + "is X reachable
from Y", answer yes / no) and "gsm" (no pairs, no frontier edits).

Minimal-pair rules (F1), all keeping the graph (edges, idx_to_symbol) byte-identical so only the query differs:
  * prosqa "target_swap" (the operator's rule, the default): same graph and root; B's target T' is another node at the
    same BFS depth from the root as A's target T (leaves preferred, as released targets are leaves; ties broken by a
    generator seeded with the record idx); B keeps A's neg_target N, which stays unreachable, in the same option
    slot, so B's question differs from A's in one word (T -> T') and the correct answer flips T -> T'. Records with
    no such node are skipped. Note: the frontier F_t is identical for A and B, so a model whose latents carry only
    the frontier is predicted to "stay" under this patch; "follow" means the latent carries the queried target.
  * prosqa "root_swap" (ours, reported next to it): same graph and option list; B's root R' is another entity from
    which N is reachable and T is not (R' with depth(N) == depth(T) preferred), so target and neg_target swap roles
    and the answer flips T -> N. Here the frontier differs between A and B, so a frontier-carrying latent is
    predicted to "follow".
  * reach "target_swap": same graph and source; B's query X' has the opposite reachability (yes <-> no; an
    unreachable X' with a parent preferred, as in the generator); reach "root_swap": same graph and query; B's
    source Y' gives the opposite answer.
Outcome of a patch: "follows" if A's patched answer option equals B's correct option, "stays" if it equals A's,
"breaks" otherwise (no parsable option, or a third option).

F3 edit specs (per example: symbol name, sign, predicted option):
  * prosqa "add_neg" (primary): add the "N in frontier" direction; predicted answer N. "remove_target": subtract the
    "T in frontier" direction; predicted answer N.
  * reach "toggle_query" (primary): subtract "X in frontier" if X is reachable (predicted "no"), add it otherwise
    (predicted "yes").
"""
from __future__ import annotations

import random
import re

from latent.data import Example
from latent.probe.extract import bfs_dist, detect_task

PAIR_RULES = {"prosqa": ("target_swap", "root_swap"), "reach": ("target_swap", "root_swap"), "gsm": ()}
EDIT_KINDS = {"prosqa": ("add_neg", "remove_target"), "reach": ("toggle_query",), "gsm": ()}
PRIMARY_EDIT = {"prosqa": "add_neg", "reach": "toggle_query"}
_WORD = re.compile(r"[A-Za-z_][\w-]*")


def answer_option(pred: str, task: str) -> str | None:
    """The answer option a prediction names: prosqa, the last word ("Sally is a sterpus." -> "sterpus");
    reach, "yes" / "no" if it is the last word; gsm, the normalised string."""
    words = _WORD.findall(pred or "")
    if task == "prosqa":
        return words[-1] if words else None
    if task == "reach":
        return words[-1].lower() if words and words[-1].lower() in ("yes", "no") else None
    return (pred or "").strip()


def gold_option(rec: dict) -> str:
    """The correct option of a target record (extract.example_targets) or an Example's meta-bearing dict."""
    if rec["task"] == "prosqa":
        return rec["symbols"][rec["target"]]
    if rec["task"] == "reach":
        return "yes" if rec["reachable"] else "no"
    return rec["answer"]


def _is_entity(n, edges):
    has_parent = {int(v) for _, v in edges}
    return [i not in has_parent for i in range(n)]


def _sentence(names, ent, u, v):
    return f"{names[u]} is a {names[v]}." if ent[u] else f"Every {names[u]} is a {names[v]}."


def _proof(names, edges, src, dst):
    """Shortest-path proof sentences src -> dst (BFS parents, smallest-index parent first)."""
    n = len(names)
    dist = bfs_dist(n, edges, src)
    ent = _is_entity(n, edges)
    path, cur = [dst], dst
    while cur != src:
        cur = min(u for u, v in edges if v == cur and dist.get(u) == dist[cur] - 1)
        path.append(cur)
    path = path[::-1]
    return [_sentence(names, ent, path[i], path[i + 1]) for i in range(len(path) - 1)]


def _prosqa_question(q, root_name, x, y):
    body, sep, _ = q.rpartition(" Is ")
    if not sep:
        raise ValueError("not a ProsQA question (no ' Is ' query)")
    return f"{body} Is {root_name} a {x} or {y}?"


def _prosqa_slots(ex):
    """(first, second) option names as they appear in A's query "Is <root> a <x> or <y>?"."""
    query = ex.question.rpartition(" Is ")[2].strip().rstrip("?").strip()
    opts = query.split(" a ", 1)[1].split(" or ")
    return opts[0].strip(), opts[-1].strip()


def make_pair(ex: Example, rule: str, seed: int = 0) -> Example | None:
    """B for example A under `rule` (module docstring), or None if A has no valid partner. B carries A's idx."""
    task = detect_task(ex)
    if rule not in PAIR_RULES.get(task, ()):
        raise ValueError(f"pair rule {rule!r} is not defined for task {task!r}")
    rng = random.Random(f"{seed}:{ex.idx}:{rule}")
    m = ex.meta
    names, edges, root = list(m["idx_to_symbol"]), [tuple(map(int, e)) for e in m["edges"]], int(m["root"])
    n = len(names)
    out_deg = {u for u, _ in edges}
    dist = bfs_dist(n, edges, root)
    if task == "prosqa":
        T, N = int(m["target"]), int(m["neg_target"])
        first, _second = _prosqa_slots(ex)
        if rule == "target_swap":
            cands = sorted(v for v, d in dist.items() if d == dist[T] and v not in (T, root))
            leaves = [v for v in cands if v not in out_deg]
            if not cands:
                return None
            T2 = rng.choice(leaves or cands)
            x, y = (names[T2], names[N]) if first == names[T] else (names[N], names[T2])
            meta = {**m, "target": T2, "neg_target": N}
            return Example(question=_prosqa_question(ex.question, names[root], x, y),
                           answer=f"{names[root]} is a {names[T2]}.", steps=_proof(names, edges, root, T2),
                           idx=ex.idx, meta=meta)
        ent = _is_entity(n, edges)
        cands = []
        for r in range(n):
            if not ent[r] or r == root:
                continue
            d2 = bfs_dist(n, edges, r)
            if N in d2 and T not in d2:
                cands.append((0 if d2[N] == dist[T] else 1, r))
        if not cands:
            return None
        best = min(c[0] for c in cands)
        R2 = rng.choice(sorted(r for k, r in cands if k == best))
        x, y = (names[T], names[N]) if first == names[T] else (names[N], names[T])
        meta = {**m, "root": R2, "target": N, "neg_target": T}
        return Example(question=_prosqa_question(ex.question, names[R2], x, y), answer=f"{names[R2]} is a {names[N]}.",
                       steps=_proof(names, edges, R2, N), idx=ex.idx, meta=meta)
    # reach
    from latent.probe.synthetic import reach_example  # local import: synthetic imports this module's users

    X, reach = int(m["query"]), bool(m["reachable"])
    if rule == "target_swap":
        if reach:
            un = sorted(v for v in range(n) if v not in dist)
            has_parent = {v for _, v in edges}
            cands = [v for v in un if v in has_parent] or un
        else:
            cands = sorted(v for v in dist if v != root)
        if not cands:
            return None
        X2 = rng.choice(cands)
        return reach_example(names, edges, root, X2, idx=ex.idx, n_steps=len(ex.steps))
    cands = []
    for r in range(n):
        if r in (root, X):
            continue
        if (X in bfs_dist(n, edges, r)) != reach:
            cands.append(r)
    if not cands:
        return None
    return reach_example(names, edges, rng.choice(sorted(cands)), X, idx=ex.idx, n_steps=len(ex.steps))


def edit_specs(rec: dict) -> list[dict]:
    """F3 edits for one target record: [{kind, symbol (node idx), name, sign, predicted}]."""
    task = rec["task"]
    names = rec.get("symbols")
    if task == "prosqa":
        T, N = rec["target"], rec["neg_target"]
        return [{"kind": "add_neg", "symbol": N, "name": names[N], "sign": 1.0, "predicted": names[N]},
                {"kind": "remove_target", "symbol": T, "name": names[T], "sign": -1.0, "predicted": names[N]}]
    if task == "reach":
        X = rec["query"]
        if rec["reachable"]:
            return [{"kind": "toggle_query", "symbol": X, "name": names[X], "sign": -1.0, "predicted": "no"}]
        return [{"kind": "toggle_query", "symbol": X, "name": names[X], "sign": 1.0, "predicted": "yes"}]
    return []


__all__ = ["PAIR_RULES", "EDIT_KINDS", "PRIMARY_EDIT", "answer_option", "gold_option", "make_pair", "edit_specs"]
