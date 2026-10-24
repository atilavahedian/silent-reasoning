"""Extraction of fed-back latent vectors and per-step state targets (plan 6.5; card C2 "Targets per task").

`extract` runs a trained LatentLM over a split through `latent.eval.evaluate(return_latents=True)` (batched, any
device) and collects, per example and per latent step j, the fed-back vector (the input embedding at latent position
j, i.e. the last-layer hidden state of position j - 1) together with the example's targets:
  * ProsQA-format records (prosqa, and the synthetic reach task): the frontier F_t, t = j // c_thought + 1, where
    F_t = the concepts reachable from the root in at most t "is a" hops (card C2), computed by `frontier_sets` from
    `edges`, `root`, `idx_to_symbol`; plus the answer correctness and the parsed prediction.
  * GSM records: the result string of step t (the text after "=" in the t-th "<<a op b = c>>" step) and the first
    token id of that string; with c_thought = 2 only the second latent of each pair carries step t (card C2).
Saved as `latents.npz` (latents [N, L, d] float32 zero-padded, n_latent [N], idx [N]) plus `index.json` (one record
per example with every target, JSON-serialisable).

Conventions fixed here (documented in the report):
  * The root itself is never in F_t (it is an entity, not a concept, and is trivially "reachable" in 0 hops).
  * Probe candidates of a record = every node with at least one parent, except the root (nodes without a parent are
    never reachable, so including them only adds trivial negatives).
"""
from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass, field

import numpy as np

_GSM_STEP = re.compile(r"<<(.*?)=(.*?)>>")
_ENTITY = re.compile(r"is an? ([A-Za-z_][\w-]*)\s*\.?\s*$")


def bfs_dist(n_nodes: int, edges, src: int) -> dict[int, int]:
    """Shortest hop count from src to every reachable node of the directed graph (src at 0)."""
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[int(u)].append(int(v))
    dist, q = {int(src): 0}, deque([int(src)])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def frontier_sets(edges, root: int, n_nodes: int, max_t: int) -> list[frozenset]:
    """Card C2 target: [F_1, ..., F_max_t], F_t = {v != root : v reachable from root in at most t hops}.

    Computed by iterated one-hop expansion (not BFS), so the BFS test is an independent check."""
    children = [[] for _ in range(n_nodes)]
    for u, v in edges:
        children[int(u)].append(int(v))
    reached, layer, out = {int(root)}, {int(root)}, []
    for _ in range(max_t):
        layer = {w for u in layer for w in children[u]} - reached
        reached |= layer
        out.append(frozenset(reached - {int(root)}))
    return out


def candidate_nodes(edges, root: int, n_nodes: int) -> list[int]:
    """Nodes with at least one parent, root excluded: the per-record probe candidates."""
    has_parent = {int(v) for _, v in edges}
    return [v for v in range(n_nodes) if v in has_parent and v != int(root)]


def latent_hops(j: int, c_thought: int) -> int:
    """Latent index j (0-based) carries the state after hop t = j // c_thought + 1 (card C2, c_thought latents per
    replaced step)."""
    return j // c_thought + 1


def gsm_results(steps) -> list[str | None]:
    """The result string of every "<<a op b = c>>" step (text after the last "=" inside <<...>>), None if absent."""
    out = []
    for s in steps:
        m = _GSM_STEP.search(s)
        if m is None:
            out.append(None)
            continue
        inner = s[m.start() + 2:m.end() - 2]
        out.append(inner.rsplit("=", 1)[1].strip() if "=" in inner else None)
    return out


def step_entity(step: str) -> str | None:
    """ProsQA step "X is a Y." / "Every X is a Y." -> "Y" (the node the step reaches)."""
    m = _ENTITY.search(step.strip())
    return m.group(1) if m else None


def first_token(tokenizer, text: str) -> int | None:
    """First token id of `text` (None for empty), add_special_tokens=False."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if ids else None


def name_tokens(tokenizer, name: str) -> list[int]:
    """Candidate first-token ids for a symbol as it can appear in text: " name" (mid-sentence) and "name"."""
    out = []
    for s in (" " + name, name):
        t = first_token(tokenizer, s)
        if t is not None and t not in out:
            out.append(t)
    return out


def detect_task(example) -> str:
    """'prosqa' if the record has a ProsQA graph with target, 'reach' if it has a graph with a query, else 'gsm'."""
    meta = example.meta if hasattr(example, "meta") else example
    if "edges" in meta and "target" in meta:
        return "prosqa"
    if "edges" in meta and "query" in meta:
        return "reach"
    return "gsm"


def example_targets(example, n_latent: int, c_thought: int, tokenizer, task: str | None = None) -> dict:
    """Per-example target record (JSON-serialisable) for latent steps j = 0..n_latent-1."""
    task = task or detect_task(example)
    meta = example.meta
    rec = {"idx": int(example.idx), "task": task, "n_latent": int(n_latent), "c_thought": int(c_thought),
           "n_steps": len(example.steps), "answer": example.answer}
    if task in ("prosqa", "reach"):
        names, edges, root = list(meta["idx_to_symbol"]), [list(map(int, e)) for e in meta["edges"]], int(meta["root"])
        n = len(names)
        max_t = latent_hops(max(n_latent - 1, 0), c_thought)
        fr = frontier_sets(edges, root, n, max_t)
        rec.update(symbols=names, edges=edges, root=root, candidates=candidate_nodes(edges, root, n),
                   frontier=[sorted(fr[latent_hops(j, c_thought) - 1]) for j in range(n_latent)],
                   hops=[latent_hops(j, c_thought) for j in range(n_latent)],
                   dist={str(k): v for k, v in bfs_dist(n, edges, root).items()})
        if task == "prosqa":
            rec.update(target=int(meta["target"]), neg_target=int(meta["neg_target"]),
                       step_entities=[step_entity(s) for s in example.steps])
        else:
            rec.update(query=int(meta["query"]), reachable=bool(meta["reachable"]),
                       step_entities=list(meta.get("step_entities", [None] * len(example.steps))))
    else:
        res = gsm_results(example.steps)
        rec.update(results=res, result_first_token=[None if r is None else first_token(tokenizer, r) for r in res],
                   latent_step=[(j // c_thought) if (j % c_thought == c_thought - 1 and j // c_thought < len(res))
                                else -1 for j in range(n_latent)])
    return rec


@dataclass
class Extraction:
    """Latents [N, L, d] (zero-padded beyond n_latent[i]), n_latent [N], records (example_targets + run fields)."""
    latents: np.ndarray
    n_latent: np.ndarray
    records: list
    info: dict = field(default_factory=dict)

    @property
    def task(self) -> str:
        return self.records[0]["task"] if self.records else self.info.get("task", "")

    def __len__(self):
        return len(self.records)


def extract(lm, tokenizer, examples, *, n_latent: int, c_thought: int = 1, batch_size: int = 32, device: str = "cpu",
            max_new_tokens: int = 128, arm: str = "coconut", task: str | None = None) -> Extraction:
    """Run `latent.eval.evaluate(..., return_latents=True)` with exactly n_latent latents per example and pair every
    example's fed-back vectors with its targets. Returns an Extraction in input order."""
    from latent.eval import evaluate

    res = evaluate(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size, device=device,
                   max_new_tokens=max_new_tokens, return_latents=True)
    by_idx = {r.idx: r for r in res}
    d = next((r.latents.shape[-1] for r in res if r.latents is not None and r.latents.numel()), 0)
    L = max([int(r.latents.shape[0]) for r in res if r.latents is not None] + [0])
    lat = np.zeros((len(examples), L, d), dtype=np.float32)
    nl = np.zeros(len(examples), dtype=np.int64)
    records = []
    for i, ex in enumerate(examples):
        r = by_idx[ex.idx]
        k = int(r.latents.shape[0]) if r.latents is not None else 0
        if k:
            lat[i, :k] = r.latents.float().numpy()
        nl[i] = k
        rec = example_targets(ex, k, c_thought, tokenizer, task)
        rec.update(correct=bool(r.correct), pred=r.pred, gold=r.gold)
        records.append(rec)
    info = {"task": records[0]["task"] if records else task, "n_latent": int(n_latent), "c_thought": int(c_thought),
            "arm": arm, "max_new_tokens": int(max_new_tokens), "accuracy": float(np.mean([r.correct for r in res]))
            if res else float("nan")}
    return Extraction(latents=lat, n_latent=nl, records=records, info=info)


def save_extraction(ex: Extraction, out_dir: str, *, latents_dtype=np.float32) -> dict:
    """Write latents.npz and index.json under out_dir; returns their paths. The npz is a run artifact (it can pass
    5 MB at 124M scale): out_dir gets a .gitignore for *.npz."""
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, "latents.npz")
    np.savez_compressed(npz, latents=ex.latents.astype(latents_dtype), n_latent=ex.n_latent,
                        idx=np.array([r["idx"] for r in ex.records], dtype=np.int64))
    idx_path = os.path.join(out_dir, "index.json")
    with open(idx_path, "w") as f:
        json.dump({"info": ex.info, "records": ex.records}, f)
    with open(os.path.join(out_dir, ".gitignore"), "w") as f:
        f.write("*.npz\n")
    return {"npz": npz, "index": idx_path}


def load_extraction(out_dir: str) -> Extraction:
    """Inverse of save_extraction."""
    z = np.load(os.path.join(out_dir, "latents.npz"))
    with open(os.path.join(out_dir, "index.json")) as f:
        js = json.load(f)
    return Extraction(latents=z["latents"].astype(np.float32), n_latent=z["n_latent"], records=js["records"],
                      info=js["info"])


__all__ = ["bfs_dist", "frontier_sets", "candidate_nodes", "latent_hops", "gsm_results", "step_entity",
           "first_token", "name_tokens", "detect_task", "example_targets", "Extraction", "extract",
           "save_extraction", "load_extraction"]
