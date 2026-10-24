"""Positive control (theory memo T3; registration 7.9): a tiny from-scratch Coconut on synthetic reachability.

Task ("reach"). A random DAG on n nodes (default 10) named by a random sample of the symbols v0..v23; nodes are in
topological index order, the first `n_sources` nodes have no parent and every other node gets one parent among the
earlier nodes plus random forward extra edges. The question lists every edge as "a > b ," (one fixed shuffled order
stored in meta["edges"], so a minimal pair with the same graph has byte-identical edge text) and ends with
"from Y reach X ?"; the answer is "yes" or "no" (balanced; "yes" queries at a uniformly drawn distance 1..D, "no"
queries preferably at nodes that have a parent, so in-degree alone does not answer). CoT step t (t = 1..D) lists
F_t, the nodes reachable from Y within t hops, sorted by symbol number (a canonical order the model can see; the
node index is not visible) ("none ." when empty), so the Coconut curriculum
replaces exactly the states the R2 probes target. Records carry ProsQA-style meta (idx_to_symbol, edges, root) plus
query and reachable, so the whole readout (extract, R1, R2, F1, F2, F3, load-bearing) runs unchanged.

Model: GPT-2 config, 2 layers, 64-dim, 4 heads, randomly initialised, trained with LatentLM.forward(mode="coconut")
under the Coconut stage curriculum (stage k replaces the first k steps by k latents; optimizer reset per stage, as
the reference) for a few hundred steps on CPU. That superposition emerges is a result, never a test assertion.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from latent.data import Collator, Example, add_latent_tokens, build_stage_dataset, tokenize_examples
from latent.model import LatentLM
from latent.probe.extract import bfs_dist, frontier_sets

SYMBOLS = [f"v{i}" for i in range(24)]
WORDS = [">", ",", ".", "from", "reach", "?", "yes", "no", "###", "none"]


class WordTokenizer:
    """Closed-vocabulary whitespace tokenizer satisfying the INTERFACE.md tokenizer protocol (no downloads)."""

    def __init__(self, words=None):
        self.id_to_tok = ["<eos>", "<pad>"] + list(words if words is not None else WORDS + SYMBOLS)
        self.tok_to_id = {t: i for i, t in enumerate(self.id_to_tok)}
        self.eos_token_id, self.pad_token_id = 0, 1
        self.special = {0, 1}

    def add_tokens(self, toks):
        n = 0
        for t in toks:
            if t not in self.tok_to_id:
                self.tok_to_id[t] = len(self.id_to_tok)
                self.id_to_tok.append(t)
                self.special.add(self.tok_to_id[t])
                n += 1
        return n

    def convert_tokens_to_ids(self, tok):
        return self.tok_to_id[tok]

    def encode(self, text, add_special_tokens=False):
        return [self.tok_to_id[w] for w in text.split()]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.id_to_tok[i] for i in ids if not (skip_special_tokens and i in self.special))

    def __len__(self):
        return len(self.id_to_tok)


def make_tokenizer() -> WordTokenizer:
    t = WordTokenizer()
    add_latent_tokens(t)
    return t


def _symbol_rank(name: str):
    return (0, int(name[1:])) if name[:1] == "v" and name[1:].isdigit() else (1, name)


def reach_example(names, edges, root: int, query: int, *, idx: int, n_steps: int) -> Example:
    """One reach record from a graph (edge order as given), a source and a query node."""
    n = len(names)
    reach = query in bfs_dist(n, edges, root)
    fr = frontier_sets(edges, root, n, n_steps)
    steps, ents, prev = [], [], set()
    order = lambda v: _symbol_rank(names[v])  # noqa: E731  (a visible canonical order: the symbol's number)
    for f in fr:
        steps.append((" ".join(names[v] for v in sorted(f, key=order)) or "none") + " .")
        new = sorted(f - prev, key=order)
        ents.append(names[new[0]] if new else None)
        prev = set(f)
    q = " ".join(f"{names[u]} > {names[v]} ," for u, v in edges) + f" from {names[root]} reach {names[query]} ?"
    return Example(question=q, answer="yes" if reach else "no", steps=steps, idx=idx,
                   meta={"idx_to_symbol": list(names), "edges": [list(e) for e in edges], "root": int(root),
                         "query": int(query), "reachable": bool(reach), "step_entities": ents})


def random_reach_record(rng: random.Random, idx: int, *, n_nodes: int = 10, n_sources: int = 2, extra_edges: int = 3,
                        depth: int = 3, max_tries: int = 100) -> Example:
    """A balanced reach record (module docstring)."""
    want_yes = rng.random() < 0.5
    for _ in range(max_tries):
        names = rng.sample(SYMBOLS, n_nodes)
        edges = {(rng.randrange(v), v) for v in range(n_sources, n_nodes)}
        for _ in range(extra_edges):
            u, v = sorted(rng.sample(range(n_nodes), 2))
            if v >= n_sources:
                edges.add((u, v))
        edges = sorted(edges)
        rng.shuffle(edges)
        root = rng.randrange(n_sources)
        dist = bfs_dist(n_nodes, edges, root)
        if want_yes:
            levels = sorted({d for v, d in dist.items() if 1 <= d <= depth})
            if not levels:
                continue
            lvl = rng.choice(levels)
            query = rng.choice(sorted(v for v, d in dist.items() if d == lvl))
        else:
            has_parent = {v for _, v in edges}
            un = sorted(v for v in range(n_nodes) if v not in dist)
            pref = [v for v in un if v in has_parent]
            if not un:
                continue
            query = rng.choice(pref or un)
        return reach_example(names, edges, root, query, idx=idx, n_steps=depth)
    raise RuntimeError("could not build a reach record; relax the graph parameters")


def make_reach_data(n: int, seed: int = 0, start_idx: int = 0, **kw) -> list[Example]:
    rng = random.Random(seed)
    return [random_reach_record(rng, start_idx + i, **kw) for i in range(n)]


def build_tiny_lm(tokenizer, *, n_layer: int = 2, n_embd: int = 64, n_head: int = 4, seed: int = 0) -> LatentLM:
    """Randomly initialised GPT-2 (no download) wrapped in LatentLM."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=n_layer, n_embd=n_embd, n_head=n_head, vocab_size=len(tokenizer), n_positions=512,
                     bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    ids = {k: tokenizer.convert_tokens_to_ids(t) for k, t in
           (("start_id", "<|start-latent|>"), ("end_id", "<|end-latent|>"), ("latent_id", "<|latent|>"))}
    return LatentLM(GPT2LMHeadModel(cfg), ids["latent_id"], ids["start_id"], ids["end_id"], tokenizer.eos_token_id)


def train_coconut(lm: LatentLM, tokenizer, examples, *, depth: int = 3, steps: int = 400, batch_size: int = 64,
                  lr: float = 1e-3, weight_decay: float = 0.01, seed: int = 0, device: str = "cpu",
                  final_stage_share: float = 0.4, log_every: int = 0) -> dict:
    """Coconut curriculum from scratch: stages 0..depth-1 get equal shares of (1 - final_stage_share) of the
    steps, the final stage (all `depth` steps latent) the rest; AdamW rebuilt at every stage change."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    ids = {k: tokenizer.convert_tokens_to_ids(t) for k, t in
           (("start_id", "<|start-latent|>"), ("end_id", "<|end-latent|>"), ("latent_id", "<|latent|>"))}
    tok_ex = tokenize_examples(examples, tokenizer)
    collate = Collator(tokenizer.pad_token_id, ids["latent_id"])
    n_early = int(steps * (1 - final_stage_share))
    per = max(1, n_early // depth) if depth else 1
    lm.to(device).train()
    log, opt, cur, t0 = [], None, None, time.time()
    for s in range(steps):
        stage = min(s // per, depth) if s < n_early else depth
        if stage != cur:
            opt = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=weight_decay)
            cur = stage
        batch = [tok_ex[rng.randrange(len(tok_ex))] for _ in range(batch_size)]
        recs = build_stage_dataset(batch, stage=stage, arm="coconut", c_thought=1, max_latent_stage=depth,
                                   pad_latent_to_max=True, uniform_prob=0.0, rng=rng, **ids)
        b = {k: v.to(device) for k, v in collate(recs).items() if k != "idx"}
        loss = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="coconut").loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        if log_every and (s % log_every == 0 or s == steps - 1):
            log.append({"step": s, "stage": stage, "loss": float(loss.item())})
    lm.eval()
    return {"steps": steps, "seconds": time.time() - t0, "final_loss": float(loss.item()) if steps else None,
            "log": log}


def run_positive_control(out_dir: str | None = None, *, n_train: int = 4000, n_probe: int = 300, steps: int = 400,
                         batch_size: int = 64, depth: int = 3, n_nodes: int = 10, seed: int = 0, device: str = "cpu",
                         readout_kw: dict | None = None) -> dict:
    """Generate data, train the tiny model, run the full readout on a disjoint probe split; writes probe_report.json and
    probe_report.md under out_dir when given. Returns the report dict (with a "training" section)."""
    from latent.probe.report import run_readout, write_report

    tok = make_tokenizer()
    gk = dict(n_nodes=n_nodes, depth=depth)
    train = make_reach_data(n_train, seed=seed, **gk)
    probe = make_reach_data(n_probe, seed=seed + 10_000, start_idx=n_train, **gk)
    seen = {e.question for e in train}
    probe = [e for e in probe if e.question not in seen] or probe
    lm = build_tiny_lm(tok, seed=seed)
    tr = train_coconut(lm, tok, train, depth=depth, steps=steps, batch_size=batch_size, seed=seed, device=device)
    kw = dict(batch_size=128, max_new_tokens=4, seed=seed)
    kw.update(readout_kw or {})
    rep = run_readout(lm, tok, probe, task="reach", n_latent=depth, device=device, out_dir=out_dir,
                      meta={"checkpoint": "synthetic positive control (in memory)", "split": "synthetic probe split"},
                      **kw)
    rep["training"] = {**{k: v for k, v in tr.items() if k != "log"}, "n_train": len(train), "n_probe": len(probe),
                       "n_layer": 2, "n_embd": 64, "depth": depth, "n_nodes": n_nodes}
    if out_dir:
        write_report(rep, out_dir)
    return rep


__all__ = ["WordTokenizer", "make_tokenizer", "reach_example", "random_reach_record", "make_reach_data",
           "build_tiny_lm", "train_coconut", "run_positive_control", "SYMBOLS"]
