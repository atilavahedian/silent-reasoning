"""ProsQA-format generator with controllable proof depth and distractor-edge count (seeded).

There is no released ProsQA generator: Coconut ships only data/prosqa_{train,valid,test}.json. This module mirrors the
record format of those files exactly (keys question, answer, steps, idx_to_symbol, edges, root, target, neg_target,
in that order) and the regularities measured on the released train split (17,886 records):
  * nodes are listed in topological order (every edge goes from a lower to a higher index) and `edges` is sorted by
    the child index; sources (in-degree 0) are exactly the person names; nodes 0 and 1 are the queried entity (root)
    and a competing entity, in random order;
  * the proof (`steps`) is a shortest root -> target path, one sentence per edge: "<Name> is a <c>." for the first,
    "Every <a> is a <b>." after that; answer "<root> is a <target>.";
  * the negative target is unreachable from the root and reachable from another entity (at depth 1..L in the
    released data); target and negative target are leaves;
  * the question is every edge as a sentence, shuffled, then "Is <root> a <x> or <y>?" with target / negative target
    in random order;
  * about 22.8 nodes (sd 2) and 36 edges (sd 6) per record, 2-7 entities, names from the 17 person names and 38
    concept names that occur in the released files (listed below).
Released depths are 3-6 steps (40% / 43% / 14% / 3%); `depth` here ranges 3-8. The test compares a depth-3
regeneration with the released depth-3 records (edges, words, nodes per record) and checks every proof by BFS.

Construction for one record: lay out a root chain root -> c1 -> ... -> cL (L = depth) and a competing chain
other -> n1 -> ... -> nL' (L' in [max(1, L-2), L] unless given), extra concept nodes (each given one parent) and
extra entities (each given one child), order nodes by a level value (chain node k has level k), then add random
forward distractor edges, rejecting any edge after which the negative target becomes reachable from the root or the
root's shortest path to the target becomes shorter than L. `n_distractor_edges` counts every edge not on the proof
path (competing chain, attachment edges and random edges).
"""
from __future__ import annotations

import random
from collections import deque

PERSONS = ["Alex", "Bob", "Carol", "Davis", "Eva", "Fae", "Gabriel", "Jack", "Max", "Oliver", "Polly", "Rex",
           "Sally", "Sam", "Stella", "Tom", "Wren"]
CONCEPTS = ["boompus", "bompus", "brimpus", "chorpus", "dumpus", "felpus", "fompus", "gerpus", "gorpus", "grimpus",
            "gwompus", "hilpus", "impus", "jelpus", "jompus", "kerpus", "lempus", "lorpus", "numpus", "quimpus",
            "rempus", "rompus", "rorpus", "scrompus", "shumpus", "sterpus", "storpus", "terpus", "timpus", "tumpus",
            "vumpus", "worpus", "wumpus", "yerpus", "yimpus", "yumpus", "zhorpus", "zumpus"]
_EXTRA_ENTITY_WEIGHTS = [0.57, 0.31, 0.10, 0.02]  # released: 2 / 3 / 4 / 5+ entities per record


def bfs_dist(n: int, edges, src: int) -> dict[int, int]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    dist, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _sentence(names, is_entity, u, v):
    return f"{names[u]} is a {names[v]}." if is_entity[u] else f"Every {names[u]} is a {names[v]}."


def generate_record(rng: random.Random, depth: int = 3, n_distractor_edges: int | None = None,
                    n_nodes: int | None = None, n_extra_entities: int | None = None,
                    neg_depth: int | None = None, max_tries: int = 50) -> dict:
    """One ProsQA-format record (format of coconut data/prosqa_*.json). depth = proof steps (3-8)."""
    if not 1 <= depth <= 8:
        raise ValueError("depth must be in 1..8")
    for _ in range(max_tries):
        rec = _try_record(rng, depth, n_distractor_edges, n_nodes, n_extra_entities, neg_depth)
        if rec is not None:
            return rec
    raise RuntimeError("could not place the requested distractor edges; lower n_distractor_edges")


def _try_record(rng, L, n_distractor_edges, n_nodes, n_extra_entities, neg_depth):
    Lp = neg_depth if neg_depth is not None else rng.randint(max(1, L - 2), L)
    n_ent_extra = (n_extra_entities if n_extra_entities is not None
                   else rng.choices(range(len(_EXTRA_ENTITY_WEIGHTS)), _EXTRA_ENTITY_WEIGHTS)[0])
    if n_nodes is None:
        n_nodes = min(28, max(14, round(rng.gauss(22.8, 2.0))))
    n_concepts = max(n_nodes - 2 - n_ent_extra, L + Lp)
    if n_concepts > len(CONCEPTS) or 2 + n_ent_extra > len(PERSONS):
        raise ValueError("graph too large for the name vocabulary")
    n_extra_concepts = n_concepts - L - Lp

    # nodes: (kind, level); kinds: root, other, ent, chain k, negchain k, concept
    nodes = [("root", 0.0, 0), ("other", 0.0, 0)]
    nodes += [("ent", rng.uniform(0, max(L - 1, 0.5)), 0) for _ in range(n_ent_extra)]
    nodes += [("chain", float(k), k) for k in range(1, L + 1)]
    nodes += [("neg", k * L / Lp + rng.uniform(-0.3, 0.3), k) for k in range(1, Lp + 1)]
    nodes += [("concept", rng.uniform(0.5, L + 0.5), 0) for _ in range(n_extra_concepts)]
    first_two = [0, 1]
    rng.shuffle(first_two)
    rest = sorted(range(2, len(nodes)), key=lambda i: (nodes[i][1], rng.random()))
    order = first_two + rest  # order[new_index] = old index
    pos = {old: new for new, old in enumerate(order)}
    n = len(order)
    is_entity = [nodes[order[i]][0] in ("root", "other", "ent") for i in range(n)]
    persons = rng.sample(PERSONS, sum(is_entity))
    concepts = rng.sample(CONCEPTS, n - sum(is_entity))
    pi, ci, names = iter(persons), iter(concepts), []
    for i in range(n):
        names.append(next(pi) if is_entity[i] else next(ci))

    def find(kind, k=0):
        return next(pos[o] for o, (kk, _, kidx) in enumerate(nodes) if kk == kind and kidx == k)

    root, other = pos[0], pos[1]
    chain = [root] + [find("chain", k) for k in range(1, L + 1)]
    negc = [other] + [find("neg", k) for k in range(1, Lp + 1)]
    target, neg_target = chain[-1], negc[-1]
    if any(chain[i] >= chain[i + 1] for i in range(L)) or any(negc[i] >= negc[i + 1] for i in range(Lp)):
        return None
    edges = [(chain[i], chain[i + 1]) for i in range(L)] + [(negc[i], negc[i + 1]) for i in range(Lp)]
    eset = set(edges)
    terminal = {target, neg_target}

    def ok(extra):
        d = bfs_dist(n, extra, root)
        return neg_target not in d and d.get(target) == L

    if not ok(edges):
        return None
    on_chains = set(chain) | set(negc)
    for v in range(n):  # attach extra concepts (one parent) and extra entities (one child)
        if v in on_chains:
            continue
        if is_entity[v]:
            cands = [w for w in range(v + 1, n) if not is_entity[w] and (v, w) not in eset]
            rng.shuffle(cands)
            placed = False
            for w in cands:
                if ok(edges + [(v, w)]):
                    edges.append((v, w)); eset.add((v, w)); placed = True
                    break
        else:
            if any(e[1] == v for e in eset):  # already given a parent by an entity's attachment edge
                continue
            cands = [u for u in range(v) if u not in terminal and (u, v) not in eset]
            rng.shuffle(cands)
            placed = False
            for u in cands:
                if ok(edges + [(u, v)]):
                    edges.append((u, v)); eset.add((u, v)); placed = True
                    break
        if not placed:
            return None

    mandatory = len(edges) - L
    if n_distractor_edges is None:
        total = min(63, max(15, round(rng.gauss(37.0 - (L - 3), 6.0))))
        want = max(total - L, mandatory)
    else:
        if n_distractor_edges < mandatory:
            raise ValueError(f"n_distractor_edges={n_distractor_edges} is below the {mandatory} edges this graph "
                             "needs (competing chain plus one edge per extra node)")
        want = n_distractor_edges
    cands = [(u, v) for v in range(n) if not is_entity[v] for u in range(v) if u not in terminal
             and (u, v) not in eset]
    rng.shuffle(cands)
    for e in cands:
        if len(edges) - L >= want:
            break
        if ok(edges + [e]):
            edges.append(e); eset.add(e)
    if len(edges) - L < want:
        if n_distractor_edges is not None:
            return None
    edges_sorted = sorted(edges, key=lambda e: e[1])  # stable: released files list edges by child index
    sents = [_sentence(names, is_entity, u, v) for u, v in edges]
    rng.shuffle(sents)
    x, y = (target, neg_target) if rng.random() < 0.5 else (neg_target, target)
    question = " ".join(sents) + f" Is {names[root]} a {names[x]} or {names[y]}?"
    steps = [_sentence(names, is_entity, chain[i], chain[i + 1]) for i in range(L)]
    return {"question": question, "answer": f"{names[root]} is a {names[target]}.", "steps": steps,
            "idx_to_symbol": names, "edges": [[u, v] for u, v in edges_sorted], "root": root, "target": target,
            "neg_target": neg_target}


def generate(n: int, depth: int = 3, seed: int = 0, **kw) -> list[dict]:
    """n records, seeded (random.Random(seed)); kw as generate_record."""
    rng = random.Random(seed)
    return [generate_record(rng, depth=depth, **kw) for _ in range(n)]


def validate_record(rec: dict) -> list[str]:
    """Problems with a ProsQA-format record (empty list = valid): the proof is a root -> target path whose sentences
    are edges of the record, it is a shortest path, the negative target is unreachable from the root (BFS), every
    question sentence is an edge and every edge a sentence, edges go forward, and the answer names the target."""
    errs = []
    names, edges = rec["idx_to_symbol"], [tuple(e) for e in rec["edges"]]
    n, root, t, ng = len(names), rec["root"], rec["target"], rec["neg_target"]
    eset = set(edges)
    has_parent = {v for _, v in edges}
    is_entity = [i not in has_parent for i in range(n)]
    by_sentence = {_sentence(names, is_entity, u, v): (u, v) for u, v in edges}
    cur = root
    for s in rec["steps"]:
        e = by_sentence.get(s)
        if e is None or e not in eset or e[0] != cur:
            errs.append(f"step {s!r} is not an edge continuing the path from {names[cur]}")
            break
        cur = e[1]
    if cur != t:
        errs.append("proof does not end at the target")
    d = bfs_dist(n, edges, root)
    if ng in d:
        errs.append("negative target reachable from root")
    if d.get(t) != len(rec["steps"]):
        errs.append(f"proof length {len(rec['steps'])} != shortest path {d.get(t)}")
    body, _, query = rec["question"].rpartition(" Is ")
    sents = [s.strip() + "." for s in body.split(".") if s.strip()]
    if sorted(sents) != sorted(by_sentence):
        errs.append("question statements differ from the edge list")
    if query.rstrip("?") not in (f"{names[root]} a {names[t]} or {names[ng]}", f"{names[root]} a {names[ng]} or {names[t]}"):
        errs.append("query does not name root, target and negative target")
    if any(u >= v for u, v in edges):
        errs.append("edge not in topological order")
    if rec["answer"] != f"{names[root]} is a {names[t]}.":
        errs.append("answer does not name the target")
    return errs


__all__ = ["generate", "generate_record", "validate_record", "bfs_dist", "PERSONS", "CONCEPTS"]
