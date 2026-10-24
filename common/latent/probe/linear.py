"""R1 logit lens, R2 linear probes and the R3 decoder interface (plan 6.5; registration 7.9 coverage rule).

R2 probe design (fixed here). ProsQA-format frontier membership is probed with one L2-regularised logistic probe per
symbol NAME (a "slot" is a concept name of the global vocabulary, e.g. "sterpus", shared across records), on the
standardised latent vector alone, pooled over latent steps. Rows are (example, latent step j, candidate node v)
with label [v in F_{t(j)}] and the probe of name(v). Why not one probe on [latent ; symbol embedding]: a linear
function of a concatenation is additive, w.h + u.e + b, so the ranking of symbols is the same for every latent and
membership (an interaction between h and e) cannot be expressed; a bilinear probe costs d^2 parameters. Per-name
probes are the standard linear readout, have d + 1 parameters each, and give the per-symbol latent-space direction
that the F3 edit needs. Pooling over steps keeps about 6x more rows per probe than per-step probes (at 124M,
d = 768 and about 200 records per name in a 500-record split).
GSM: one multinomial logistic probe (pooled over steps) for the first token id of step t's result, from the latent
that carries step t; classes seen fewer than `min_count` times in the training folds share one "other" class.

Held-out protocol: k-fold over EXAMPLES (all rows of an example in one fold), so every score is out-of-fold and the
probe used to edit example i (F3) never saw example i. Chance: (a) the registered shuffled-label baseline, the same
probe trained on labels permuted within each (name, latent step) cell (GSM: within each step) of the training folds
and scored on the true held-out labels; permuting within the step keeps each step's prevalence, so a pooled probe
cannot beat the baseline by reading the step index alone; (b) the analytic constant-predictor level (F1 of "always in" = 2p / (1 + p); GSM majority class).
A step j is DECODED iff score_j > shuffled_j + 2 sigma, sigma = sqrt(s (1 - s) / n_j), s = shuffled_j, n_j = number
of examples with step j (registration 7.9, binomial over examples).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from latent.probe.extract import Extraction, name_tokens
from latent.probe.stats import binomial_sigma, bootstrap_mean_ci, bootstrap_stat_ci


# ----------------------------------------------------------------------------------------------- fitting primitives
def kfold_assign(n: int, k: int, seed: int = 0) -> np.ndarray:
    """fold id per example: a seeded permutation dealt round-robin into k folds."""
    k = max(2, min(k, n))
    perm = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, dtype=np.int64)
    fold[perm] = np.arange(n) % k
    return fold


def _lbfgs(params, closure_loss, max_iter):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = closure_loss()
        loss.backward()
        return loss
    opt.step(closure)


def fit_grouped_logistic(X: np.ndarray, rows_x: np.ndarray, rows_g: np.ndarray, y: np.ndarray, n_groups: int,
                         l2: float = 1e-2, max_iter: int = 200):
    """One logistic probe per group, fitted jointly (the objective is separable, so this equals separate fits):
    sum_g [ mean_{rows of g} BCE(x_r . w_g + b_g, y_r) + l2 / 2 ||w_g||^2 ]. X is [n_vectors, d] (already
    standardised); row r uses vector rows_x[r] and group rows_g[r]. Returns W [G, d], b [G]."""
    Xt = torch.as_tensor(X, dtype=torch.float32)
    ix, ig = torch.as_tensor(rows_x), torch.as_tensor(rows_g)
    yt = torch.as_tensor(y, dtype=torch.float32)
    cnt = torch.bincount(ig, minlength=n_groups).clamp(min=1).float()
    wrow = 1.0 / cnt[ig]
    W = torch.zeros(n_groups, Xt.shape[1], requires_grad=True)
    b = torch.zeros(n_groups, requires_grad=True)

    def loss():
        z = (Xt @ W.T)[ix, ig] + b[ig]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")
        return (bce * wrow).sum() + 0.5 * l2 * (W ** 2).sum()
    _lbfgs([W, b], loss, max_iter)
    return W.detach().numpy(), b.detach().numpy()


def fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1e-2, max_iter: int = 200):
    """Multinomial logistic regression: mean CE + l2 / 2 ||W||^2. Returns W [C, d], b [C]."""
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    W = torch.zeros(n_classes, Xt.shape[1], requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)

    def loss():
        return torch.nn.functional.cross_entropy(Xt @ W.T + b, yt) + 0.5 * l2 * (W ** 2).sum()
    _lbfgs([W, b], loss, max_iter)
    return W.detach().numpy(), b.detach().numpy()


def _standardize(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def raw_space(W, b, mu, sd):
    """Map a probe on standardised features (x - mu) / sd back to raw latent space: w_raw = w / sd,
    b_raw = b - (mu / sd) . w. Then x . w_raw + b_raw equals the standardised logit."""
    W_raw = W / sd
    return W_raw, b - W_raw @ mu


# ------------------------------------------------------------------------------------------------ R2 frontier probes
@dataclass
class FrontierProbes:
    """Out-of-fold frontier probes. names: group (symbol name) list; fold [N]; per fold k and group g: W_raw[k, g],
    b_raw[k, g], trained[k, g], scale[k, g] (std over training latents of the projection on the unit direction)."""
    names: list
    fold: np.ndarray
    W_raw: np.ndarray
    b_raw: np.ndarray
    trained: np.ndarray
    scale: np.ndarray
    rows: dict
    metrics: dict = field(default_factory=dict)

    def direction(self, example_pos: int, name: str):
        """(unit direction u in raw latent space, scale) of the out-of-fold probe for `name` used on example
        `example_pos`, or None when that probe was not trained."""
        g = self.names.index(name) if name in self.names else -1
        k = int(self.fold[example_pos])
        if g < 0 or not self.trained[k, g]:
            return None
        w = self.W_raw[k, g]
        return w / np.linalg.norm(w), float(self.scale[k, g])

    def logit(self, example_pos: int, name: str, h: np.ndarray) -> float | None:
        g = self.names.index(name) if name in self.names else -1
        k = int(self.fold[example_pos])
        if g < 0 or not self.trained[k, g]:
            return None
        return float(h @ self.W_raw[k, g] + self.b_raw[k, g])


def frontier_rows(ext: Extraction):
    """Row table for the frontier probes: example position, latent step, node, name id, label."""
    names = sorted({r["symbols"][v] for r in ext.records for v in r["candidates"]})
    gid = {n: i for i, n in enumerate(names)}
    ex_i, st_j, node, grp, lab = [], [], [], [], []
    for i, r in enumerate(ext.records):
        for j in range(int(ext.n_latent[i])):
            fr = set(r["frontier"][j])
            for v in r["candidates"]:
                ex_i.append(i); st_j.append(j); node.append(v); grp.append(gid[r["symbols"][v]]); lab.append(v in fr)
    return names, {k: np.asarray(v, dtype=np.int64) for k, v in
                   dict(ex=ex_i, step=st_j, node=node, group=grp, label=lab).items()}


def train_frontier_probes(ext: Extraction, *, k: int = 5, l2: float = 1e-2, seed: int = 0, min_rows: int = 8,
                          max_iter: int = 200, sigma_k: float = 2.0) -> FrontierProbes:
    """k-fold per-name logistic probes (module docstring); fills oof scores for the real and shuffled-label probes."""
    names, rows = frontier_rows(ext)
    N, L, d = ext.latents.shape
    G = len(names)
    fold = kfold_assign(N, k, seed)
    K = int(fold.max()) + 1 if N else 0
    vec = rows["ex"] * L + rows["step"]
    H = ext.latents.reshape(N * L, d)
    W_raw = np.zeros((K, G, d), np.float32); b_raw = np.zeros((K, G), np.float32)
    trained = np.zeros((K, G), bool); scale = np.ones((K, G), np.float32)
    oof = {"real": np.zeros(len(vec), np.float32), "shuffled": np.zeros(len(vec), np.float32)}
    rng = np.random.default_rng(seed + 1)
    for f in range(K):
        tr = fold[rows["ex"]] != f
        te = ~tr
        used = np.unique(vec[tr])
        mu, sd = _standardize(H[used])
        Xs = (H - mu) / sd
        g_tr, y_tr = rows["group"][tr], rows["label"][tr]
        cnt = np.bincount(g_tr, minlength=G); pos = np.bincount(g_tr, weights=y_tr, minlength=G)
        ok = (cnt >= min_rows) & (pos > 0) & (pos < cnt)
        prior = np.where(cnt > 0, pos / np.maximum(cnt, 1), 0.0)
        y_sh = y_tr.copy()
        cell = g_tr * L + rows["step"][tr]  # permute within (name, step): keeps each step's prevalence
        for c in np.unique(cell):
            m = np.nonzero(cell == c)[0]
            y_sh[m] = y_tr[rng.permutation(m)]
        for kind, labels in (("real", y_tr), ("shuffled", y_sh)):
            W, b = fit_grouped_logistic(Xs, vec[tr], g_tr, labels, G, l2=l2, max_iter=max_iter)
            z = (Xs[vec[te]] * W[rows["group"][te]]).sum(1) + b[rows["group"][te]]
            const = np.where(prior[rows["group"][te]] >= 0.5, 10.0, -10.0)
            oof[kind][te] = np.where(ok[rows["group"][te]], z, const)
            if kind == "real":
                Wr, br = raw_space(W, b, mu, sd)
                W_raw[f], b_raw[f], trained[f] = Wr, br, ok
                for g in np.nonzero(ok)[0]:
                    u = Wr[g] / np.linalg.norm(Wr[g])
                    scale[f, g] = float(np.std(H[used] @ u)) or 1.0
    rows["oof_real"], rows["oof_shuffled"] = oof["real"], oof["shuffled"]
    fp = FrontierProbes(names=names, fold=fold, W_raw=W_raw, b_raw=b_raw, trained=trained, scale=scale, rows=rows)
    fp.metrics = frontier_metrics(rows, N, L, seed=seed, sigma_k=sigma_k)
    fp.metrics.update(k=K, l2=l2, min_rows=min_rows, n_names=G,
                      trained_fraction=float(trained.mean()) if trained.size else 0.0)
    return fp


def _step_counts(rows, score_key, N, j):
    m = rows["step"] == j
    ex = rows["ex"][m]
    t = rows["label"][m].astype(bool)
    p = rows[score_key][m] > 0
    tp = np.bincount(ex, weights=(t & p), minlength=N)
    nt = np.bincount(ex, weights=t, minlength=N)
    npred = np.bincount(ex, weights=p, minlength=N)
    corr = np.bincount(ex, weights=(t == p), minlength=N)
    tot = np.bincount(ex, minlength=N)
    present = tot > 0
    return tp[present], nt[present], npred[present], corr[present], tot[present]


def frontier_metrics(rows, N, L, *, seed=0, n_boot=1000, sigma_k=2.0):
    """Per latent step: oof F1 (bootstrap CI over examples), accuracy, shuffled-label F1, analytic chance, decoded."""
    steps = []
    for j in range(L):
        tp, nt, npd, corr, tot = _step_counts(rows, "oof_real", N, j)
        n = len(tp)
        if n == 0:
            continue
        f1 = lambda ix, tp=tp, nt=nt, npd=npd: 2 * tp[ix].sum() / max(nt[ix].sum() + npd[ix].sum(), 1e-12)
        point, lo, hi = bootstrap_stat_ci(n, f1, n_boot=n_boot, seed=seed)
        stp, snt, snp, scorr, stot = _step_counts(rows, "oof_shuffled", N, j)
        sh = 2 * stp.sum() / max(snt.sum() + snp.sum(), 1e-12)
        prev = nt.sum() / max(tot.sum(), 1)
        sig = binomial_sigma(sh, n)
        steps.append({"step": j, "n_examples": int(n), "f1": point, "f1_ci95": [lo, hi],
                      "accuracy": float(corr.sum() / tot.sum()), "shuffled_f1": float(sh),
                      "shuffled_accuracy": float(scorr.sum() / stot.sum()), "prevalence": float(prev),
                      "chance_f1_always_in": float(2 * prev / (1 + prev)) if prev > 0 else 0.0,
                      "sigma": sig, "decoded": bool(point > sh + sigma_k * sig)})
    return {"metric": "f1", "per_step": steps, "sigma_k": sigma_k}


# ---------------------------------------------------------------------------------------------------- R2 GSM probes
def train_gsm_probes(ext: Extraction, *, k: int = 5, l2: float = 1e-2, seed: int = 0, min_count: int = 3,
                     max_iter: int = 200, sigma_k: float = 2.0) -> dict:
    """k-fold multinomial probe for the first token of step t's result from the latent carrying step t."""
    N, L, d = ext.latents.shape
    ex_i, st_j, lab = [], [], []
    for i, r in enumerate(ext.records):
        for j in range(int(ext.n_latent[i])):
            s = r["latent_step"][j]
            if s >= 0 and r["result_first_token"][s] is not None:
                ex_i.append(i); st_j.append(j); lab.append(r["result_first_token"][s])
    ex_i, st_j, lab = np.array(ex_i, np.int64), np.array(st_j, np.int64), np.array(lab, np.int64)
    fold = kfold_assign(N, k, seed)
    pred = {"real": np.full(len(lab), -2), "shuffled": np.full(len(lab), -2)}
    majority = np.zeros(len(lab), bool)
    rng = np.random.default_rng(seed + 1)
    for f in range(int(fold.max()) + 1 if N else 0):
        tr, te = fold[ex_i] != f, fold[ex_i] == f
        if not tr.any() or not te.any():
            continue
        vals, cnts = np.unique(lab[tr], return_counts=True)
        classes = list(vals[cnts >= min_count])
        cid = {c: i for i, c in enumerate(classes)}
        other = len(classes)
        enc = np.array([cid.get(c, other) for c in lab])
        X = ext.latents[ex_i, st_j]
        mu, sd = _standardize(X[tr])
        Xs = (X - mu) / sd
        maj = np.bincount(enc[tr], minlength=other + 1).argmax()
        majority[te] = enc[te] == maj
        y_sh = enc[tr].copy()
        for j in np.unique(st_j[tr]):  # permute within step
            m = np.nonzero(st_j[tr] == j)[0]
            y_sh[m] = enc[tr][rng.permutation(m)]
        for kind, y in (("real", enc[tr]), ("shuffled", y_sh)):
            W, b = fit_softmax(Xs[tr], y, other + 1, l2=l2, max_iter=max_iter)
            p = (Xs[te] @ W.T + b).argmax(1)
            pred[kind][te] = np.array([classes[c] if c < other else -1 for c in p])
    steps = []
    for j in sorted(set(st_j.tolist())):
        m = st_j == j
        acc = (pred["real"][m] == lab[m]).astype(float)
        sh = float((pred["shuffled"][m] == lab[m]).mean())
        mean, lo, hi = bootstrap_mean_ci(acc, seed=seed)
        sig = binomial_sigma(sh, int(m.sum()))
        steps.append({"step": int(j), "n_examples": int(m.sum()), "accuracy": mean, "accuracy_ci95": [lo, hi],
                      "shuffled_accuracy": sh, "chance_majority": float(majority[m].mean()), "sigma": sig,
                      "decoded": bool(mean > sh + sigma_k * sig)})
    return {"metric": "accuracy", "per_step": steps, "k": int(fold.max()) + 1 if N else 0, "l2": l2,
            "min_count": min_count, "n_rows": int(len(lab))}


# ------------------------------------------------------------------------------------------------------ R1 logit lens
@torch.no_grad()
def logit_lens(lm, tokenizer, ext: Extraction, *, k: int = 5, seed: int = 0, n_show: int = 3) -> dict:
    """Project every latent through the LM head (the latent is the post-final-norm hidden state, so this is the
    model's own next-token readout at that position) and report per latent j: whether the entity of CoT step
    s = j // c_thought (the step latent j replaced, i.e. the next CoT step at that position; ProsQA: the node the
    step reaches, GSM: the result's first token) is in the top-k ("next_step_entity_topk"), the same for step s + 1,
    and the mean number of top-k tokens that name a node of F_t (the frontier mixture of T3). Names are matched by
    their first token (" name" or "name")."""
    head = lm.lm_head
    w = head.weight
    N, L, d = ext.latents.shape
    out, shown = [], []
    for j in range(L):
        idx = [i for i in range(N) if ext.n_latent[i] > j]
        if not idx:
            continue
        h = torch.as_tensor(ext.latents[idx, j], dtype=w.dtype, device=w.device)
        top = head(h).topk(k, dim=-1).indices.cpu().numpy()
        hit_next, hit_after, mix = [], [], []
        for row, i in enumerate(idx):
            r, tk = ext.records[i], set(top[row].tolist())
            ents = _step_targets(tokenizer, r)
            c = int(r.get("c_thought", 1))
            for lst, s in ((hit_next, j // c), (hit_after, j // c + 1)):
                if s < len(ents) and ents[s]:
                    lst.append(float(bool(tk & set(ents[s]))))
            if "frontier" in r:
                fr_tok = {t for v in r["frontier"][j] for t in name_tokens(tokenizer, r["symbols"][v])}
                mix.append(float(len(tk & fr_tok)))
            if j == 0 and len(shown) < n_show:
                shown.append({"idx": r["idx"], "top": [_safe_decode(tokenizer, int(t)) for t in top[row]]})
        row = {"step": j, "n_examples": len(idx)}
        for name, lst in (("next_step_entity_topk", hit_next), ("following_step_entity_topk", hit_after),
                          ("frontier_tokens_in_topk", mix)):
            m, lo, hi = bootstrap_mean_ci(lst, seed=seed)
            row[name] = m
            row[name + "_ci95"] = [lo, hi]
            row[name + "_n"] = len(lst)
        out.append(row)
    return {"k": k, "per_step": out, "examples": shown, "note": "R1 is reported and never counts (registration 7.9)"}


def _safe_decode(tokenizer, t):
    try:
        return tokenizer.decode([t])
    except (IndexError, KeyError, ValueError):  # padded vocabulary rows have no token
        return f"<id {t}>"


def _step_targets(tokenizer, r):
    if r["task"] == "gsm":
        res = r["results"]
        return [None if x is None else [t for t in [r["result_first_token"][s]] if t is not None]
                for s, x in enumerate(res)] if res else []
    return [None if e is None else name_tokens(tokenizer, e) for e in r.get("step_entities", [])]


# ------------------------------------------------------------------------------------------------- R3 decoder hook
class StateReadout(Protocol):
    """R3 interface (card C2): a callable from one fed-back latent vector to a state prediction.
    readout(latent [d] tensor, hop t >= 1, record) -> for prosqa / reach an iterable of node indices predicted in F_t
    (node index = position in record["symbols"], the C2 decoder's symbol slot); for gsm the predicted first token id
    of step t's result (int)."""
    def __call__(self, latent: torch.Tensor, hop: int, record: dict) -> Any: ...


class C2DecoderReadout:
    """Adapter from a batched decoder fn(latents [n, d], hops [n] long) -> logits to StateReadout. prosqa / reach:
    logits [n, max_symbols], slot v = node v, sigmoid > threshold, restricted to the record's candidates. gsm:
    logits [n, n_heads, V] (or [n, V]); the first head's argmax is the first result token."""

    def __init__(self, fn, task: str, threshold: float = 0.5):
        self.fn, self.task, self.threshold = fn, task, threshold

    @torch.no_grad()
    def __call__(self, latent, hop, record):
        z = self.fn(latent.reshape(1, -1), torch.tensor([hop], dtype=torch.long))[0]
        if self.task == "gsm":
            return int((z[0] if z.dim() == 2 else z).argmax().item())
        p = torch.sigmoid(z).cpu().numpy()
        return [v for v in record["candidates"] if v < len(p) and p[v] > self.threshold]


def score_readout(readout, ext: Extraction, *, seed: int = 0, sigma_k: float = 2.0) -> dict:
    """Score an R3 readout with the R2 metrics and decoded rule; the chance baseline is the same readout applied to
    another example's latent (a seeded derangement) and scored against this example's targets."""
    N, L, _ = ext.latents.shape
    perm = np.random.default_rng(seed).permutation(N)
    perm = np.roll(perm, 1)[np.argsort(perm)] if N > 1 else perm  # each i maps to a different example
    task = ext.task
    if task == "gsm":
        ex_i, st, ok_r, ok_s = [], [], [], []
        for i, r in enumerate(ext.records):
            for j in range(int(ext.n_latent[i])):
                s = r["latent_step"][j]
                if s < 0 or r["result_first_token"][s] is None or ext.n_latent[perm[i]] <= j:
                    continue
                y = r["result_first_token"][s]
                ok_r.append(float(readout(torch.as_tensor(ext.latents[i, j]), s + 1, r) == y))
                ok_s.append(float(readout(torch.as_tensor(ext.latents[perm[i], j]), s + 1, r) == y))
                ex_i.append(i); st.append(j)
        st, ok_r, ok_s = np.array(st), np.array(ok_r), np.array(ok_s)
        steps = []
        for j in sorted(set(st.tolist())):
            m = st == j
            mean, lo, hi = bootstrap_mean_ci(ok_r[m], seed=seed)
            sh = float(ok_s[m].mean()); sig = binomial_sigma(sh, int(m.sum()))
            steps.append({"step": j, "n_examples": int(m.sum()), "accuracy": mean, "accuracy_ci95": [lo, hi],
                          "shuffled_accuracy": sh, "sigma": sig, "decoded": bool(mean > sh + sigma_k * sig)})
        return {"metric": "accuracy", "per_step": steps}
    rows = {k: [] for k in ("ex", "step", "label", "oof_real", "oof_shuffled")}
    for i, r in enumerate(ext.records):
        for j in range(int(ext.n_latent[i])):
            if ext.n_latent[perm[i]] <= j:
                continue
            hop = r["hops"][j]
            pr = set(readout(torch.as_tensor(ext.latents[i, j]), hop, r))
            ps = set(readout(torch.as_tensor(ext.latents[perm[i], j]), hop, r))
            fr = set(r["frontier"][j])
            for v in r["candidates"]:
                rows["ex"].append(i); rows["step"].append(j); rows["label"].append(v in fr)
                rows["oof_real"].append(1.0 if v in pr else -1.0); rows["oof_shuffled"].append(1.0 if v in ps else -1.0)
    rows = {k: np.asarray(v) for k, v in rows.items()}
    rows["ex"], rows["step"] = rows["ex"].astype(np.int64), rows["step"].astype(np.int64)
    return frontier_metrics(rows, N, L, seed=seed, sigma_k=sigma_k)


__all__ = ["kfold_assign", "fit_grouped_logistic", "fit_softmax", "raw_space", "FrontierProbes", "frontier_rows",
           "train_frontier_probes", "frontier_metrics", "train_gsm_probes", "logit_lens", "StateReadout",
           "C2DecoderReadout", "score_readout"]
