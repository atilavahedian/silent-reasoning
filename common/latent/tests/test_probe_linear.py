"""latent.probe.linear: probe fitting recovers separable structure, raw-space mapping, k-fold, frontier probes on a
superposition construction, GSM probes, the R3 readout interface, and the logit lens."""
import numpy as np
import pytest
import torch

from latent.probe.extract import Extraction
from latent.probe.linear import (C2DecoderReadout, fit_grouped_logistic, fit_softmax, kfold_assign, raw_space,
                                 score_readout, train_frontier_probes, train_gsm_probes)


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """Tiny models run fastest single-threaded, and a shared CPU is often oversubscribed; restored afterwards."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_kfold_assign_disjoint_balanced():
    f = kfold_assign(23, 5, seed=0)
    assert set(f.tolist()) == set(range(5))
    assert max(np.bincount(f)) - min(np.bincount(f)) <= 1
    assert not np.array_equal(f, kfold_assign(23, 5, seed=1))


def test_grouped_logistic_recovers_separable():
    rng = np.random.default_rng(0)
    d, G = 10, 3
    w_true = rng.standard_normal((G, d))
    X = rng.standard_normal((600, d))
    g = rng.integers(0, G, 600)
    y = ((X * w_true[g]).sum(1) > 0).astype(float)
    W, b = fit_grouped_logistic(X, np.arange(600), g, y, G, l2=1e-3)
    Xt = rng.standard_normal((400, d)); gt = rng.integers(0, G, 400)
    yt = (Xt * w_true[gt]).sum(1) > 0
    acc = (((Xt * W[gt]).sum(1) + b[gt] > 0) == yt).mean()
    assert acc > 0.9


def test_softmax_recovers_separable():
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((5, 8)) * 4
    y = rng.integers(0, 5, 500)
    X = centers[y] + rng.standard_normal((500, 8))
    W, b = fit_softmax(X[:300], y[:300], 5, l2=1e-3)
    assert ((X[300:] @ W.T + b).argmax(1) == y[300:]).mean() > 0.9


def test_raw_space_logit_identity():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 6)) * 3 + 7
    mu, sd = X.mean(0), X.std(0)
    W, b = rng.standard_normal((2, 6)), rng.standard_normal(2)
    Wr, br = raw_space(W, b, mu, sd)
    assert np.allclose(((X - mu) / sd) @ W.T + b, X @ Wr.T + br)


def superposition_extraction(n=80, L=3, d=48, n_names=12, noise=0.1, seed=0):
    """Latent j = sum of name embeddings of F_{j+1} + noise: the T3 superposition, decodable by construction."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n_names, d))
    names = [f"c{i}" for i in range(n_names)]
    lat = np.zeros((n, L, d), np.float32)
    recs = []
    for i in range(n):
        sym = ["Root"] + list(rng.choice(names, size=7, replace=False))
        cand = list(range(1, 8))
        fr, cur = [], set()
        for j in range(L):
            cur = cur | set(rng.choice(cand, size=2, replace=False).tolist())
            fr.append(sorted(cur))
            lat[i, j] = sum(emb[names.index(sym[v])] for v in cur) + noise * rng.standard_normal(d)
        recs.append({"idx": i, "task": "prosqa", "symbols": sym, "candidates": cand, "frontier": fr,
                     "hops": [j + 1 for j in range(L)], "target": 1, "neg_target": 2, "root": 0})
    return Extraction(latents=lat, n_latent=np.full(n, L), records=recs), emb, names


def test_frontier_probes_decode_superposition():
    ext, _, _ = superposition_extraction()
    fp = train_frontier_probes(ext, k=4, l2=1e-2, seed=0)
    for s in fp.metrics["per_step"]:
        assert s["f1"] > 0.9, s
        assert s["decoded"] and s["shuffled_f1"] < s["f1"] - 0.2
        assert s["f1_ci95"][0] <= s["f1"] <= s["f1_ci95"][1]
    # out-of-fold: the probe used for example i was trained without i's fold
    assert fp.fold.shape == (80,) and fp.trained.any()


def test_frontier_probes_noise_not_decoded():
    ext, _, _ = superposition_extraction(seed=3)
    ext.latents = np.random.default_rng(0).standard_normal(ext.latents.shape).astype(np.float32)
    fp = train_frontier_probes(ext, k=4, seed=0)
    assert not any(s["decoded"] for s in fp.metrics["per_step"])


def test_gsm_probes_decode_separable():
    rng = np.random.default_rng(4)
    n, L, d = 120, 4, 16
    centers = rng.standard_normal((6, d)) * 5
    lat = np.zeros((n, L, d), np.float32)
    recs = []
    for i in range(n):
        toks = rng.integers(0, 6, size=2).tolist()
        for s, tk in enumerate(toks):
            lat[i, 2 * s + 1] = centers[tk] + rng.standard_normal(d)
        recs.append({"idx": i, "task": "gsm", "latent_step": [-1, 0, -1, 1], "result_first_token": toks,
                     "results": [str(x) for x in toks]})
    ext = Extraction(latents=lat, n_latent=np.full(n, L), records=recs)
    out = train_gsm_probes(ext, k=4, seed=0)
    assert [s["step"] for s in out["per_step"]] == [1, 3]
    assert all(s["accuracy"] > 0.9 and s["decoded"] for s in out["per_step"])


def test_r3_interface_oracle_and_adapter():
    ext, emb, names = superposition_extraction(n=30)

    def oracle(latent, hop, record):  # a perfect readout: looks the answer up
        i = next(k for k, r in enumerate(ext.records) if r is record)
        return record["frontier"][hop - 1] if np.allclose(ext.latents[i, hop - 1], latent.numpy()) else []
    m = score_readout(oracle, ext, seed=0)
    assert all(s["f1"] == 1.0 and s["decoded"] for s in m["per_step"])

    def fn(lat, hops):  # a C2-shaped decoder: logits over symbol slots
        return torch.zeros(lat.shape[0], 8)
    r = C2DecoderReadout(fn, "prosqa")
    assert r(torch.zeros(48), 1, ext.records[0]) == []  # sigmoid(0) = 0.5 is not above threshold
