"""latent.probe.edit: edit direction math, the load-bearing rule and fractions, the registration 7.9 labels."""
import numpy as np
import pytest

from latent.probe.edit import edit_delta, f1_load_bearing, load_bearing, random_unit, step_load_bearing
from latent.probe.linear import FrontierProbes


def probes_with(w, b=0.3, scale=2.0):
    d = len(w)
    return FrontierProbes(names=["s"], fold=np.zeros(1, np.int64), W_raw=np.asarray(w, np.float32).reshape(1, 1, d),
                          b_raw=np.full((1, 1), b, np.float32), trained=np.ones((1, 1), bool),
                          scale=np.full((1, 1), scale, np.float32), rows={})


@pytest.mark.parametrize("alpha,sign", [(1.0, 1.0), (4.0, 1.0), (2.0, -1.0)])
def test_edit_moves_probe_logit_by_alpha_scale_norm(alpha, sign):
    rng = np.random.default_rng(0)
    w = rng.standard_normal(16)
    fp = probes_with(w)
    u, sc = fp.direction(0, "s")
    assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-6) and np.allclose(u, w / np.linalg.norm(w), atol=1e-6)
    h = rng.standard_normal(16).astype(np.float32)
    delta = edit_delta(u, sc, alpha, sign)
    assert np.isclose(np.linalg.norm(delta), alpha * sc, rtol=1e-5)
    assert np.isclose(np.dot(delta, u), sign * alpha * sc, rtol=1e-5)  # parallel to the probe direction
    gain = fp.logit(0, "s", h + delta) - fp.logit(0, "s", h)
    assert np.isclose(gain, sign * alpha * sc * np.linalg.norm(w), rtol=1e-4)


def test_direction_none_when_untrained_or_unknown():
    fp = probes_with(np.ones(4))
    fp.trained[:] = False
    assert fp.direction(0, "s") is None and fp.direction(0, "missing") is None


def test_random_unit_seeded():
    a, b = random_unit(32, (0, 1, 2)), random_unit(32, (0, 1, 2))
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)
    assert not np.allclose(a, random_unit(32, (0, 1, 3)))


def rows(effects, cis):
    return [{"step": t, "per_alpha": [{"alpha": a, "effect": e, "effect_ci": ci} for a, e, ci in zip((1, 2), es, cs)]}
            for t, (es, cs) in enumerate(zip(effects, cis))]


def test_step_load_bearing_rule():
    r = rows([(0.02, 0.30), (0.10, 0.04), (0.20, 0.0)],
             [([-0.01, 0.05], [0.10, 0.50]), ([0.01, 0.2], [0.0, 0.1]), ([-0.05, 0.4], [0.0, 0.0])])
    lb = step_load_bearing(r, min_effect=0.05)
    assert lb[0] == (True, 2, 0.30)      # alpha 2 passes (effect 0.30, CI excludes 0)
    assert lb[1] == (True, 1, 0.10)
    assert lb[2][0] is False             # big effect but CI includes 0
    assert step_load_bearing(r, min_effect=0.5)[0][0] is False


def test_f1_load_bearing_ignores_all_row():
    out = {"per_step": [{"step": 0, "effect_vs_null": 0.2, "effect_ci95": [0.1, 0.3]},
                        {"step": 1, "effect_vs_null": 0.01, "effect_ci95": [0.0, 0.02]},
                        {"step": "all", "effect_vs_null": 0.9, "effect_ci95": [0.8, 1.0]}]}
    assert f1_load_bearing(out) == {0: (True, None, 0.2), 1: (False, None, 0.01)}


def test_load_bearing_fractions_and_labels():
    w = {0: 100, 1: 100, 2: 100, 3: 100}
    res = load_bearing({0: True, 1: True, 2: True, 3: False}, w, {0: True, 1: True, 2: False, 3: True})
    assert np.isclose(res["coverage"], 0.75)
    assert np.isclose(res["load_bearing_fraction"], 2 / 3)        # among decoded steps (registration 7.9)
    assert np.isclose(res["load_bearing_over_all_steps"], 0.5)    # decoded and moved over all steps (brief)
    assert res["label"] == "explained"
    assert load_bearing({0: True, 1: False, 2: False, 3: False}, w, {0: True})["label"] == "partly explained"
    assert load_bearing({0: True, 1: True, 2: True, 3: True}, w, {0: True})["label"] == "partly explained"
    assert load_bearing({0: True, 1: True}, w, {})["label"] == "unexplained"
    none = load_bearing({}, w, {})
    assert none["coverage"] == 0.0 and none["load_bearing_fraction"] == 0.0


def test_load_bearing_weights_by_examples():
    res = load_bearing({0: True, 1: True}, {0: 300, 1: 100}, {0: (True, 1, 0.2), 1: (False, None, None)})
    assert np.isclose(res["load_bearing_fraction"], 0.75)
