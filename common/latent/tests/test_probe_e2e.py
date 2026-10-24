"""End to end: the full readout on a tiny random GPT-2 with the FakeTokenizer (ProsQA-format fixture and GSM), the
synthetic positive control with a tiny budget, and the run_probes.py CLI with an injected model."""
import importlib.util
import json
import os

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent.model import LatentLM
from latent.probe.extract import load_extraction
from latent.probe.report import REPORT_FIELDS, run_readout, write_report
from latent.probe.synthetic import run_positive_control
from latent.prosqa_gen import generate
from latent.tests.fake_tokenizer import FakeTokenizer


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CLI = os.path.join(REPO, "experiments", "2026-10-17-silent-reasoning", "code", "run_probes.py")
FAST = dict(alphas=(1.0, 4.0), k_folds=3, batch_size=16, max_new_tokens=8)


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """Tiny models run fastest single-threaded, and a shared CPU is often oversubscribed; restored afterwards."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def tiny(t, seed=0):
    torch.manual_seed(seed)
    ids = {k: t.convert_tokens_to_ids(v) for k, v in (("latent_id", "<|latent|>"), ("start_id", "<|start-latent|>"),
                                                      ("end_id", "<|end-latent|>"))}
    m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=len(t) + 8, n_positions=512))
    return LatentLM(m, ids["latent_id"], ids["start_id"], ids["end_id"], t.eos_token_id).eval()


@pytest.fixture(scope="module")
def prosqa(tmp_path_factory):
    p = tmp_path_factory.mktemp("pq") / "prosqa_test.json"
    p.write_text(json.dumps(generate(24, depth=3, seed=5)))
    t = FakeTokenizer()
    D.add_latent_tokens(t)
    ex = D.load_examples(str(p))
    D.tokenize_examples(ex, t)
    return str(p), t, ex, tiny(t)


def check_report(rep):
    for k in REPORT_FIELDS:
        assert k in rep, k
    lb = rep["load_bearing"]
    for k in ("coverage", "load_bearing_fraction", "load_bearing_over_all_steps", "label", "per_step"):
        assert k in lb
    assert lb["label"] in ("explained", "partly explained", "unexplained")
    json.dumps(rep, default=float)


def test_prosqa_readout_end_to_end(prosqa, tmp_path):
    _, t, ex, lm = prosqa
    rep = run_readout(lm, t, ex, task="prosqa", n_latent=3, out_dir=str(tmp_path), **FAST)
    check_report(rep)
    assert [s["step"] for s in rep["R2"]["per_step"]] == [0, 1, 2]
    assert set(rep["F1"]["by_rule"]) == {"target_swap", "root_swap"}
    f1 = rep["F1"]["by_rule"]["target_swap"]
    assert f1["n_pairs"] > 0 and [s["step"] for s in f1["per_step"]] == [0, 1, 2, "all"]
    for s in f1["per_step"]:
        assert abs(s["follows"]["rate"] + s["stays"]["rate"] + s["breaks"]["rate"] - 1) < 1e-9
    assert set(rep["F3"]["per_kind"]) == {"add_neg", "remove_target"}
    assert all(len(r["per_alpha"]) == 2 for r in rep["F3"]["per_kind"]["add_neg"] if r["n_editable"])
    assert "drop" in rep["F2"] and len(rep["R1"]["per_step"]) == 3
    paths = write_report(rep, str(tmp_path))
    md = open(paths["md"]).read()
    assert "load-bearing fraction" in md and chr(0x2014) not in md
    ext = load_extraction(str(tmp_path))
    assert ext.latents.shape == (24, 3, 32) and len(ext.records) == 24


def test_prosqa_recompute_downstream_runs(prosqa):
    _, t, ex, lm = prosqa
    rep = run_readout(lm, t, ex[:12], task="prosqa", n_latent=3, downstream="recompute", lb_intervention="F1",
                      edit_kinds=("add_neg",), pair_rules=("root_swap",), **FAST)
    check_report(rep)
    assert rep["load_bearing"]["intervention"] == "F1" and rep["load_bearing"]["secondary"]["intervention"] == "F3"


def test_gsm_readout_end_to_end(tmp_path):
    recs = [{"question": f"Tom has {a} apples and gets {b} more . How many ?", "steps": [f"<<{a}+{b}={a + b}>>",
             f"<<{a + b}*2={2 * (a + b)}>>"], "answer": str(2 * (a + b))} for a, b in zip(range(1, 19), range(3, 21))]
    p = tmp_path / "gsm_test.json"
    p.write_text(json.dumps(recs))
    t = FakeTokenizer()
    D.add_latent_tokens(t)
    ex = D.load_examples(str(p))
    D.tokenize_examples(ex, t)
    rep = run_readout(tiny(t), t, ex, task="gsm", n_latent=4, c_thought=2, **FAST)
    check_report(rep)
    assert [s["step"] for s in rep["R2"]["per_step"]] == [1, 3]
    assert "skipped_reason" in rep["F1"] and "skipped_reason" in rep["F3"]
    assert rep["load_bearing"]["load_bearing_fraction"] is None and rep["load_bearing"]["label"] == "unexplained"


def test_positive_control_pipeline_runs(tmp_path):
    rep = run_positive_control(str(tmp_path), n_train=64, n_probe=30, steps=8, batch_size=16,
                               readout_kw=dict(alphas=(2.0,), k_folds=3))
    check_report(rep)
    assert rep["meta"]["task"] == "reach" and rep["training"]["steps"] == 8
    assert os.path.exists(tmp_path / "probe_report.json") and os.path.exists(tmp_path / "probe_report.md")
    assert "toggle_query" in rep["F3"]["per_kind"]


def test_cli_with_injected_model(prosqa, tmp_path, monkeypatch):
    path, t, ex, lm = prosqa
    spec = importlib.util.spec_from_file_location("run_probes", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_model", lambda cfg, ckpt, device, model_path=None: (lm, t))
    cfg = os.path.join(REPO, "experiments", "2026-10-17-silent-reasoning", "configs", "prosqa_coconut.json")
    out = tmp_path / "run"
    rep = mod.main(["--checkpoint", "unused.pt", "--config", cfg, "--split", path, "--task", "prosqa", "--out",
                    str(out), "--device", "cpu", "--n-latent", "3", "--max-examples", "12", "--max-new-tokens", "8",
                    "--alphas", "2", "--k-folds", "3", "--skip", "F1"])
    check_report(rep)
    assert json.load(open(out / "probe_report.json"))["meta"]["split"] == path
    assert (out / "probe_report.md").exists() and (out / "latents.npz").exists()


def fake_r3(path, lm, tok, task):
    """A loader with the --r3-loader signature: a C2-shaped decoder that marks slot 0..2 in every frontier."""
    from latent.probe.linear import C2DecoderReadout

    def fn(lat, hops):
        z = torch.full((lat.shape[0], 64), -5.0)
        z[:, :3] = 5.0
        return z
    return C2DecoderReadout(fn, task)


def test_r3_readout_via_cli_loader(prosqa, tmp_path, monkeypatch):
    path, t, ex, lm = prosqa
    spec = importlib.util.spec_from_file_location("run_probes", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_model", lambda cfg, ckpt, device, model_path=None: (lm, t))
    cfg = os.path.join(REPO, "experiments", "2026-10-17-silent-reasoning", "configs", "prosqa_coconut.json")
    rep = mod.main(["--checkpoint", "unused.pt", "--config", cfg, "--split", path, "--task", "prosqa", "--out",
                    str(tmp_path / "r3"), "--device", "cpu", "--n-latent", "3", "--max-examples", "12",
                    "--max-new-tokens", "6", "--alphas", "2", "--k-folds", "3", "--skip", "F1,F2",
                    "--r3", "unused.pt", "--r3-loader", "latent.tests.test_probe_e2e:fake_r3", "--decoded-by", "R3"])
    assert rep["load_bearing"]["readout"] == "R3"
    assert [s["step"] for s in rep["R3"]["per_step"]] == [0, 1, 2]
    # the decoder ignores the latent, so it never beats its own wrong-latent baseline
    assert not any(s["decoded"] for s in rep["R3"]["per_step"]) and rep["load_bearing"]["coverage"] == 0.0
