"""The experiment's Phase 2 tools: code/make_screen_data.py (registration A3.4.5), code/run_eval.py (registered-read
guard, per-example CSV without text, summary, band models, scrambled) and code/read_oracle.py (K1 table, refusal on
snippet targets). Tiny fixtures; no download (latent.train.load_base is replaced by a tiny model)."""
import csv
import hashlib
import importlib.util
import json
import os
import random

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent import prosqa_gen as G
from latent import train as T
from latent.tests.fake_tokenizer import FakeTokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP = os.path.join(REPO, "experiments", "2026-10-17-silent-reasoning")


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, "code", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prosqa_hard_generation_rule():
    msd = _mod("make_screen_data")
    out, info = msd.make_prosqa_hard(seed=0, splits=(("train", 40), ("valid", 6), ("test", 6)))
    again, _ = msd.make_prosqa_hard(seed=0, splits=(("train", 40), ("valid", 6), ("test", 6)))
    assert out == again and [len(out[s]) for s in ("train", "valid", "test")] == [40, 6, 6]
    keys = ["question", "answer", "steps", "idx_to_symbol", "edges", "root", "target", "neg_target"]
    qs = set()
    for recs in out.values():
        for r in recs:
            assert list(r) == keys and len(r["steps"]) in (5, 6, 7) and G.validate_record(r) == []
            d = G.bfs_dist(len(r["idx_to_symbol"]), r["edges"], r["root"])
            assert d[r["target"]] == len(r["steps"]) and r["neg_target"] not in d
            assert r["question"] not in qs
            qs.add(r["question"])
    # the doubled default draw: twice the generator's clip(round(gauss(37 - (depth - 3), 6)), 15, 63) - depth
    rng = random.Random(3)
    d0 = msd.default_distractor_draw(rng, 6)
    rng2 = random.Random(3)
    assert d0 == min(63, max(15, round(rng2.gauss(34.0, 6.0)))) - 6
    mean_edges = np.mean([len(r["edges"]) - len(r["steps"]) for r in out["train"]])
    assert 40 < mean_edges < 75 and info["redraws"] >= 0 and info["length_redraws"] == 0
    # the context cap (GPT-2 positions; a logged deviation): questions over the cap are redrawn at the same depth
    t = FakeTokenizer()
    capped, ci = msd.make_prosqa_hard(seed=0, splits=(("train", 30),), tokenizer=t, max_question_tokens=420)
    assert ci["length_redraws"] > 0 and all(len(t.encode(r["question"] + "\n")) <= 420 for r in capped["train"])
    assert msd.MAX_QUESTION_TOKENS == 1024 - 2 - 7 - 128


def test_gsm_subset_rule():
    msd = _mod("make_screen_data")
    recs = [{"question": f"q{i}", "steps": [], "answer": str(i)} for i in range(100)]
    sub, idx = msd.gsm_subset(recs, n=10, seed=0)
    assert idx == [int(i) for i in np.random.default_rng(0).permutation(100)[:10]]
    assert [r["answer"] for r in sub] == [str(i) for i in idx]
    assert msd.index_sha256(idx) == hashlib.sha256("\n".join(str(i) for i in sorted(idx)).encode()).hexdigest()


def test_screen_data_files_match_the_manifest():
    man = json.load(open(os.path.join(EXP, "configs", "data_manifest.json")))
    if "screen" not in man:
        pytest.skip("screen block not written yet")
    files = man["screen"]["prosqa_hard"]["files"] + [man["screen"]["gsm50k"]["train"]]
    for f in files:
        p = os.path.join(REPO, f["path"])
        if not os.path.exists(p):
            pytest.skip(f"{f['path']} absent")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == f["sha256"], f["path"]
    assert [f["records"] for f in man["screen"]["prosqa_hard"]["files"]] == [10000, 300, 500]
    assert man["screen"]["gsm50k"]["train"]["records"] == 50000
    assert man["screen"]["prosqa_hard"]["duplicate_questions_across_splits"] == 0
    cap = man["screen"]["prosqa_hard"]["context_cap"]
    assert cap["longest_cot_training_sequence_tokens"] <= 1024
    assert all(v["max"] <= cap["max_question_tokens"] for v in cap["question_tokens"].values())


# ------------------------------------------------------------------------------------------------- run_eval.py
def _fixture(tmp_path, n=6):
    WORDS = "alex bob wumpus yumpus zumpus rompus is a every".split()
    rng = np.random.default_rng(0)
    recs = [{"question": " ".join(rng.choice(WORDS, 5)) + " ?", "steps": ["a b .", "c d .", "e f ."],
             "answer": " ".join(rng.choice(WORDS, 2)) + " ."} for _ in range(n)]
    for split in ("train", "valid", "test"):
        (tmp_path / f"toy_{split}.json").write_text(json.dumps(recs))
    t = FakeTokenizer()
    D.add_latent_tokens(t)
    D.tokenize_examples(D.load_examples(str(tmp_path / "toy_train.json")), t)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=4, n_embd=32, n_head=2, vocab_size=len(t) + 2, n_positions=256))
    ck = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ck)
    return t, model, ck


@pytest.fixture()
def evalenv(tmp_path, monkeypatch):
    t, model, ck = _fixture(tmp_path)

    def fake_load_base(model_id, dtype=None, device="cpu", resize=True):
        m = GPT2LMHeadModel(model.config)
        m.load_state_dict(model.state_dict())
        tok = FakeTokenizer()
        tok.id_to_tok, tok.tok_to_id = list(t.id_to_tok), dict(t.tok_to_id)
        return m, tok
    monkeypatch.setattr(T, "load_base", fake_load_base)
    re_ = _mod("run_eval")
    log = tmp_path / "critic_log.md"
    log.write_text("| Logged (UTC) | Ruling |\n|---|---|\n| 2026-10-19T10:00Z | registered read of the toy |\n")
    monkeypatch.setattr(re_, "CRITIC_LOG", str(log))

    def cfg(name, **kw):
        d = dict(name=name, arm="coconut", model_id="toy", train_path=str(tmp_path / "toy_train.json"),
                 val_path=str(tmp_path / "toy_valid.json"), c_thought=1, max_latent_stage=3, max_new_tokens=128,
                 eval_batch_size=4, device="cpu", save_dir=str(tmp_path / name))
        d.update(kw)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d))
        return str(p)
    return re_, cfg, str(ck), tmp_path


def test_run_eval_writes_csv_and_summary(evalenv):
    re_, cfg, ck, tmp = evalenv
    out = tmp / "o1"
    s = re_.main(["--config", cfg("band", latent_band=[1, 3]), "--checkpoint", ck, "--out", str(out),
                  "--n-boot", "200"])
    rows = list(csv.DictReader(open(out / "examples.csv")))
    assert len(rows) == 6 and list(rows[0]) == list(re_.CSV_FIELDS)
    assert all(r["latent_steps"] == "3" and r["latent_positions"] == "3" for r in rows)  # final regime: 1 x 3
    summ = json.load(open(out / "summary.json"))
    assert summ == json.loads(json.dumps(s))
    for k in ("accuracy", "accuracy_ci95", "emitted_tokens_mean", "latent_steps_mean", "sequential_passes_mean",
              "flops_mean", "paper_style_tokens_mean"):
        assert k in summ["summary"]
    assert summ["variant"]["layer_fraction"] == 0.5 and summ["timing"]["latent_passes"] == 6
    assert summ["examples_csv_sha256"] == hashlib.sha256(open(out / "examples.csv", "rb").read()).hexdigest()
    scr = re_.main(["--config", cfg("band", latent_band=[1, 3]), "--checkpoint", ck, "--out", str(tmp / "o2"),
                    "--intervention", "scrambled", "--n_latent", "2", "--n-boot", "100"])
    assert scr["n_latent"] == 2 and scr["intervention"] == "scrambled"
    z = re_.main(["--config", cfg("full"), "--checkpoint", ck, "--out", str(tmp / "o3"), "--intervention",
                  "zero_latent", "--n-boot", "100"])
    assert z["n_latent"] == 0 and z["summary"]["latent_steps_mean"] == 0


def test_run_eval_test_split_guard(evalenv):
    re_, cfg, ck, tmp = evalenv
    c = cfg("full")
    with pytest.raises(SystemExit):
        re_.main(["--config", c, "--checkpoint", ck, "--split", "test"])
    with pytest.raises(SystemExit):
        re_.main(["--config", c, "--checkpoint", ck, "--split", "test", "--registered-read", "2026-01-01T00:00Z"])
    out = tmp / "t1"
    s = re_.main(["--config", c, "--checkpoint", ck, "--split", "test", "--registered-read", "2026-10-19T10:00Z",
                  "--out", str(out), "--n-boot", "100"])
    assert s["split"] == "test" and s["split_file"].endswith("toy_test.json")
    with pytest.raises(SystemExit):   # write-once
        re_.main(["--config", c, "--checkpoint", ck, "--split", "test", "--registered-read", "2026-10-19T10:00Z",
                  "--out", str(out)])


# ---------------------------------------------------------------------------------------------- read_oracle.py
def _summary(tmp, task, arm, seed, acc, **variant):
    d = tmp / f"{task}_{arm}_s{seed}"
    d.mkdir()
    s = {"task": task, "arm": arm, "seed": seed, "split": "test", "intervention": "none", "gpu_type": "A100",
         "variant": variant, "summary": {"accuracy": acc}}
    (d / "summary.json").write_text(json.dumps(s))


def _pdf_published(tmp):
    ro = _mod("read_oracle")
    pub = ro.snippet_template()
    pub.update(source="pdf", pdf_sha256="ab" * 32, critic_log_row="2026-10-19T09:00Z")
    for c in pub["cells"]:
        c["pdf_table"], c["pdf_row"] = "Table 1", c["arm"]
    p = tmp / "pub.json"
    p.write_text(json.dumps(pub))
    return ro, str(p)


def test_read_oracle_refuses_snippet_values(tmp_path):
    ro = _mod("read_oracle")
    _summary(tmp_path, "prosqa", "cot", 0, 0.75)
    glob_ = str(tmp_path / "*" / "summary.json")
    with pytest.raises(SystemExit):
        ro.main(["--summaries", glob_, "--out", str(tmp_path / "k1")])
    tpl = tmp_path / "tpl.json"
    ro.main(["--summaries", "none", "--write-template", str(tpl)])
    with pytest.raises(SystemExit):
        ro.main(["--published", str(tpl), "--summaries", glob_, "--out", str(tmp_path / "k1")])
    t = ro.main(["--published", str(tpl), "--summaries", glob_, "--out", str(tmp_path / "k1"), "--allow-snippet"])
    assert all("snippet" in r["labels"] for r in t["rows"])
    assert all("snippet" in s for k, s in t["sentences"].items() if k in ("prosqa", "gsm", "prontoqa"))


def test_read_oracle_k1_arithmetic(tmp_path):
    ro, pub = _pdf_published(tmp_path)
    for s, acc in enumerate((0.96, 0.97, 0.98)):
        _summary(tmp_path, "prosqa", "coconut", s, acc)
    for s, acc in enumerate((0.70, 0.72, 0.74)):
        _summary(tmp_path, "prosqa", "cot", s, acc)
    _summary(tmp_path, "prosqa", "no_cot", 0, 0.76)
    _summary(tmp_path, "prosqa", "pause", 0, 0.97)
    _summary(tmp_path, "prosqa", "coconut_no_curriculum", 0, 0.70)
    _summary(tmp_path, "prosqa", "coconut", 9, 0.10, state_sup=True)        # a Phase 2 arm never enters K1
    t = ro.main(["--published", pub, "--summaries", str(tmp_path / "*" / "summary.json"), "--out",
                 str(tmp_path / "k1")])
    rows = {(r["task"], r["arm"]): r for r in t["rows"]}
    c = rows[("prosqa", "coconut")]
    # SD 1.0 (df 2) pooled with the published spread 0.3 (df 2): sqrt((2 + 0.18) / 4) = 0.74 -> floor 1.0 -> tol 2.0
    assert c["n"] == 3 and c["sigma_cell"] == 1.0 and c["tolerance"] == 2.0 and c["verdict"] == "matched"
    cot = rows[("prosqa", "cot")]
    assert cot["sigma_cell"] == pytest.approx(2.0) and cot["tolerance"] == pytest.approx(4.0)
    assert cot["verdict"] == "missed"                                       # |72 - 77.5| = 5.5 > 4
    assert rows[("prosqa", "no_cot")]["verdict"] == "matched" and "single seed" in rows[("prosqa", "no_cot")]["labels"]
    nc = rows[("prosqa", "coconut_no_curriculum")]
    assert nc["published"] == pytest.approx(76.0) and nc["verdict"] == "missed"   # 70 vs our no_cot 76, tol 2
    assert rows[("gsm", "cot")]["verdict"] == "UNTESTED"
    assert t["sentences"]["prosqa"].startswith("K1 ProsQA: fail (cot, coconut_no_curriculum)")
    assert t["sentences"]["gsm"].startswith("K1 GSM8k-Aug: UNTESTED")
    assert "ProntoQA UNTESTED" in t["sentences"]["phase1_registered"]
    assert os.path.exists(tmp_path / "k1" / "k1_table.md")
