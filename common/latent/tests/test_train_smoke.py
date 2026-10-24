"""latent.train end to end on a random tiny GPT-2 and the FakeTokenizer: stage schedule, logging, checkpoints,
optimizer reset semantics, resume, determinism; and every experiment config loads into TrainConfig."""
import glob
import importlib.util
import json
import os

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent import train as T
from latent.tests.fake_tokenizer import FakeTokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP = os.path.join(REPO, "experiments", "2026-10-17-silent-reasoning")
WORDS = "alex bob wumpus yumpus zumpus rompus tumpus lempus is a every not".split()


def make(n, seed):
    rng = np.random.default_rng(seed)
    return [{"question": " ".join(rng.choice(WORDS, size=5)) + " ?",
             "steps": [" ".join(rng.choice(WORDS, size=3)) + " ." for _ in range(2 + i % 2)],
             "answer": " ".join(rng.choice(WORDS, size=2)) + " ."} for i in range(n)]


@pytest.fixture()
def data(tmp_path):
    tr, va = tmp_path / "train.json", tmp_path / "valid.json"
    tr.write_text(json.dumps(make(20, 0)))
    va.write_text(json.dumps(make(6, 1)))
    return str(tr), str(va), tmp_path


def base(paths):
    t = FakeTokenizer()
    D.add_latent_tokens(t)
    for p in paths:
        D.tokenize_examples(D.load_examples(p), t)  # grow the fake vocabulary first
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=len(t), n_positions=256)), t


def cfg(data, name, **kw):
    tr, va, tmp = data
    d = dict(name=name, arm="coconut", model_id="unused", train_path=tr, val_path=va, c_thought=1,
             epochs_per_stage=1, max_latent_stage=2, num_epochs=3, batch_size=4, grad_accum=2, device="cpu",
             save_dir=str(tmp / name), max_new_tokens=4, eval_batch_size=4, log_every=0)
    d.update(kw)
    return T.TrainConfig.from_dict(d)


def rows(c):
    with open(os.path.join(c.save_dir, "log.jsonl")) as f:
        return [json.loads(x) for x in f]


def run(data, c):
    m, t = base([c.train_path, c.val_path])
    return T.train(c, model=m, tokenizer=t)


def test_coconut_epoch_loop_logs_and_checkpoints(data):
    c = cfg(data, "coco", num_epochs=4)  # fixture steps are 2 or 3; D16 final regime starts at stage 3
    out = run(data, c)
    r = rows(c)
    assert [x["epoch"] for x in r] == [1, 2, 3, 4] and [x["stage"] for x in r] == [0, 1, 2, 3]
    for x in r:
        assert x["tokens"] > 0 and x["flops"] > 0 and x["opt_steps"] == 3 and x["n_train"] == 20
        assert np.isfinite(x["train_loss"]) and np.isfinite(x["val_loss"]) and 0 <= x["val_acc"] <= 1
    assert os.path.exists(os.path.join(c.save_dir, "ckpt_last.pt"))
    # A2 / D16: final regime = stage >= min(max_latent_stage + 1, max steps in train) = min(3, 3) = 3
    assert [x["final_stage"] for x in r] == [False, False, False, True]
    assert "ckpt_best_final.pt" in r[3]["saved"] and os.path.exists(os.path.join(c.save_dir, "ckpt_best_final.pt"))
    assert "ckpt_best.pt" in r[0]["saved"] and all(x["max_new_tokens"] == 4 for x in r)
    assert out["best_final_epoch"] == 4
    assert json.load(open(os.path.join(c.save_dir, "config.json")))["arm"] == "coconut"
    sd = torch.load(os.path.join(c.save_dir, "ckpt_last.pt"))
    assert not any(k.startswith("base_causallm.") for k in sd)  # plain HF keys
    assert out["rows"] == r


def test_auto_resume_and_explicit_resume(data):
    c = cfg(data, "res", num_epochs=2)
    run(data, c)
    c2 = cfg(data, "res", num_epochs=3)  # same save_dir: continue after epoch 2 as run.py does
    run(data, c2)
    assert [x["epoch"] for x in rows(c2)] == [1, 2, 3] and rows(c2)[-1]["stage"] == 2
    ck = os.path.join(c.save_dir, "ckpt_last.pt")
    bad = cfg(data, "bad", num_epochs=3, resume=2, load_model_path=ck, load_model_sha256="0" * 64)
    with pytest.raises(ValueError):
        run(data, bad)
    # GSM-coconut style (registration D15): start from a named, hashed checkpoint and skip the first epochs; the
    # stage arithmetic is unchanged and only the D16 final regime (stage >= 3 on this fixture) enters the pool
    c3 = cfg(data, "skip", num_epochs=4, resume=1, load_model_path=ck, load_model_sha256=T.sha256_file(ck),
             load_model_epoch=2)
    out = run(data, c3)
    assert [(x["epoch_index"], x["stage"], x["final_stage"]) for x in rows(c3)] == [(1, 1, False), (2, 2, False),
                                                                                    (3, 3, True)]
    assert out["best_final_epoch"] == 4


def test_checkpoint_selection_rule(data, monkeypatch):
    """Registration 3.3: ckpt_best_final = best valid accuracy among final-stage epochs, strict > (ties to the
    earliest); ckpt_best = best over all epochs."""
    from latent.eval import ExampleResult
    accs = iter([0.9, 0.5, 0.4, 0.6, 0.6, 0.5])

    def fake_eval(*a, **k):
        n = round(next(accs) * 10)
        return [ExampleResult(idx=i, correct=i < n, pred="", gold="", text="", prefill_tokens=1, latent_steps=0,
                              emitted_tokens=1, sequential_passes=2, flops=1.0) for i in range(10)]

    monkeypatch.setattr(T, "evaluate", fake_eval)
    c = cfg(data, "sel", num_epochs=6)
    out = run(data, c)
    saved = [x["saved"] for x in rows(c)]
    assert [("ckpt_best.pt" in s_, "ckpt_best_final.pt" in s_) for s_ in saved] == [
        (True, False), (False, False), (False, False), (False, True), (False, False), (False, False)]
    assert (out["best_epoch"], out["best_final_epoch"], out["best_final_acc"]) == (1, 4, 0.6)


def test_max_new_tokens_must_be_explicit(data):
    with pytest.raises(ValueError):
        run(data, cfg(data, "nomnt", max_new_tokens=None))


@pytest.mark.parametrize("reset,mode,eps,expect", [(True, "epoch", 2, 4), (True, "stage", 2, 2), (False, "epoch", 2, 1)])
def test_optimizer_reset_semantics(data, monkeypatch, reset, mode, eps, expect):
    made = []
    real = torch.optim.AdamW

    class Counting(real):
        def __init__(self, *a, **k):
            made.append(k)
            super().__init__(*a, **k)

    monkeypatch.setattr(torch.optim, "AdamW", Counting)
    c = cfg(data, f"opt{reset}{mode}", reset_optimizer=reset, optimizer_reset=mode, epochs_per_stage=eps,
            num_epochs=4, max_new_tokens=1)
    run(data, c)
    assert len(made) == expect
    assert made[0] == {"lr": 1e-4, "weight_decay": 0.01}


def test_deterministic_given_seed(data):
    a = run(data, cfg(data, "d1", num_epochs=2, max_new_tokens=2))["rows"]
    b = run(data, cfg(data, "d2", num_epochs=2, max_new_tokens=2))["rows"]
    assert [x["train_loss"] for x in a] == [x["train_loss"] for x in b]
    c = run(data, cfg(data, "d3", num_epochs=2, max_new_tokens=2, seed=1))["rows"]
    assert [x["train_loss"] for x in a] != [x["train_loss"] for x in c]


@pytest.mark.parametrize("arm", ["cot", "no_cot", "pause", "coconut_no_curriculum"])
def test_other_arms_run(data, arm, monkeypatch):
    """Also registration D5: the embedding resize / "<<" initialisation happens for latent arms only."""
    calls = []
    real_init = T.init_latent_embeddings
    monkeypatch.setattr(T, "init_latent_embeddings", lambda *a, **k: calls.append(1) or real_init(*a, **k))
    c = cfg(data, arm, arm=arm, num_epochs=1, max_new_tokens=2, save_only_improve=True)
    m, t = base([c.train_path, c.val_path])
    T.train(c, model=m, tokenizer=t)
    r = rows(c)
    assert len(r) == 1
    assert r[0]["stage"] == 0  # scheduled stage; latent.data pins coconut_no_curriculum to the last stage
    assert r[0]["final_stage"] == (arm != "pause")
    assert not os.path.exists(os.path.join(c.save_dir, "ckpt_last.pt"))  # save_only_improve
    assert len(calls) == (0 if arm in ("cot", "no_cot") else 1)


def test_scheduled_stage_and_max_new_tokens():
    c = T.TrainConfig(name="x", arm="coconut", epochs_per_stage=5, val_path="data/prosqa_valid.json")
    assert [T.scheduled_stage(c, e) for e in (0, 4, 5, 34, 49)] == [0, 0, 1, 6, 9]
    assert T.scheduled_stage(T.TrainConfig(name="x", arm="cot"), 30) == 0
    assert T.default_max_new_tokens(c) == 128
    assert [T.is_final_stage(c, s, max_steps_train=6) for s in (5, 6, 9)] == [False, True, True]  # ProsQA: 3-6 steps
    assert [T.is_final_stage(c, s) for s in (5, 6, 7)] == [False, False, True]  # unknown steps: cap + 1 (D16)
    assert [T.is_final_stage(c, s, max_steps_train=4) for s in (3, 4)] == [False, True]
    assert T.default_max_new_tokens(T.TrainConfig(name="x", arm="cot", val_path="data/gsm_valid.json")) == 64


def _run_train_module():
    spec = importlib.util.spec_from_file_location("run_train", os.path.join(EXP, "code", "run_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_experiment_configs_load_and_mirror_the_reference():
    rt = _run_train_module()
    paths = sorted(glob.glob(os.path.join(EXP, "configs", "*.json")))
    names = {os.path.basename(p)[:-5] for p in paths}
    non_train = ("data_manifest.json", "reference_env.json")        # the manifest and the prep VM's environment record
    every = [p for p in paths if not p.endswith(non_train)] + \
        sorted(glob.glob(os.path.join(EXP, "configs", "screen", "*.json")))
    assert len(every) >= len(paths) - len(non_train) + 28
    for p in every:   # every config in configs/ and configs/screen/ loads into TrainConfig
        c = rt.load_config(p, ["device=cpu"])
        assert c.max_new_tokens == T.default_max_new_tokens(c) and c.name == os.path.basename(p)[:-5]
        T.check_phase2(c)
    want = {"prosqa_cot", "prosqa_no_cot", "prosqa_coconut", "prosqa_pause", "prosqa_coconut_no_curriculum",
            "prontoqa_coconut", "gsm_cot", "gsm_coconut", "gsm_no_cot", "smoke_prosqa", "prosqa_coconut_strict"}
    assert want <= names
    cs = {n: rt.load_config(os.path.join(EXP, "configs", n + ".json"), ["device=cpu"]) for n in want}
    for n, c in cs.items():
        want_prec = "fp32" if n == "prosqa_coconut_strict" else "fp32-tf32"  # A4: oracle fp32-tf32; pair strict
        assert (c.lr, c.weight_decay, c.batch_size * c.grad_accum, c.seed, c.precision) == (1e-4, 0.01, 128, 0, want_prec)
        assert c.max_new_tokens == (64 if n.startswith("gsm") else 128) == T.default_max_new_tokens(c)  # D3
        assert c.optimizer_reset == "epoch" and c.uniform_prob == 0.0
        assert c.reset_optimizer == (c.arm not in ("cot", "no_cot"))  # D2
        assert os.path.isabs(c.train_path) and c.train_path.startswith(REPO) and c.device == "cpu"
    pc = cs["prosqa_coconut"]
    assert (pc.c_thought, pc.epochs_per_stage, pc.max_latent_stage, pc.pad_latent_to_max, pc.num_epochs,
            pc.reset_optimizer, pc.save_only_improve) == (1, 5, 6, True, 50, True, False)
    for n in ("prosqa_pause", "prosqa_coconut_no_curriculum"):
        same = {k: v for k, v in vars(cs[n]).items() if k not in ("name", "arm", "save_dir")}
        assert same == {k: v for k, v in vars(pc).items() if k not in ("name", "arm", "save_dir")}
    g = cs["gsm_coconut"]
    assert (g.c_thought, g.epochs_per_stage, g.max_latent_stage, g.num_epochs, g.resume) == (2, 3, 3, 25, 3)
    assert g.load_model_path.endswith(os.path.join("runs", "gsm_cot", "ckpt_best_final.pt"))  # D15
    # A2 / D16: GSM8k-Aug has examples with more than 3 steps, so stage 3 (epochs 9-11) still keeps language steps;
    # the selection pool is epochs 12-24
    assert [e for e in range(g.resume, g.num_epochs) if T.is_final_stage(g, T.scheduled_stage(g, e), 20)] == \
        list(range(12, 25))
    assert [e for e in range(g.resume, g.num_epochs) if T.is_final_stage(g, T.scheduled_stage(g, e))] == \
        list(range(12, 25))
    assert cs["gsm_cot"].save_only_improve and not cs["gsm_cot"].reset_optimizer
    s = cs["smoke_prosqa"]
    assert (s.max_train_examples, s.num_epochs, s.max_val_examples) == (2000, 2, 100)


def test_run_train_requires_explicit_max_new_tokens(tmp_path):
    rt = _run_train_module()
    d = json.load(open(os.path.join(EXP, "configs", "smoke_prosqa.json")))
    d["max_new_tokens"] = None
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        rt.load_config(str(p))


SCREEN_ARMS = ("cot_lr3e-5", "cot_lr1e-4", "cot_lr3e-4", "no_cot", "coconut", "pause", "c1b_k8", "c1b_k4", "c2",
               "c2_shuffled", "c3", "c3_foreign", "c2c3", "plain_depth_355m")


def test_screen_configs_follow_the_registration():
    """Registration A3.3 / A3.4 / A3.4.5: the screen configs of both tasks."""
    rt = _run_train_module()
    for task in ("prosqa_hard", "gsm50k"):
        cs = {a: rt.load_config(os.path.join(EXP, "configs", "screen", f"{task}_{a}.json"), ["device=cpu"])
              for a in SCREEN_ARMS}
        for a, c in cs.items():
            # A13.2 (b): the 355M plain-depth control trains at micro-batch 16 x grad_accum 8 (effective 128 unchanged)
            # after its 32 x 1024 probe ran out of memory on the A100-80GB (deviations.md 2026-10-20 09:58 UTC)
            mb = (16, 8) if a == "plain_depth_355m" else (32, 4)
            assert (c.precision, c.seed, c.batch_size, c.grad_accum, c.weight_decay, c.uniform_prob) == \
                ("bf16", 0, *mb, 0.01, 0.0)
            assert c.batch_size * c.grad_accum == 128
            assert c.save_dir.endswith(os.path.join("runs", f"{task}_{a}"))
            assert c.max_new_tokens == (128 if task == "prosqa_hard" else 64)
            assert c.train_path.endswith("prosqa_hard_train.json" if task == "prosqa_hard" else "gsm50k_train.json")
            assert c.val_path.endswith("prosqa_hard_valid.json" if task == "prosqa_hard" else "gsm_valid.json")
            latent = a not in ("cot_lr3e-5", "cot_lr1e-4", "cot_lr3e-4", "no_cot", "plain_depth_355m")
            assert c.model_id == ("openai-community/gpt2-medium" if a == "plain_depth_355m" else "openai-community/gpt2")
            if latent:
                assert c.lr == 1e-4 and c.reset_optimizer and c.optimizer_reset == "epoch" and not c.save_only_improve
                assert c.arm == ("pause" if a == "pause" else "coconut")
                if task == "prosqa_hard":
                    assert (c.c_thought, c.epochs_per_stage, c.max_latent_stage, c.num_epochs, c.resume) == \
                        (1, 3, 7, 30, 0) and c.load_model_path is None
                    pool = [e for e in range(c.num_epochs) if T.is_final_stage(c, T.scheduled_stage(c, e), 7)]
                    assert pool == list(range(21, 30))
                else:
                    assert (c.c_thought, c.epochs_per_stage, c.max_latent_stage, c.num_epochs, c.resume) == \
                        (2, 3, 3, 25, 3)
                    assert c.load_model_path.endswith(os.path.join("runs", "gsm50k_cot_lr1e-4", "ckpt_best_final.pt"))
                    pool = [e for e in range(c.resume, c.num_epochs) if T.is_final_stage(c, T.scheduled_stage(c, e), 13)]
                    assert pool == list(range(12, 25))
            else:
                assert c.arm == ("cot" if a.startswith("cot") else "no_cot")
                assert not c.reset_optimizer and c.save_only_improve and c.load_model_path is None
                assert c.num_epochs == (30 if task == "prosqa_hard" else 25)
                assert c.lr == {"cot_lr3e-5": 3e-5, "cot_lr3e-4": 3e-4}.get(a, 1e-4)
        assert cs["c1b_k8"].latent_band == [2, 10] and cs["c1b_k4"].latent_band == [4, 8]
        assert all(cs[a].latent_band is None for a in SCREEN_ARMS if not a.startswith("c1b"))
        flags = {a: (cs[a].state_sup, cs[a].state_null, cs[a].dep_loss, cs[a].dep_null) for a in SCREEN_ARMS}
        assert flags["c2"] == (True, "none", False, "none") and flags["c2_shuffled"] == (True, "shuffled", False, "none")
        assert flags["c3"] == (False, "none", True, "none") and flags["c3_foreign"] == (False, "none", True, "foreign")
        assert flags["c2c3"] == (True, "none", True, "none") and flags["coconut"] == (False, "none", False, "none")
        assert all((c.state_lambda, c.dep_margin, c.dep_beta) == (0.1, 0.2, 1.0) for c in cs.values())
        # the latent arms differ from full-depth coconut only in their mechanism fields
        mech = ("name", "arm", "save_dir", "latent_band", "state_sup", "state_null", "dep_loss", "dep_null")
        ref = {k: v for k, v in vars(cs["coconut"]).items() if k not in mech}
        for a in ("pause", "c1b_k8", "c1b_k4", "c2", "c2_shuffled", "c3", "c3_foreign", "c2c3"):
            assert {k: v for k, v in vars(cs[a]).items() if k not in mech} == ref, a
