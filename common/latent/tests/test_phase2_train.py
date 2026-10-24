"""End-to-end CPU smoke of every Phase 2 arm (C1b k-band, C2, C2-shuffled, C3, C3-foreign, C2+C3) on tiny
fixtures for 4 epochs (stage 0 included), plus the resume and numerics rules of registration A3.2 / A3.6:
D17 (a resumed cot run keeps its AdamW moments: epoch-3 loss bit-identical to an uninterrupted run), D18 (TF32 flags
logged and asserted), D19 (every resume logged with its checkpoint sha256; runs_index_row.csv)."""
import csv
import json
import os

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent import prosqa_gen as G
from latent import train as T
from latent.tests.fake_tokenizer import FakeTokenizer

WORDS = "alex bob wumpus yumpus zumpus rompus tumpus lempus is a every not".split()


def gsm_like(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a, b, c = (int(x) for x in rng.integers(1, 9, 3))
        steps = [f"<<{a}+{b}={a + b}>>", f"<<{a + b}*{c}={(a + b) * c}>>"][: 1 + i % 2]
        out.append({"question": " ".join(rng.choice(WORDS, size=5)) + f" {a} {b} {c} ?", "steps": steps,
                    "answer": steps[-1].split("=")[-1][:-2]})
    return out


def prosqa_like(n, seed):
    import random
    rng = random.Random(seed)
    return [G.generate_record(rng, depth=2 + i % 2, n_nodes=8, n_extra_entities=0, n_distractor_edges=5)
            for i in range(n)]


@pytest.fixture()
def files(tmp_path):
    out = {}
    for kind, fn in (("gsm", gsm_like), ("prosqa", prosqa_like)):
        tr, va = tmp_path / f"{kind}_train.json", tmp_path / f"{kind}_valid.json"
        tr.write_text(json.dumps(fn(16, 0)))
        va.write_text(json.dumps(fn(5, 1)))
        out[kind] = (str(tr), str(va))
    return out, tmp_path


def base(paths, n_layer=2):
    t = FakeTokenizer()
    D.add_latent_tokens(t)
    for p in paths:
        D.tokenize_examples(D.load_examples(p), t)
    for ex in D.load_examples(paths[0]):    # symbol tokens of the C2 targets
        for s in ex.meta.get("idx_to_symbol", []):
            t.encode(" " + s)
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=n_layer, n_embd=32, n_head=2, vocab_size=len(t) + 2,
                                      n_positions=512)), t


def cfg(files, name, kind="gsm", **kw):
    (paths, tmp) = files
    tr, va = paths[kind]
    d = dict(name=name, arm="coconut", model_id="unused", train_path=tr, val_path=va, c_thought=1,
             epochs_per_stage=1, max_latent_stage=2, num_epochs=4, batch_size=4, grad_accum=2, device="cpu",
             save_dir=str(tmp / name), max_new_tokens=3, eval_batch_size=8, log_every=0)
    d.update(kw)
    return T.TrainConfig.from_dict(d)


def run(c, n_layer=2):
    m, t = base([c.train_path, c.val_path], n_layer)
    return T.train(c, model=m, tokenizer=t)


def rows(c):
    with open(os.path.join(c.save_dir, "log.jsonl")) as f:
        return [json.loads(x) for x in f]


ARMS = {
    "c1b": dict(latent_band=[1, 3]),
    "c2": dict(state_sup=True),
    "c2_shuffled": dict(state_sup=True, state_null="shuffled"),
    "c3": dict(dep_loss=True),
    "c3_foreign": dict(dep_loss=True, dep_null="foreign"),
    "c2c3": dict(state_sup=True, dep_loss=True),
}


@pytest.mark.parametrize("kind", ["gsm", "prosqa"])
@pytest.mark.parametrize("arm", list(ARMS))
def test_phase2_arm_end_to_end(files, arm, kind):
    if arm == "c1b" and kind == "prosqa":
        pytest.skip("one task suffices for the band smoke")
    c = cfg(files, f"{arm}_{kind}", kind=kind, c_thought=1 if kind == "prosqa" else 2, **ARMS[arm])
    out = run(c, n_layer=4 if arm == "c1b" else 2)
    r = rows(c)
    assert [x["stage"] for x in r] == [0, 1, 2, 3] and all(np.isfinite(x["train_loss"]) for x in r)
    assert all(x["numerics"]["float32_matmul_precision"] == "highest" for x in r)   # D18, fp32
    sd = torch.load(os.path.join(c.save_dir, "ckpt_last.pt"))
    assert not any(k.startswith(("base_causallm.", "inp.", "tok_heads", "sym_head")) for k in sd)
    assert out["best_final_epoch"] is not None
    if ARMS[arm].get("state_sup"):
        assert "train_state_loss" not in r[0] and all("train_state_loss" in x for x in r[1:])
        assert r[1]["flops_state_decoder"] > 0
        st = json.load(open(os.path.join(c.save_dir, "config.json")))["_state_targets"]
        assert 0.0 <= st["answer_in_last_target_frac"] <= 1.0 and st["kind"] == kind
        assert os.path.exists(os.path.join(c.save_dir, "state_decoder_last.pth"))
        assert os.path.exists(os.path.join(c.save_dir, "state_decoder_best_final.pth"))
        if ARMS[arm].get("state_null") == "shuffled":
            assert "answer_in_last_used_target_frac" in st
    if ARMS[arm].get("dep_loss"):
        assert "train_dep_hinge" not in r[0] and all("train_dep_hinge" in x for x in r[1:])
        passes = 3 if ARMS[arm].get("dep_null") == "foreign" else 2
        assert r[0]["flops_extra_passes"] == 0
        for x in r[1:]:   # every extra pass is charged (training-heavy)
            assert x["flops_extra_passes"] == pytest.approx((passes - 1) / passes * (x["flops"] -
                                                                                   x["flops_state_decoder"]))
    if arm == "c1b":
        assert all(x["layer_fraction"] == 0.5 for x in r)
        assert r[0]["val_latent_step_ms"] is None and r[2]["val_latent_step_ms"] > 0 and r[2]["val_token_step_ms"] > 0


def test_band_training_flops_are_cheaper(files):
    full = run(cfg(files, "full", num_epochs=3), n_layer=4)["rows"]
    band = run(cfg(files, "band", num_epochs=3, latent_band=[1, 3]), n_layer=4)["rows"]
    assert full[0]["flops"] == band[0]["flops"]           # stage 0: no latents
    assert band[2]["flops"] < full[2]["flops"]


def test_phase2_fields_need_a_latent_arm(files):
    for kw in (dict(arm="cot", state_sup=True), dict(arm="pause", latent_band=[0, 1]),
               dict(dep_loss=True, dep_null="placebo"), dict(state_sup=True, state_null="random")):
        with pytest.raises(ValueError):
            run(cfg(files, "bad", **kw))


def test_resume_keeps_the_optimizer_moments_d17_and_logs_d19(files):
    """D17: cot (reset_optimizer False) run 3 epochs straight vs 2 epochs then a resume to 3: the epoch-3 train loss
    is bit-identical (CPU), which needs the AdamW moments; the same resume without them differs. D19: the resume is
    logged with the checkpoint sha256 and counted in runs_index_row.csv."""
    kw = dict(arm="cot", reset_optimizer=False, save_only_improve=False, c_thought=0, max_latent_stage=0)
    straight = run(cfg(files, "straight", num_epochs=3, **kw))["rows"]
    c2 = cfg(files, "resumed", num_epochs=2, **kw)
    run(c2)
    st = json.load(open(os.path.join(c2.save_dir, "ckpt_state.json")))
    assert st["optimizer_path"] == T.OPTIMIZER_FILE and st["latest_sha256"] == T.sha256_file(st["latest_path"])
    c3 = cfg(files, "resumed", num_epochs=3, **kw)
    resumed = run(c3)["rows"]
    assert resumed[0]["epoch"] == 3 and resumed[0]["train_loss"] == straight[2]["train_loss"]
    rf = resumed[0]["resumed_from"]
    assert rf["after_epoch"] == 2 and rf["sha256"] == st["latest_sha256"] and rf["optimizer_restored"]
    with open(os.path.join(c3.save_dir, "runs_index_row.csv")) as f:
        row = next(csv.DictReader(f))
    assert (row["resumes"], row["final_epoch"], row["fragmented"], row["status"]) == ("1", "3", "False", "done")
    # negative control: drop the optimizer state from the resume point
    c4 = cfg(files, "fresh_opt", num_epochs=2, **kw)
    run(c4)
    p = os.path.join(c4.save_dir, "ckpt_state.json")
    s4 = json.load(open(p))
    s4.pop("optimizer_path")
    json.dump(s4, open(p, "w"))
    fresh = run(cfg(files, "fresh_opt", num_epochs=3, **kw))["rows"]
    assert fresh[0]["train_loss"] != straight[2]["train_loss"]


def test_resume_refuses_a_changed_checkpoint(files):
    c = cfg(files, "tamper", num_epochs=1)
    run(c)
    torch.save({"x": torch.zeros(1)}, os.path.join(c.save_dir, "ckpt_last.pt"))
    with pytest.raises(ValueError):
        run(cfg(files, "tamper", num_epochs=2))


def test_numerics_flags_d18():
    st = T.set_numerics("fp32")
    assert st == {"cuda.matmul.allow_tf32": False, "cudnn.allow_tf32": False, "float32_matmul_precision": "highest"}
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with pytest.raises(RuntimeError):
            T.numerics_state("fp32")
        assert T.numerics_state("fp16")["cuda.matmul.allow_tf32"] is True     # fp16 rows log, do not assert
        with pytest.raises(RuntimeError):
            T.numerics_state("bf16")                    # bf16 asserts the fp32-tf32 set (A10.5): cudnn is off here
    finally:
        T.set_numerics("fp32")


def test_bf16_registered_flags_a10_5():
    """Amendment A10.5 (A4 item 4): bf16 autocast runs on top of the oracle's TF32 setting."""
    assert T.REGISTERED_FLAGS["bf16"] == (True, True, "high") == T.REGISTERED_FLAGS["fp32-tf32"]
    try:
        T.set_numerics("fp32")
        st = T.set_numerics("bf16")
        assert st == {"cuda.matmul.allow_tf32": True, "cudnn.allow_tf32": True, "float32_matmul_precision": "high"}
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision()) == (True, True, "high")
        assert T.numerics_state("bf16") == st                                  # the per-epoch assertion passes
    finally:
        T.set_numerics("fp32")
