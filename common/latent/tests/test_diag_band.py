"""latent.diag.band_logits and code/run_diag.py --mode band_logits (A6.2 rule 3's strict-fp32 logits check; critic entry
2026-10-20T12:40Z, the registered diagnostic; A14.1): on a model whose band covers every layer the HF path and band
[0, L] agree within 1e-4 (check (1) passes, no stream diverges); a band-only departure makes (1) a BUG; the batch-1 vs
batch-B plumbing of (3) compares each example in its own file-order batch and carries positions and margins; the
same-shape swap of (4) shows a cross-row leak at both numerics; PathRecorder follows generate on both paths and
generate_rows scores as evaluate scores; the script writes a hashed write-once record with the numerics per check and
refuses what the row does not register. Tiny random GPT-2, CPU."""
import contextlib
import csv
import json

import pytest
import torch

from latent import diag
from latent import eval as E
from latent.model import LatentLM
from latent.tests import test_diag as TD

BS, MNT, N, NL, _sha = TD.BS, TD.MNT, TD.N, TD.NL, TD._sha
setup, diagenv, _restore_tf32 = TD.setup, TD.diagenv, TD._restore_tf32      # test_diag's fixtures (tiny model, toy
#                                                                              run_diag env, TF32 flags put back)

EXTRA = (7, 9)        # outside the first batch of BS = 4; 9 sits in a short last batch (8, 9), as 497 does at 500 / 32


@contextlib.contextmanager
def _plain():
    yield {"set": "plain"}


def _numerics():
    """run_diag.py's own numerics sets for a bf16 screen config: strict fp32, then bf16 autocast over TF32 on."""
    from latent.tests import test_screen_tools as sct
    return sct._mod("run_diag").numerics_sets("bf16", "cpu")


def _run(lm, t, ex, **kw):
    args = dict(arm="coconut", n_latent=NL, batch_size=BS, device="cpu", max_new_tokens=MNT, numerics=_numerics(),
                extra_idx=EXTRA)
    args.update(kw)
    return diag.band_logits(lm, t, ex, **args)


def test_band_model_shares_the_weights_and_covers_every_layer(setup):
    t, ex, lm = setup
    band = diag.band_model(lm)
    assert band.base_causallm is lm.base_causallm and band.latent_band == (0, 2) and band.layer_fraction == 1.0
    with pytest.raises(ValueError, match="full-depth"):
        diag.band_model(band)


def test_generate_rows_follow_evaluate_on_both_paths(setup):
    """The rows the checks compare are evaluate's own generations: same tokens and correctness per example at the
    read's batching and at batch 1, on the HF path and on band [0, L]; step 0's top-1 is the first emitted token."""
    t, ex, lm = setup
    recs = diag.prompt_records(t, ex, arm="coconut", n_latent=NL)
    by = {e.idx: e for e in ex}
    for m in (lm, diag.band_model(lm)):
        for bs in (BS, 1):
            ref = {r.idx: r for r in E.evaluate(m, t, ex, arm="coconut", n_latent=NL, batch_size=bs, device="cpu",
                                                max_new_tokens=MNT)}
            got = {}
            for s in range(0, N, bs):
                got.update(diag.generate_rows(m, t, recs[s:s + bs], by, mode="coconut", device="cpu",
                                              max_new_tokens=MNT))
            assert sorted(got) == list(range(N))
            for i, g in got.items():
                text = t.decode(recs[i]["input_ids"] + g["tokens"], skip_special_tokens=True)
                assert (text, g["correct"], len(g["tokens"])) == (ref[i].text, ref[i].correct, ref[i].emitted_tokens)
                assert g["B"] == len(recs[i // bs * bs:i // bs * bs + bs]) and g["step0_consistent"]
                assert int(g["top2_ids"][0, 0]) == g["stream"][0] == int(g["first"].argmax())
                assert g["latents"].shape == (NL, 32) and g["top2_values"].shape == (len(g["stream"]), 2)
        assert "generate" not in m.__dict__ and "_run" not in m.__dict__ and not m.lm_head._forward_hooks


def test_paths_agree_when_the_band_covers_every_layer(setup):
    t, ex, lm = setup
    rows, s = _run(lm, t, ex)
    c = s["checks"]
    assert s["band"] == [0, 2] and s["batch"]["idx"] == list(range(BS)) and s["extra_idx"] == list(EXTRA)
    assert c["1"]["verdict"] == "pass" and c["1"]["n"] == BS and c["1"]["first_answer_logits"]["max"] <= 1e-5
    assert c["1"]["diverging"] == 0 and c["1"]["first_argmax_differs"] == 0 and c["1"]["latents"]["max"] <= 1e-5
    assert c["1"]["correct"]["hf"] == c["1"]["correct"]["band"] and c["1"]["widths"] == [rows[0]["width_band"]]
    assert c["2"]["verdict"] == "printed (no line)" and c["2"]["n"] == BS
    for k in diag.BAND_NUMERICS:
        g = c["3"][k]
        assert g["n"] == BS + len(EXTRA) and g["diverging"] == 0 and g["of_batch"] == {"n": BS, "diverging": 0}
        assert set(g["extra"]) == {"7", "9"}
        assert (g["extra"]["7"]["batch_start"], g["extra"]["7"]["B_batch"]) == (4, 4)       # its file-order batch
        assert (g["extra"]["9"]["batch_start"], g["extra"]["9"]["B_batch"]) == (8, 2)       # a short last batch
        assert g["extra"]["9"]["first_margin_b1"] > 0 and g["extra"]["9"]["div_pos"] is None
        for q in ("last", "logits", "latents"):
            assert c["4"][k][f"{q}_swap"]["n"] == BS and c["4"][k][f"{q}_swap"]["over_tol"] == 0
            assert c["4"][k][f"{q}_rep"]["max"] == 0.0                     # fixed shape and content: bit for bit
    assert c["3"]["verdict"] == "0 diverging at strict fp32" and c["4"]["verdict"] == "within 0.0001 at each"
    assert s["bug"] == [] and s["swap_pool"] == N - BS
    for k in diag.BAND_NUMERICS:
        st = s["numerics"][k]
        assert st["torch_at_start"] == st["torch_at_end"]
        assert st["torch_at_start"]["autocast_enabled"] is (k == "registered")
    assert s["numerics"]["registered"]["torch_at_start"]["autocast_dtype"] == "bfloat16"
    per = {(k, n): sum(r["check"] == k and r["numerics"] == n for r in rows) for k in "1234"
           for n in diag.BAND_NUMERICS}
    assert per == {("1", "strict"): BS, ("1", "registered"): 0, ("2", "strict"): 0, ("2", "registered"): BS,
                   ("3", "strict"): BS + 2, ("3", "registered"): BS + 2, ("4", "strict"): BS, ("4", "registered"): BS}
    assert not lm.training and "generate" not in lm.__dict__


def test_check_1_is_a_bug_when_the_band_path_departs(setup, monkeypatch):
    """A band-only departure (one hidden unit of every manual block output moved by 0.01, which LayerNorm does not
    remove; the HF path never calls _block) is over the 1e-4 line: check (1) says BUG; B = 1 against B = batch and the
    swap see the same departure on both sides."""
    t, ex, lm = setup
    orig = LatentLM._block

    def moved(self, i, x, mask, kv):
        y, kv2 = orig(self, i, x, mask, kv)
        y = y.clone()
        y[..., 0] += 0.01
        return y, kv2
    monkeypatch.setattr(LatentLM, "_block", moved)
    rows, s = _run(lm, t, ex)
    c = s["checks"]
    assert c["1"]["verdict"] == "BUG" and c["1"]["first_answer_logits"]["over_tol"] == BS
    assert c["1"]["first_answer_logits"]["max"] > 1e-4
    assert any(b.startswith("check (1):") for b in s["bug"])
    assert c["3"]["strict"]["diverging"] == 0 and c["4"]["verdict"] == "within 0.0001 at each"
    assert any("check (1), strict fp32" in f and ": BUG;" in f for f in s["flags"])


def test_shape_check_compares_each_example_in_its_own_batch(setup, monkeypatch):
    """Check (3)'s plumbing: a band decode step that favours one token only when B > 1 parts the B = 1 and B = batch
    streams at the first decode step, with a large margin under the batch only; the extra in a batch of 2 diverges
    too; the first-answer-token logits are untouched (the prefill is not moved), so check (1) still passes."""
    t, ex, lm = setup
    tok = t.tok_to_id["bob"]
    orig = LatentLM._band_decode

    def favour(m, i, o):
        o = o.clone()
        o[:, -1, tok] += 100.0
        return o

    def bump(self, state, tk, pos):           # the head of a decode pass at B > 1, ahead of the recorder's hook
        if tk.shape[0] == 1:
            return orig(self, state, tk, pos)
        h = self.lm_head.register_forward_hook(favour, prepend=True)
        try:
            return orig(self, state, tk, pos)
        finally:
            h.remove()
    monkeypatch.setattr(LatentLM, "_band_decode", bump)
    rows, s = _run(lm, t, ex)
    g = s["checks"]["3"]["strict"]
    assert g["diverging"] > 0 and s["checks"]["3"]["verdict"] == f"{g['diverging']} diverging at strict fp32"
    for d in g["divergences"]:
        assert d["div_pos"] == 1 and d["margin_batch"] > 50 > d["margin_b1"] >= 0
        assert int(d["top2_batch"].split("|")[0]) == tok and int(d["top2_b1"].split("|")[0]) != tok
    assert g["extra"]["9"]["B_batch"] == 2 and g["extra"]["9"]["diverges"] == ("9" in map(str, g["idx_diverging"]))
    assert g["first_answer_logits"]["max"] <= 1e-5                    # step 0 unmoved: a stream effect only
    assert s["checks"]["1"]["verdict"] == "pass" and s["checks"]["1"]["diverging"] > 0
    assert any(b.startswith("check (3):") for b in s["bug"])


def test_swap_shows_a_cross_row_leak_at_both_numerics(setup):
    t, ex, lm = setup
    h = lm.base_causallm.transformer.ln_f.register_forward_hook(lambda m, i, o: o + 0.5 * o.mean(0, keepdim=True))
    try:
        rows, s = _run(lm, t, ex)
    finally:
        h.remove()
    c4 = s["checks"]["4"]
    for k in diag.BAND_NUMERICS:
        assert c4[k]["last_swap"]["over_tol"] == BS and c4[k]["last_rep"]["max"] == 0.0
    assert c4["verdict"] == "over 0.0001" and sum(b.startswith("check (4)") for b in s["bug"]) >= 2


def test_band_logits_refusals(setup):
    t, ex, lm = setup
    with pytest.raises(ValueError, match="latent-feedback"):
        _run(lm, t, ex, arm="cot", n_latent=0)
    with pytest.raises(ValueError, match="numerics"):
        _run(lm, t, ex, numerics={"strict": _plain})
    with pytest.raises(ValueError, match=r"extra idx \[99\]"):
        _run(lm, t, ex, extra_idx=(99,))
    with pytest.raises(ValueError, match="full-depth"):
        _run(diag.band_model(lm), t, ex)
    @contextlib.contextmanager
    def loose():                                                              # a "strict" set that is not
        torch.set_float32_matmul_precision("high")
        yield {}
    with pytest.raises(RuntimeError, match="not strict fp32"):
        _run(lm, t, ex, numerics={"strict": loose, "registered": _plain})
    sets = _numerics()

    @contextlib.contextmanager
    def autocast_strict():
        with sets["strict"]() as st, torch.autocast("cpu", dtype=torch.bfloat16):
            yield st
    with pytest.raises(RuntimeError, match="not strict fp32"):
        _run(lm, t, ex, numerics={"strict": autocast_strict, "registered": sets["registered"]})
    lm.base_causallm.to(torch.float64)
    try:
        with pytest.raises(RuntimeError, match="weights torch.float64"):
            _run(lm, t, ex)
    finally:
        lm.base_causallm.to(torch.float32)

    assert not lm.training


def test_numerics_that_move_during_the_checks_are_refused(setup, monkeypatch):
    t, ex, lm = setup
    orig, calls = diag._flags_now, []

    def drifting(device):                     # the second read (the end of the strict set) differs from the first
        calls.append(1)
        return {**orig(device), "float32_matmul_precision": "medium"} if len(calls) == 2 else orig(device)
    monkeypatch.setattr(diag, "_flags_now", drifting)
    with pytest.raises(RuntimeError, match="changed during the checks"):
        _run(lm, t, ex)
    assert not lm.training and "generate" not in lm.__dict__ and not lm.lm_head._forward_hooks


def test_a_recorder_misread_is_an_instrument_bug(setup, monkeypatch):
    t, ex, lm = setup
    orig = diag.PathRecorder._end

    def misread(self, out):
        orig(self, out)
        self.calls[-1]["rows"][0]["step0_consistent"] = False
    monkeypatch.setattr(diag.PathRecorder, "_end", misread)
    rows, s = _run(lm, t, ex)
    assert s["checks"]["1"]["step0_inconsistent"] == [0] and any("(instrument)" in b for b in s["bug"])


# --------------------------------------------------------------------------------------- run_diag.py band_logits
def _band_cfg(cfg, **kw):
    return cfg("prosqa_coconut", "coconut", precision="bf16", **kw)


def test_run_diag_band_logits_writes_the_record(diagenv):
    mod, cfg, ck, tmp = diagenv
    out = tmp / "b1"
    s = mod.main(["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--out", str(out),
                  "--extra-idx", "5,7", "--set", "seed=0", "--set", 'name="prosqa_coconut_s0"', "--commit", "c0ffee"])
    assert json.loads((out / "summary.json").read_text()) == json.loads(json.dumps(s))
    assert (s["mode"], s["critic_row"], s["critic_row_in_log"], s["registered_read"], s["split"], s["commit"]) == (
        "band_logits", "2026-10-20T12:40Z", True, None, "valid", "c0ffee")
    assert s["answers"].startswith("A6.2 rule 3") and s["run"] == "prosqa_coconut_s0" and s["seed"] == 0
    assert s["checkpoint_sha256"] == _sha(ck) and s["split_sha256"] == _sha(tmp / "toy_valid.json")
    assert s["config"].endswith("prosqa_coconut.json") and s["precision"] == "bf16"
    assert (s["band"], s["n_layer"], s["batch_size"], s["n"], s["n_latent"]) == ([0, 4], 4, 3, 8, 2)
    assert s["batch"]["idx"] == [0, 1, 2] and s["extra_idx"] == [5, 7]
    st, rg = s["numerics"]["strict"], s["numerics"]["registered"]
    assert (st["tf32_requested"], st["cuda.matmul.allow_tf32"], st["cudnn.allow_tf32"], st["float32_matmul_precision"],
            st["autocast"], st["registered_flags"]) == (False, False, False, "highest", None, "fp32")
    assert st["torch_at_start"]["autocast_enabled"] is False and st["torch_at_end"] == st["torch_at_start"]
    assert (rg["tf32_requested"], rg["cuda.matmul.allow_tf32"], rg["float32_matmul_precision"], rg["autocast"],
            rg["registered_flags"]) == (True, True, "high", "bf16", "bf16")
    assert rg["torch_at_start"]["autocast_enabled"] is True and rg["torch_at_start"]["autocast_dtype"] == "bfloat16"
    assert s["numerics_per_check"]["1"] == "strict" and s["numerics_per_check"]["2"] == "registered"
    assert "may not set precision" in s["numerics_note"]
    assert s["verdicts"]["1"] == "pass" and s["bug"] is False and s["results"]["bug"] == []
    assert {"torch", "cuda", "cudnn", "transformers", "device", "gpu_type"} <= set(s) and s["device"] == "cpu"
    for k in "1234":
        assert s["results"]["checks"][k]["rule"].startswith(("A6.2", "A14.1", "printed"))
    assert set(s["results"]["checks"]["3"]["strict"]["extra"]) == {"5", "7"}
    assert s["files"] == {"examples.csv": _sha(out / "examples.csv")} == {"examples.csv": s["examples_csv_sha256"]}
    rows = list(csv.DictReader(open(out / "examples.csv")))
    assert len(rows) == 2 * (3 + 5 + 3) and {r["check"] for r in rows} == set("1234")
    assert {"idx", "numerics", "first_maxabs", "diverges", "margin_b1", "last_swap", "last_rep"} <= set(rows[0]) | set(
        rows[-1]) | {k for r in rows for k in r}
    text = (out / "summary.json").read_text()
    for w in ("inert", "mechanism", "reproduces", "K2 would fire"):          # A11.3: none outside CHANGELOG
        assert w not in text
    with pytest.raises(SystemExit, match="write-once"):
        mod.main(["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--out", str(out)])


def test_run_diag_band_logits_refusals(diagenv, monkeypatch):
    mod, cfg, ck, tmp = diagenv
    c = _band_cfg(cfg)
    base = ["--checkpoint", ck, "--mode", "band_logits", "--out", str(tmp / "x")]
    for extra, msg in ((["--set", "latent_band=[0,4]"], "latent_band null"), (["--tf32", "off"], "--tf32 both"),
                       (["--tf32", "on"], "--tf32 both"), (["--extra-idx", "8"], r"outside the split"),
                       (["--extra-idx", "a"], "comma list"), (["--set", 'precision="fp16"'], "registered flag"),
                       (["--split", "test"], "valid split only"), (["--n-boot", "10"], "--n-boot")):
        with pytest.raises(SystemExit, match=msg):
            mod.main(["--config", c, *base, *extra])
    with pytest.raises(SystemExit, match="latent-feedback"):
        mod.main(["--config", cfg("prosqa_cot", "cot"), *base])
    with pytest.raises(SystemExit, match="belongs to --mode band_logits"):
        mod.main(["--config", cfg("prosqa_cot", "cot"), "--checkpoint", ck, "--mode", "cot_batch", "--extra-idx", "1"])
    log = tmp / "critic_log.md"
    log.write_text("| Logged (UTC) | Ruling |\n|---|---|\n| 2026-10-20T12:57Z | another row |\n")
    monkeypatch.setattr(mod, "CRITIC_LOG", str(log))
    with pytest.raises(SystemExit, match="registered by critic entry 2026-10-20T12:40Z"):
        mod.main(["--config", c, *base])
    assert not (tmp / "x").exists()


def test_numerics_sets(diagenv):
    mod, cfg, ck, tmp = diagenv
    sets = mod.numerics_sets("bf16", "cpu")
    with sets["strict"]() as st:
        assert (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()) == (False, "highest")
        assert not torch.is_autocast_enabled("cpu") and st["autocast"] is None
    with sets["registered"]() as st:
        assert (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()) == (True, "high")
        assert torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.bfloat16
        assert st["autocast"] == "bf16"
    assert not torch.is_autocast_enabled("cpu")
    with mod.numerics_sets("fp32-tf32", "cpu")["registered"]() as st:
        assert st["autocast"] is None and not torch.is_autocast_enabled("cpu") and st["cudnn.allow_tf32"] is True
    with pytest.raises(RuntimeError, match="autocast on"):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with sets["strict"]():
                pass


# ------------------------------------------------------------------- float64, the strict set alone (kernel-order bound)
def test_float64_strict_set_alone(setup, monkeypatch):
    """dtype=float64 with the strict set alone: the recorder keeps float64, the HF path and band [0, L] agree to
    float64 noise, (2) and the registered columns are not run; a band-only departure of 1e-7 (fp32 kernel-order size)
    shows plainly at float64; refusals: float64 with both sets or without strict_only, an unknown dtype, float32
    weights under dtype=float64."""
    t, ex, lm = setup
    sets = _numerics()
    strict = {"strict": sets["strict"]}
    lm.base_causallm.to(torch.float64)
    try:
        recs = diag.prompt_records(t, ex, arm="coconut", n_latent=NL)
        g = diag.generate_rows(lm, t, recs[:BS], {e.idx: e for e in ex}, mode="coconut", device="cpu",
                               max_new_tokens=MNT)
        assert {g[0][k].dtype for k in ("first", "latents", "top2_values")} == {torch.float64}
        rows, s = _run(lm, t, ex, numerics=strict, dtype=torch.float64, strict_only=True)
        c = s["checks"]
        assert s["dtype"] == "float64" and s["sets"] == ["strict"] and set(s["numerics"]) == {"strict"}
        assert c["1"]["verdict"] == "pass" and c["1"]["first_answer_logits"]["max"] <= 1e-12
        assert c["1"]["latents"]["max"] <= 1e-12 and c["1"]["first_argmax_differs"] == 0 and c["1"]["diverging"] == 0
        assert c["2"] == {"numerics": "registered", "line": None, "verdict": "not run (strict set alone)"}
        assert "registered" not in c["3"] and "registered" not in c["4"] and c["4"]["numerics"] == "strict"
        assert c["3"]["verdict"] == "0 diverging at strict float64" and c["4"]["verdict"] == "within 0.0001 at each"
        assert any(f.startswith("check (1), strict float64:") for f in s["flags"]) and s["bug"] == []
        assert "check (2), registered numerics: not run (strict set alone)" in s["flags"]
        assert {r["numerics"] for r in rows} == {"strict"} and {r["check"] for r in rows} == {"1", "3", "4"}
        assert s["dtypes"] == {"weights": ["float64"], "activations": ["float64"]} and "1b" not in c
        rows, s = _run(lm, t, ex, numerics=strict, dtype=torch.float64, strict_only=True, paths_b1=True, tol=1e-8)
        c = s["checks"]
        b = c["1b"]                                      # the HF path at B = 1 against band [0, L] at B = 1
        assert (b["verdict"], b["B"], b["n"], b["of_batch"]["n"], b["line"]) == ("pass", 1, BS + 2, BS, "<= 1e-08")
        assert b["first_answer_logits"]["max"] <= 1e-12 and b["latents"]["max"] <= 1e-12
        assert b["first_argmax_differs"] == 0 and b["diverging"] == 0 and s["tol"] == 1e-8
        assert sum(r["check"] == "1b" for r in rows) == BS + 2 and c["1"]["line"] == "<= 1e-08"
        assert any(f.startswith("check (1b), strict float64:") for f in s["flags"])
        h = lm.base_causallm.transformer.h[0].mlp.register_forward_hook(lambda m, i, o: o.float())
        try:                                            # one module output at float32: refused, not read
            with pytest.raises(RuntimeError, match="not float64 throughout"):
                _run(lm, t, ex, numerics=strict, dtype=torch.float64, strict_only=True)
        finally:
            h.remove()
        for kw, msg in ((dict(numerics=sets), "strict set alone"), (dict(numerics=strict), "strict set alone"),
                        (dict(numerics=sets, strict_only=True), "numerics must map exactly")):
            with pytest.raises(ValueError, match=msg):
                _run(lm, t, ex, dtype=torch.float64, **kw)
        with pytest.raises(ValueError, match="dtype"):
            _run(lm, t, ex, numerics=strict, dtype=torch.float16, strict_only=True)
        orig = LatentLM._block

        def moved(self, i, x, mask, kv):
            y, kv2 = orig(self, i, x, mask, kv)
            y = y.clone()
            y[..., 0] += 1e-7
            return y, kv2
        with monkeypatch.context() as m:
            m.setattr(LatentLM, "_block", moved)
            _, s2 = _run(lm, t, ex, numerics=strict, dtype=torch.float64, strict_only=True)
        assert s2["checks"]["1"]["first_answer_logits"]["max"] > 1e-10
    finally:
        lm.base_causallm.to(torch.float32)
    with pytest.raises(RuntimeError, match="not strict float64"):
        _run(lm, t, ex, numerics=strict, dtype=torch.float64, strict_only=True)
    rows, s = _run(lm, t, ex, numerics=strict, strict_only=True)                 # fp32, the strict set alone
    assert s["dtype"] == "float32" and s["checks"]["1"]["verdict"] == "pass"
    assert any(f.startswith("check (1), strict fp32:") for f in s["flags"])
    assert g[0]["first"].dtype == torch.float64 and not lm.training


def test_run_diag_band_logits_float64_strict_only(diagenv):
    mod, cfg, ck, tmp = diagenv
    out = tmp / "b64"
    s = mod.main(["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--out", str(out),
                  "--extra-idx", "5,7", "--dtype", "float64", "--strict-only", "--commit", "c0ffee"])
    assert json.loads((out / "summary.json").read_text()) == json.loads(json.dumps(s))
    assert (s["dtype"], s["strict_only"], s["tf32_modes"], s["base"], s["precision"]) == (
        "float64", True, ["strict"], None, "bf16")
    assert s["verdicts"] == {"1": "pass", "1b": "pass", "2": "not run (strict set alone)",
                             "3": "0 diverging at strict float64", "4": "within 1e-08 at each"}
    assert s["results"]["checks"]["1"]["first_answer_logits"]["max"] <= 1e-12 and s["bug"] is False
    assert s["tol"] == 1e-8 and s["results"]["checks"]["1"]["line"] == "<= 1e-08" and s["verdicts"]["1b"] == "pass"
    assert s["dtypes"] == {"weights": ["float64"], "activations": ["float64"]}
    assert s["results"]["checks"]["1"]["rule"].startswith("read against 1e-08 (float64")
    assert s["results"]["checks"]["1b"]["rule"].startswith("beside (1)") and s["numerics_per_check"]["1b"] == "strict"
    st = s["numerics"]["strict"]
    assert st["dtype"] == "float64" and st["set"].startswith("strict float64") and set(s["numerics"]) == {"strict"}
    assert s["numerics_per_check"]["2"] == "not run (--strict-only)" and s["results"]["checks"]["2"]["rule"] is None
    rows = list(csv.DictReader(open(out / "examples.csv")))
    assert {r["numerics"] for r in rows} == {"strict"} and len(rows) == 3 + 5 + 5 + 3
    s32 = mod.main(["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--out",
                    str(tmp / "b32"), "--extra-idx", "5,7", "--strict-only"])
    assert s32["dtype"] == "float32" and s32["numerics"]["strict"]["set"].startswith("strict fp32")
    assert "dtype" not in s32["numerics"]["strict"] and s32["verdicts"]["3"] == "0 diverging at strict fp32"
    assert s32["tol"] == 1e-4 and s32["dtypes"] == {"weights": ["float32"], "activations": ["float32"]}
    assert s32["results"]["checks"]["1"]["rule"].startswith("A6.2") and "1b" in s32["verdicts"]


def test_run_diag_band_logits_float64_refusals(diagenv):
    mod, cfg, ck, tmp = diagenv
    c = _band_cfg(cfg)
    base = ["--config", c, "--checkpoint", ck, "--mode", "band_logits"]
    for extra, msg in ((["--dtype", "float64", "--out", str(tmp / "x")], "add --strict-only"),
                       (["--strict-only"], "give --out"), (["--dtype", "float64", "--strict-only"], "give --out"),
                       (["--base-config", str(tmp)], "give --out"), (["--tol", "1e-6"], "give --out"),
                       (["--tol", "0", "--out", str(tmp / "x")], "positive")):
        with pytest.raises(SystemExit, match=msg):
            mod.main([*base, *extra])
    for extra in (["--dtype", "float64"], ["--strict-only"], ["--base-config", str(tmp)], ["--tol", "1e-6"]):
        with pytest.raises(SystemExit, match="belong to --mode band_logits"):
            mod.main(["--config", cfg("prosqa_cot", "cot"), "--checkpoint", ck, "--mode", "cot_batch", *extra])
    assert not (tmp / "x").exists()


def _pinned_dir(tmp, files=("config.json", "vocab.json")):
    d = tmp / "base"
    d.mkdir(exist_ok=True)
    for k, f in enumerate(files):
        (d / f).write_text(f"{f} {k}")
    man = tmp / "manifest.json"
    man.write_text(json.dumps({"models": [{"id": "toy", "revision": "abc",
                                           "files_sha256": {f: _sha(d / f) for f in files}}]}))
    return d, man


def test_pinned_base_files(diagenv):
    mod, cfg, ck, tmp = diagenv
    d, man = _pinned_dir(tmp)
    got = mod.pinned_base_files("toy", str(d), str(man))
    assert got == {"files_sha256": {f: _sha(d / f) for f in ("config.json", "vocab.json")}, "revision": "abc"}
    (d / "extra.txt").write_text("not pinned")                       # a file the pin does not list is ignored
    assert mod.pinned_base_files("toy", str(d), str(man)) == got
    with pytest.raises(SystemExit, match="no files_sha256 pinned"):
        mod.pinned_base_files("other", str(d), str(man))
    (d / "config.json").write_text("changed")
    with pytest.raises(SystemExit, match="config.json sha256 .* is not the pinned"):
        mod.pinned_base_files("toy", str(d), str(man))
    (d / "config.json").unlink()
    with pytest.raises(SystemExit, match="has no pinned config.json"):
        mod.pinned_base_files("toy", str(d), str(man))


def test_run_diag_band_logits_base_config(diagenv, monkeypatch):
    """--base-config: the base is a random init (two different seeds here) whose every tensor the checkpoint replaces:
    the records agree exactly and name the pinned files; a checkpoint that misses a tensor, or a load that leaves one
    differing from the checkpoint, is refused before any check runs."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from latent import train as T
    mod, cfg, ck, tmp = diagenv
    sd = torch.load(ck)
    V = sd["transformer.wte.weight"].shape[0]
    d, man = _pinned_dir(tmp)
    monkeypatch.setattr(mod, "MANIFEST", str(man))

    def fake_base(seed):
        def make(path):
            assert path == str(d)
            torch.manual_seed(seed)                     # base vocab below the checkpoint's: resized, then loaded
            m = GPT2LMHeadModel(GPT2Config(n_layer=4, n_embd=32, n_head=2, vocab_size=V - 3, n_positions=256))
            return m, T.load_base("toy")[1]             # the fixture's tokenizer (diagenv's load_base)
        return make

    def cut(n, path):                                   # the checkpoint at vocabulary n
        torch.save({k: (v[:n] if k.endswith(("wte.weight", "lm_head.weight")) else v) for k, v in sd.items()}, path)
        return path
    assert len(T.load_base("toy")[1]) == V - 2
    full = cut(V - 2, tmp / "ckpt_full.pt")             # at the resized vocabulary (len(tokenizer))
    outs = []
    for seed in (1, 2):
        monkeypatch.setattr(mod, "base_from_config", fake_base(seed))
        outs.append(mod.main(["--config", _band_cfg(cfg), "--checkpoint", str(full), "--mode", "band_logits",
                              "--out", str(tmp / f"bc{seed}"), "--extra-idx", "5", "--strict-only",
                              "--base-config", str(d)]))
    a, b = outs
    assert a["base"]["checkpoint_covers"] == len(sd) and a["base"]["files_sha256"] == {
        f: _sha(d / f) for f in ("config.json", "vocab.json")} and a["base"]["revision"] == "abc"
    assert a["base"]["model_id"] == "toy" and a["base"]["dir"] == str(d)
    assert json.dumps(a["results"]) == json.dumps(b["results"]) and a["verdicts"]["1"] == "pass"
    partial = tmp / "ckpt_partial.pt"
    torch.save({k: v for k, v in torch.load(full).items() if k != "transformer.ln_f.bias"}, partial)
    with pytest.raises(SystemExit, match=r"missing \['transformer.ln_f.bias'\]"):
        mod.main(["--config", _band_cfg(cfg), "--checkpoint", str(partial), "--mode", "band_logits",
                  "--out", str(tmp / "bcp"), "--extra-idx", "5", "--strict-only", "--base-config", str(d)])
    small = cut(V - 3, tmp / "ckpt_small.pt")           # the base's own vocabulary: loaded before the resize, so the
    with pytest.raises(SystemExit, match="differing"):  # resized embedding is not the checkpoint's
        mod.main(["--config", _band_cfg(cfg), "--checkpoint", str(small), "--mode", "band_logits",
                  "--out", str(tmp / "bcd"), "--extra-idx", "5", "--strict-only", "--base-config", str(d)])
    assert not (tmp / "bcp").exists() and not (tmp / "bcd").exists()


def test_run_diag_band_logits_critic_row_attention_and_peak_memory(diagenv, monkeypatch):
    """Critic entry 2026-10-20T21:33Z (2): the float64 record names the row that registers its job (--critic-row; the
    mode's row kept in mode_critic_row), the attention implementation the loader resolved and the peak memory; a
    --critic-row that is not a row, or given to another mode, is refused."""
    mod, cfg, ck, tmp = diagenv
    out = tmp / "r64"
    s = mod.main(["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--out", str(out),
                  "--extra-idx", "5,7", "--dtype", "float64", "--strict-only", "--tol", "1e-8",
                  "--critic-row", "2026-10-20T21:33Z", "--commit", "c0ffee"])
    assert json.loads((out / "summary.json").read_text()) == json.loads(json.dumps(s))
    assert (s["critic_row"], s["mode_critic_row"], s["critic_row_in_log"]) == (
        "2026-10-20T21:33Z", "2026-10-20T12:40Z", True)
    assert s["attn_implementation"] in ("eager", "sdpa") and "sdpa_backends_enabled" not in s     # cpu
    pm = s["peak_memory"]
    assert pm["device_max_allocated_bytes"] is None and pm["device_max_reserved_bytes"] is None
    assert pm["host_max_rss_bytes"] > 0
    assert s["dtypes"] == {"weights": ["float64"], "activations": ["float64"]} and s["tf32_modes"] == ["strict"]
    st = s["numerics"]["strict"]
    for k in ("torch_at_start", "torch_at_end"):
        assert (st[k]["cuda.matmul.allow_tf32"], st[k]["cudnn.allow_tf32"], st[k]["float32_matmul_precision"],
                st[k]["autocast_enabled"]) == (False, False, "highest", False)
    rows = list(csv.DictReader(open(out / "examples.csv")))       # 3 + 5 + 5 + 3: (1), (1b), (3), (4)
    assert [sum(r["check"] == c for r in rows) for c in ("1", "1b", "3", "4")] == [3, 5, 5, 3]
    s0 = mod.main(["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--out",
                   str(tmp / "r32"), "--extra-idx", "5,7"])
    assert (s0["critic_row"], s0["mode_critic_row"]) == ("2026-10-20T12:40Z", "2026-10-20T12:40Z")
    with pytest.raises(SystemExit, match="--critic-row '2026-10-20T21:59Z' is not a row"):
        mod.main(["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--out", str(tmp / "x"),
                  "--critic-row", "2026-10-20T21:59Z"])
    with pytest.raises(SystemExit, match="--critic-row belong to --mode band_logits"):
        mod.main(["--config", cfg("prosqa_cot", "cot"), "--checkpoint", ck, "--mode", "cot_batch",
                  "--critic-row", "2026-10-20T21:33Z"])
    assert not (tmp / "x").exists()


def test_peak_memory_and_attention_record_on_cuda(diagenv, monkeypatch):
    """On cuda the record reads torch.cuda's peak counters and the SDPA backend switches (stubbed here: CPU only)."""
    mod, *_ = diagenv
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 23 * 2 ** 30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 25 * 2 ** 30)
    pm = mod.peak_memory("cuda")
    assert (pm["device_max_allocated_bytes"], pm["device_max_reserved_bytes"]) == (23 * 2 ** 30, 25 * 2 ** 30)

    class M:
        class config:
            _attn_implementation = "sdpa"
    r = mod.attention_record(M(), "cuda")
    assert r["attn_implementation"] == "sdpa" and {"flash", "mem_efficient", "math"} <= set(r["sdpa_backends_enabled"])
    assert mod.attention_record(M(), "cpu") == {"attn_implementation": "sdpa"}


# ------------------------------------------------------------------- --checks, the partial files, memory release
# The float64 entry of diag-band-prosqa-hard-main (attempt 2ad4) ran out of memory in check (4) and left no record;
# critic entries 2026-10-20T22:36Z: --checks 1,1b,3 for its second admission, a partial file is never a record.
PRE_CHECKS = "rev387"         # the commit before --checks: run_diag.py and latent/diag.py as the records were made
GENERATED = ("utc", "seconds", "peak_memory")                     # differ run to run
NEW = ("complete", "checks_requested", "checks_run", "checks_not_run", "partial_files")      # added by --checks
REASON = "not run: CUDA out of memory in the same-shape companion swap at float64 (test)"


def _from_git(name, rel, repo):
    import subprocess
    import types
    r = subprocess.run(["git", "-C", repo, "show", f"{PRE_CHECKS}:{rel}"], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"{PRE_CHECKS} not in this clone ({r.stderr.strip()})")
    m = types.ModuleType(name)
    m.__file__ = f"{repo}/{rel}"
    exec(compile(r.stdout, m.__file__, "exec"), m.__dict__)
    return m


def _strip(s):
    s = json.loads(json.dumps(s))
    for k in GENERATED + NEW:
        s.pop(k, None)
    for st in s["numerics"].values():
        st.pop("seconds")
    return s


@pytest.mark.parametrize("extra", [[], ["--strict-only"], ["--dtype", "float64", "--strict-only"],
                                   ["--dtype", "float64", "--strict-only", "--tol", "1e-8",
                                    "--critic-row", "2026-10-20T21:33Z"]])
def test_default_record_equals_the_record_before_checks(diagenv, monkeypatch, extra):
    """Without --checks the record is the one run_diag.py / latent.diag wrote at rev23 (both loaded from git), field
    for field and examples.csv byte for byte, but for the generated fields (utc, seconds, peak memory, each set's
    seconds) and the fields --checks adds (complete, checks_requested, checks_run, checks_not_run, partial_files)."""
    import latent
    mod, cfg, ck, tmp = diagenv
    args = ["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--extra-idx", "5,7",
            "--commit", "c0ffee", *extra]
    new = mod.main([*args, "--out", str(tmp / "new")])
    old_diag = _from_git("latent._diag_pre_checks", "common/latent/diag.py", mod.REPO)
    old_mod = _from_git("run_diag_pre_checks", "code/run_diag.py", mod.REPO)
    old_mod.CRITIC_LOG = mod.CRITIC_LOG
    with monkeypatch.context() as m:
        m.setattr(latent, "diag", old_diag)
        old = old_mod.main([*args, "--out", str(tmp / "old")])
    assert not set(NEW) & set(old) and _strip(new) == _strip(old)
    assert (tmp / "new" / "examples.csv").read_bytes() == (tmp / "old" / "examples.csv").read_bytes()
    assert new["complete"] is True and new["checks_not_run"] == {} and new["checks_run"] == new["checks_requested"]
    assert new["checks_requested"] == (["1", "1b", "3", "4"] if extra else ["1", "2", "3", "4"])
    assert set(new["partial_files"]) == {"partial.json", "partial_examples.csv"}
    part = json.loads((tmp / "new" / "partial.json").read_text())       # the last partial: every check, complete false
    assert part["complete"] is False and part["checks_run"] == new["checks_run"]
    assert part["results"]["checks"] == new["results"]["checks"] and part["verdicts"] == new["verdicts"]
    assert (tmp / "new" / "partial_examples.csv").read_bytes() == (tmp / "new" / "examples.csv").read_bytes()


def test_run_diag_checks_1_1b_3_at_float64(diagenv):
    """--checks 1,1b,3 (the second admission's form): (1), (1b), (3) with the numbers of the full run, (4) recorded
    "not run (--checks)" with no line and no flag line, its reason in checks_not_run, rows 3 + 5 + 5; complete."""
    mod, cfg, ck, tmp = diagenv
    args = ["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--extra-idx", "5,7",
            "--dtype", "float64", "--strict-only", "--tol", "1e-8", "--commit", "c0ffee"]
    full = mod.main([*args, "--out", str(tmp / "full")])
    s = mod.main([*args, "--out", str(tmp / "sub"), "--checks", "1,1b,3", "--not-run-reason", REASON])
    assert json.loads((tmp / "sub" / "summary.json").read_text()) == json.loads(json.dumps(s))
    assert (s["complete"], s["checks_requested"], s["checks_run"]) == (True, ["1", "1b", "3"], ["1", "1b", "3"])
    assert s["checks_not_run"] == {"4": REASON}
    assert s["verdicts"] == {"1": "pass", "1b": "pass", "2": "not run (strict set alone)",
                             "3": "0 diverging at strict float64", "4": "not run (--checks)"}
    c = s["results"]["checks"]
    assert c["4"] == {"numerics": "strict", "line": None, "verdict": "not run (--checks)", "rule": None}
    for k in ("1", "1b", "3"):
        assert json.dumps(c[k]) == json.dumps(full["results"]["checks"][k])
    assert s["flags"] == [f for f in full["flags"] if not f.startswith("check (4)")] and s["bug"] is False
    assert s["numerics_per_check"]["4"] == "not run (--checks)" and s["numerics_per_check"]["3"] == "strict (the line)"
    rows = list(csv.DictReader(open(tmp / "sub" / "examples.csv")))
    assert [sum(r["check"] == k for r in rows) for k in ("1", "1b", "3", "4")] == [3, 5, 5, 0]
    full_rows = list(csv.DictReader(open(tmp / "full" / "examples.csv")))
    assert [{k: v for k, v in r.items() if v != ""} for r in rows] == [
        {k: v for k, v in r.items() if v != ""} for r in full_rows if r["check"] != "4"]


def test_a_failure_in_check_4_leaves_the_partial_files_not_a_record(diagenv, monkeypatch):
    """An out-of-memory in check (4) (as in attempt 2ad4): partial.json and partial_examples.csv carry (1), (1b), (3)
    with `complete` false; no summary.json or examples.csv is written; read_screen's gate refuses the partial form."""
    from latent.tests import test_screen_tools as sct
    mod, cfg, ck, tmp = diagenv
    args = ["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits", "--extra-idx", "5,7",
            "--dtype", "float64", "--strict-only", "--tol", "1e-8", "--commit", "c0ffee"]
    full = mod.main([*args, "--out", str(tmp / "full")])

    def oom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 10.32 GiB")
    monkeypatch.setattr(diag, "swap_check", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        mod.main([*args, "--out", str(tmp / "oom")])
    assert sorted(p.name for p in (tmp / "oom").iterdir()) == ["partial.json", "partial_examples.csv"]
    p = json.loads((tmp / "oom" / "partial.json").read_text())
    assert (p["complete"], p["checks_requested"], p["checks_run"]) == (False, ["1", "1b", "3", "4"], ["1", "1b", "3"])
    assert "4" not in p["results"]["checks"] and "4" not in p["verdicts"] and p["partial_files"] is None
    for k in ("1", "1b", "3"):
        assert json.dumps(p["results"]["checks"][k]) == json.dumps(full["results"]["checks"][k])
    assert p["files"] == {"partial_examples.csv": _sha(tmp / "oom" / "partial_examples.csv")}
    rs = sct._mod("read_screen")
    assert rs._diag_complete(p) is False and rs._diag_complete(full) is True
    assert rs._diag_complete({**p, "checks_requested": ["1", "1b", "3"]}) is True     # all it asked for: readable


def test_run_diag_checks_refusals(diagenv):
    mod, cfg, ck, tmp = diagenv
    base = ["--config", _band_cfg(cfg), "--checkpoint", ck, "--mode", "band_logits"]
    out = ["--out", str(tmp / "x")]
    for extra, msg in ((["--checks", "1,2,3", "--strict-only", *out], "registered set"),
                       (["--checks", "1,1b,3", *out], r"\(1b\) runs with paths_b1"),
                       (["--checks", "1,5", *out], "a non-empty list"), (["--checks", ",", *out], "a non-empty list"),
                       (["--checks", "1,3"], "a --checks subset are off the registered check: give --out"),
                       (["--not-run-reason", "memory", *out], "give a subset and a reason"),
                       (["--checks", "1,2,3,4", "--not-run-reason", "memory", *out], "give a subset and a reason"),
                       (["--checks", "1,3", "--not-run-reason", " ", *out], "give a subset and a reason")):
        with pytest.raises(SystemExit, match=msg):
            mod.main([*base, *extra])
    for extra in (["--checks", "1"], ["--not-run-reason", "x"]):
        with pytest.raises(SystemExit, match="belong to --mode band_logits"):
            mod.main(["--config", cfg("prosqa_cot", "cot"), "--checkpoint", ck, "--mode", "cot_batch", *extra])
    assert not (tmp / "x").exists()
    s = mod.main([*base, "--checks", "4,3,2,1", "--extra-idx", "5,7", "--out", str(tmp / "all")])     # the full set in any order: the default
    assert s["checks_requested"] == ["1", "2", "3", "4"] and s["checks_not_run"] == {}


def test_band_logits_checks_partials_and_release(setup, monkeypatch):
    """band_logits(checks=..., on_check=...): the partial summaries after each check (a check run under both sets is
    finished after the second), the release between checks (empty_cache on cuda only), (1b) without (3) from band
    [0, L]'s own B = 1 rows with (3)'s numbers, and band_checks' refusals."""
    t, ex, lm = setup
    seen = []
    rows, s = _run(lm, t, ex, on_check=lambda r, p: seen.append((len(r), p["checks_run"], p["complete"])))
    assert [x[1] for x in seen] == [["1"], ["1"], ["1"], ["1", "2"], ["1", "2", "3"], ["1", "2", "3", "4"]]
    assert [x[0] for x in seen][-1] == len(rows) and not any(x[2] for x in seen) and s["complete"] is True
    strict = {"strict": _numerics()["strict"]}
    seen.clear()
    _, s64 = _run(lm, t, ex, numerics=strict, strict_only=True, paths_b1=True,
                  on_check=lambda r, p: seen.append(p["checks_run"]))
    assert seen == [["1"], ["1", "3"], ["1", "1b", "3"], ["1", "1b", "3", "4"]]
    _, s1b = _run(lm, t, ex, numerics=strict, strict_only=True, paths_b1=True, checks=["1b"])
    assert json.dumps(s1b["checks"]["1b"]) == json.dumps(s64["checks"]["1b"])
    assert {k: s1b["checks"][k]["verdict"] for k in ("1", "3", "4")} == {k: diag.BAND_NOT_RUN for k in ("1", "3", "4")}
    assert s1b["flags"] == [f for f in s64["flags"] if f.startswith(("check (1b)", "check (2)"))]
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    diag._release("cpu")
    assert calls == []
    diag._release("cuda:0")
    assert calls == [1]
    assert diag.band_checks(None, strict_only=True, paths_b1=True) == ["1", "1b", "3", "4"]
    assert diag.band_checks(["3", "1"]) == ["1", "3"]
    for kw, msg in ((dict(checks=[]), "non-empty"), (dict(checks=["6"]), "non-empty"),
                    (dict(checks=["1b"]), "paths_b1"), (dict(checks=["2"], strict_only=True), "registered set")):
        with pytest.raises(ValueError, match=msg):
            diag.band_checks(kw.pop("checks"), **kw)
