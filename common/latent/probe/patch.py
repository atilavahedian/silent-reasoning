"""F1 counterfactual patching, F2 scrambled-latent drop, and the bank mechanism both F1 and F3 use.

Bank mechanism. A patched or edited run is `LatentLM.generate(..., mode="scrambled", scramble_bank=bank)` where
bank[i] is example i's OWN clean latents with only step t replaced (by B's step-t latent for F1, by h_t + delta for
F3), so exactly one fed-back vector differs from the clean run (`single_step_bank`, asserted tensor-wise in the
tests). downstream="frozen" (default, the operator's spec): later latents keep their clean values, so the change reaches
the answer only through attention to position t. downstream="recompute" (ours): latents t+1..L-1 are recomputed from
the patched prefix (the model's own feedback rule, last-layer hidden state of the previous position), so the change
also propagates through the latent chain; this uses `LatentLM.embedding` and `LatentLM.backbone` (public
properties of the current model.py) for the recomputation passes. Because the scrambled mode runs the prompt in one
pass where coconut mode runs one pass per latent (numerically equal up to float error), every comparison is made
against the NULL bank (own latents through the same scrambled path), never against the clean coconut predictions.
"""
from __future__ import annotations

import numpy as np
import torch

from latent.data import Collator, answer_extract, tokenize_examples
from latent.eval import build_prompts, evaluate, summarize
from latent.probe.stats import bootstrap_mean_ci, paired_diff_ci
from latent.probe.tasks import answer_option, make_pair


def single_step_bank(own: torch.Tensor, donor: torch.Tensor, t: int) -> torch.Tensor:
    """own [L, d] with row t replaced by donor[t] (a copy; own is not modified)."""
    if own.shape != donor.shape:
        raise ValueError(f"bank shapes differ: {tuple(own.shape)} vs {tuple(donor.shape)}")
    out = own.clone()
    out[t] = donor[t]
    return out


def delta_bank(own: torch.Tensor, t: int, delta) -> torch.Tensor:
    """own [L, d] with delta added to row t only (a copy)."""
    out = own.clone()
    out[t] = out[t] + torch.as_tensor(delta, dtype=out.dtype, device=out.device)
    return out


def _ids(tokenizer):
    return {k: tokenizer.convert_tokens_to_ids(t) for k, t in
            (("start_id", "<|start-latent|>"), ("end_id", "<|end-latent|>"), ("latent_id", "<|latent|>"))}


class BankRunner:
    """Greedy generation of a fixed set of prompts (question + markers + n_latent latents, as eval.build_prompts)
    with caller-supplied latent banks; scoring mirrors latent.eval.evaluate (decode prompt + generation with
    skip_special_tokens=True, answer_extract)."""

    def __init__(self, lm, tokenizer, examples, *, n_latent: int, batch_size: int = 32, device: str = "cpu",
                 max_new_tokens: int = 128):
        self.lm, self.tok, self.n_latent = lm, tokenizer, int(n_latent)
        self.batch_size, self.device, self.max_new_tokens = batch_size, device, max_new_tokens
        ids = _ids(tokenizer)
        self.latent_id = ids["latent_id"]
        recs = build_prompts(tokenize_examples(examples, tokenizer), tokenizer, arm="coconut", n_latent=n_latent)
        self.prompts = {r["idx"]: r for r in recs}
        self.collator = Collator(tokenizer.pad_token_id, self.latent_id)

    @torch.no_grad()
    def recompute(self, batch, bank: torch.Tensor, from_steps) -> torch.Tensor:
        """Recompute bank[i, k] for k > from_steps[i] by the coconut feedback rule from the (patched) prefix."""
        lm = self.lm
        ids = batch["input_ids"]
        lat = ids == self.latent_id
        bi, pi = lat.nonzero(as_tuple=True)
        rank = lat.long().cumsum(1) - 1
        first = int(lat.any(0).nonzero()[0].item())
        bank = bank.clone()
        fs = torch.as_tensor(from_steps, device=bank.device)
        for k in range(int(fs.min().item()) + 1, bank.shape[1]):
            emb = lm.embedding(ids).index_put((bi, pi), bank.to(lm.embedding.weight.dtype)[bi, rank[bi, pi]])
            h = lm.backbone(inputs_embeds=emb[:, :first + k], attention_mask=batch["attention_mask"][:, :first + k],
                            position_ids=batch["position_ids"][:, :first + k]).last_hidden_state
            upd = fs < k
            bank[upd, k] = h[upd, first + k - 1].to(bank.dtype)
        return bank

    @torch.no_grad()
    def run(self, jobs, downstream: str = "frozen") -> list[str]:
        """jobs: list of (idx, bank [n_latent, d], step t). Returns the extracted prediction per job."""
        if downstream not in ("frozen", "recompute"):
            raise ValueError(f"unknown downstream {downstream!r}")
        was = self.lm.training
        self.lm.eval()
        preds = []
        for s in range(0, len(jobs), self.batch_size):
            chunk = jobs[s:s + self.batch_size]
            recs = [self.prompts[j[0]] for j in chunk]
            b = {k: v.to(self.device) for k, v in self.collator(recs).items()}
            bank = torch.stack([j[1] for j in chunk]).to(self.device)
            if downstream == "recompute":
                bank = self.recompute(b, bank, [j[2] for j in chunk])
            g = self.lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"],
                                 max_new_tokens=self.max_new_tokens, mode="scrambled", scramble_bank=bank)
            for r, gen in zip(recs, g.tokens):
                preds.append(answer_extract(self.tok.decode(r["input_ids"] + gen, skip_special_tokens=True)))
        self.lm.train(was)
        return preds


def latents_of(lm, tokenizer, examples, *, n_latent, batch_size, device, max_new_tokens):
    """Clean fed-back latents ([n_latent, d] cpu float) and results per example, via latent.eval.evaluate."""
    res = evaluate(lm, tokenizer, examples, arm="coconut", n_latent=n_latent, batch_size=batch_size, device=device,
                   max_new_tokens=max_new_tokens, return_latents=True)
    return [r.latents.float() for r in res], res


def _rate(x, seed):
    m, lo, hi = bootstrap_mean_ci(x, seed=seed)
    return {"rate": m, "ci95": [lo, hi]}


def run_patching(lm, tokenizer, examples, *, task: str, rule: str, n_latent: int, own_latents=None,
                 batch_size: int = 32, device: str = "cpu", max_new_tokens: int = 128, downstream: str = "frozen",
                 seed: int = 0, include_all_steps: bool = True) -> dict:
    """F1: for every example A with a partner B under `rule` (tasks.make_pair), replace A's latent at step t by B's
    and classify the answer as follows / stays / breaks; per-step rates with bootstrap CIs over pairs, and the paired
    indicators (follows under the patch vs follows under the null bank) that the load-bearing rule reads."""
    pairs = [(i, make_pair(ex, rule, seed=seed)) for i, ex in enumerate(examples)]
    pairs = [(i, b) for i, b in pairs if b is not None]
    out = {"rule": rule, "downstream": downstream, "n_records": len(examples), "n_pairs": len(pairs), "per_step": []}
    if not pairs:
        out["skipped_reason"] = "no record has a partner under this rule"
        return out
    A = [examples[i] for i, _ in pairs]
    Bx = [b for _, b in pairs]
    if own_latents is None:
        own, _ = latents_of(lm, tokenizer, A, n_latent=n_latent, batch_size=batch_size, device=device,
                            max_new_tokens=max_new_tokens)
    else:
        own = [own_latents[i] for i, _ in pairs]
    lb, resb = latents_of(lm, tokenizer, Bx, n_latent=n_latent, batch_size=batch_size, device=device,
                          max_new_tokens=max_new_tokens)
    runner = BankRunner(lm, tokenizer, A, n_latent=n_latent, batch_size=batch_size, device=device,
                        max_new_tokens=max_new_tokens)
    a_opt = [_gold(ex, task) for ex in A]
    b_opt = [_gold(b, task) for b in Bx]
    base = [answer_option(p, task) for p in runner.run([(ex.idx, o, n_latent) for ex, o in zip(A, own)])]
    out["b_clean_option_correct"] = float(np.mean([answer_option(r.pred, task) == o for r, o in zip(resb, b_opt)]))
    out["null_bank_follow_rate"] = float(np.mean([p == o for p, o in zip(base, b_opt)]))
    out["null_bank_stay_rate"] = float(np.mean([p == o for p, o in zip(base, a_opt)]))
    steps = list(range(n_latent)) + (["all"] if include_all_steps and n_latent > 0 else [])
    for t in steps:
        jobs = [(ex.idx, (b.clone() if t == "all" else single_step_bank(o, b, t)), (-1 if t == "all" else t))
                for ex, o, b in zip(A, own, lb)]
        pred = [answer_option(p, task) for p in runner.run(jobs, downstream="frozen" if t == "all" else downstream)]
        fol = np.array([p == o for p, o in zip(pred, b_opt)], float)
        stay = np.array([p == o for p, o in zip(pred, a_opt)], float)
        brk = 1.0 - np.maximum(fol, stay)
        ctrl = np.array([p == o for p, o in zip(base, b_opt)], float)
        d, lo, hi = paired_diff_ci(fol, ctrl, seed=seed)
        out["per_step"].append({"step": t, "n_pairs": len(pred), "follows": _rate(fol, seed), "stays": _rate(stay, seed),
                                "breaks": _rate(brk, seed), "effect_vs_null": d, "effect_ci95": [lo, hi],
                                "moved": fol.tolist(), "control": ctrl.tolist()})
    return out


def _gold(ex, task):
    from latent.probe.tasks import gold_option
    m = ex.meta
    if task == "prosqa":
        return gold_option({"task": task, "symbols": m["idx_to_symbol"], "target": m["target"]})
    if task == "reach":
        return gold_option({"task": task, "reachable": m["reachable"]})
    return gold_option({"task": task, "answer": ex.answer})


def scrambled_drop(lm, tokenizer, examples, *, arm: str = "coconut", n_latent: int, batch_size: int = 32,
                   device: str = "cpu", max_new_tokens: int = 128, seed: int = 0) -> dict:
    """F2: a thin wrapper over latent.eval.evaluate(intervention=None | "scrambled") and summarize: accuracy of
    each, and the drop (clean minus scrambled) with a paired example-bootstrap CI."""
    kw = dict(arm=arm, n_latent=n_latent, batch_size=batch_size, device=device, max_new_tokens=max_new_tokens)
    clean = evaluate(lm, tokenizer, examples, **kw)
    scr = evaluate(lm, tokenizer, examples, intervention="scrambled", scramble_seed=seed, **kw)
    by = {r.idx: r for r in scr}
    a = np.array([float(r.correct) for r in clean])
    b = np.array([float(by[r.idx].correct) for r in clean])
    d, lo, hi = paired_diff_ci(a, b, seed=seed)
    sc, ss = summarize(clean, n_boot=2000, seed=seed), summarize(scr, n_boot=2000, seed=seed)
    return {"clean_accuracy": sc.get("accuracy"), "clean_accuracy_ci95": sc.get("accuracy_ci95"),
            "scrambled_accuracy": ss.get("accuracy"), "scrambled_accuracy_ci95": ss.get("accuracy_ci95"),
            "drop": d, "drop_ci95": [lo, hi], "n": len(clean)}


__all__ = ["single_step_bank", "delta_bank", "BankRunner", "latents_of", "run_patching", "scrambled_drop"]
