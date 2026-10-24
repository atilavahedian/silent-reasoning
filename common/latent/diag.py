"""Batch-isolation and fed-vector diagnostics on the eval path, for a real checkpoint on the valid split (registration
7.10 items 4 and 5; A11.3 (1); A12.3; critic entries 2026-10-19T15:35Z and 2026-10-19T16:59Z). Ours (no reference).

Everything runs through the code a read runs: prompts from latent.eval.build_prompts, batches from latent.data.Collator,
the prefill from LatentLM._run, greedy generation, scoring (answer_extract) and the scrambled bank (scramble_donors,
seed 0) from latent.eval.evaluate. Nothing here decodes, scores or deranges on its own. A `Recorder` only watches:
a forward pre-hook on the backbone sees the input embeddings it actually receives (the KV-cache length gives each
pass's first column), a forward hook on the LM head sees every pass's logits, and a wrapper on LatentLM.generate
brackets one batch. HF path only (a latent_band model does not call the backbone; refused).

* `isolation` (7.10 item 5; A12.3 (i); A11.3 (1) "batch isolation at B = 1 against 32"): per example, the logits at
  every attended prompt position ("logits"), at the last prompt position, whose argmax is the first emitted token
  ("last": the first-answer-token logits) and the fed-back latents ("latents"), at B = 1 against B = batch_size in two
  companion sets: "nat" (the read's batches, file order) and "perm" (batches of a seeded permutation). Two controls:
  "swap" (the nat batch with row i fixed and every other row replaced by an example from outside the batch, padded
  to the same width and latent column: identical shapes and row i, other companion content; row j changing row i
  shows here at either numerics) and "rep" (B = 1 computed twice; last-position logits and latents).
* `generation` (A12.3 (ii) and (iii)): evaluate at B = 1, at B = batch_size ("nat") and on the permuted order
  ("perm"); per example the emitted stream (the tokens, then eos when the example stopped on it), the first position
  where B = 1 and B = batch_size differ, the top-2 logit margin there under both, and correctness under each.
* `scramble` (7.10 item 4 with the real model; A11.3 (1)): evaluate(intervention="scrambled") watched by a Recorder;
  per example and latent rank, the L2 distance and cosine between the vector the scrambled pass fed and the example's
  own latent at that rank (from the same call's own-feedback pass, the bank's source). BUG: a zero distance, a fed
  vector that is not its donor's own latent bit for bit, a donor equal to the example, or a prompt column not fed
  exactly once. Also the first-answer-token logits under the own and the scrambled pass (changed when the max
  absolute difference exceeds tol) and their argmax.
* `band_logits` (A6.2 rule 3's strict-fp32 logits check; critic entry 2026-10-20T12:40Z, the registered diagnostic;
  A14.1): the HF path against band [0, L] (A6.2 rule 1's denominator, built over the same weights) on one checkpoint,
  each check under the numerics the caller enters for it. A `PathRecorder` follows LatentLM.generate on either path
  (step 0 = the logits generate takes its first token from; step s = decode pass s). (1) and (2): one batch (the
  split's first batch_size examples in file order) on both paths at the batch's shape: the first-answer-token logits,
  the fed latents and the greedy streams; (3): band [0, L] at B = 1 against B = batch_size in the read's own batch
  (the file-order chunk holding the example) on that batch plus extra idx; (4): the same-shape companion swap on band
  [0, L] at B = batch_size, with the batch computed twice beside it ("rep"). dtype=float64 with the strict set alone
  runs the same checks on a float64 model (kernel-order noise near 1e-12: a path difference shows, noise does not).
"""
from __future__ import annotations

import contextlib
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from latent import eval as E
from latent.data import Collator, answer_extract, answer_normalize, tokenize_examples
from latent.model import LatentLM

TOL = 1e-4                  # 7.10 item 5: example i's logits agree within 1e-4 (fp32) whatever its batch companions
SETS = ("nat", "perm")      # the two companion sets of A12.3 (i)
ISO_KEYS = tuple(f"{q}_{s}" for q in ("logits", "last", "latents") for s in (*SETS, "swap")) + ("rep_last",
                                                                                                   "rep_latents")


def _past_len(pkv) -> int:
    """Columns already in the KV cache (a Cache object or the legacy tuple), i.e. the first column of this pass."""
    if pkv is None:
        return 0
    if hasattr(pkv, "get_seq_length"):
        return int(pkv.get_seq_length())
    return int(pkv[0][0].shape[-2])


def permutation(n: int, seed: int) -> list[int]:
    """The "perm" companion set's order: numpy default_rng(seed).permutation(n) with position 0 moved to the front
    (tokenize_examples verifies the first record's piecewise tokenization, as the reference does; record 0 is the one
    every validation pass already verified)."""
    p = [int(k) for k in np.random.default_rng(seed).permutation(n)]
    return [0] + [k for k in p if k != 0] if n else p


def prompt_records(tokenizer, examples, *, arm: str, n_latent: int) -> list[dict]:
    """The generation prompts evaluate builds (tokenize_examples, then build_prompts at n_latent), in example order."""
    return E.build_prompts(tokenize_examples(examples, tokenizer), tokenizer, arm=arm, n_latent=n_latent)


def _maxabs(a, b):
    if a is None or b is None or a.numel() == 0:
        return None
    if torch.float64 in (a.dtype, b.dtype):        # a float64 run (band_logits dtype=float64) keeps its digits
        return float((a.double() - b.double()).abs().max())
    return float((a.float() - b.float()).abs().max())


def _keep(x):
    """A recorded tensor detached in its precision class: float64 stays float64 (band_logits dtype=float64), anything
    else is float32 as before."""
    return x.detach().double() if x.dtype == torch.float64 else x.detach().float()


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """First index where two emitted streams differ (eos included), None when equal; a strict prefix diverges at its
    length."""
    for t, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return t
    return None if len(a) == len(b) else min(len(a), len(b))


class Recorder:
    """Watches a LatentLM while evaluate runs (7.10 item 4: the vectors the backbone receives; item 6: every prompt
    column fed exactly once, the passes counted). One entry per LatentLM.generate call in `calls`: mode, padded width,
    prompt and decode pass counts, `covered_once`, and per row: the attended prompt ids, `fed` (the embedding received
    at each <|latent|> column, [n_latent, d] float cpu), `tokens`, `stream` (tokens + eos when stopped on it), the
    top-2 logits (values, ids) of every step (step 0 = the last prompt position; step s = decode pass s) and, with
    keep_first_logits, `first_logits` (the full step-0 vector)."""

    def __init__(self, lm, keep_first_logits: bool = False):
        if getattr(lm, "latent_band", None) is not None:
            raise ValueError("Recorder follows the HF path; a latent_band model does not call the backbone")
        self.lm, self.keep = lm, keep_first_logits
        self.calls: list[dict] = []
        self._cur = None
        self._hooks = []

    def __enter__(self):
        orig = self.lm.generate

        def generate(input_ids, attention_mask, position_ids, **kw):
            self._begin(input_ids, attention_mask, kw.get("mode", "coconut"), kw["max_new_tokens"])
            try:
                out = orig(input_ids, attention_mask, position_ids, **kw)
                self._end(out)
            finally:
                self._cur = None
            return out
        self.lm.generate = generate          # instance attribute: evaluate's lm.generate(...) goes through it
        self._hooks = [self.lm.backbone.register_forward_pre_hook(self._pre, with_kwargs=True),
                       self.lm.lm_head.register_forward_hook(self._head)]
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.lm.__dict__.pop("generate", None)
        return False

    def _begin(self, input_ids, attention_mask, mode, max_new_tokens):
        B, L = input_ids.shape
        cols = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        last = torch.where(attention_mask.bool(), cols, torch.full_like(cols, -1)).max(1).values   # as _generate
        self._cur = {"mode": mode, "B": B, "L": L, "max_new_tokens": int(max_new_tokens), "last": last,
                     "input_ids": input_ids.detach().cpu(), "attention_mask": attention_mask.detach().cpu(),
                     "fed": None, "covered": torch.zeros(L, dtype=torch.long), "prompt_passes": 0,
                     "decode_passes": 0, "r0": None, "first": None, "top2": {}}

    def _pre(self, module, args, kwargs):
        c = self._cur
        emb = kwargs.get("inputs_embeds")
        if c is None or emb is None:
            return None
        r0 = _past_len(kwargs.get("past_key_values"))
        c["r0"] = r0
        if r0 < c["L"]:
            hi = min(c["L"], r0 + emb.shape[1])
            if c["fed"] is None:
                c["fed"] = torch.full((c["B"], c["L"], emb.shape[-1]), float("nan"), dtype=emb.dtype,
                                      device=emb.device)
            c["fed"][:, r0:hi] = emb[:, :hi - r0].detach()
            c["covered"][r0:hi] += 1
            c["prompt_passes"] += 1
        else:
            c["decode_passes"] += 1
        return None

    @staticmethod
    def _put_top2(c, step, rows, lg):
        v, i = lg.topk(2, dim=-1)
        if step not in c["top2"]:
            c["top2"][step] = (torch.full((c["B"], 2), float("nan"), device=lg.device),
                               torch.full((c["B"], 2), -1, dtype=torch.long, device=lg.device))
        V, I = c["top2"][step]
        if rows is None:
            V[:], I[:] = v, i
        else:
            V[rows], I[rows] = v, i

    def _head(self, module, inputs, output):
        c = self._cur
        if c is None or c["r0"] is None:
            return None
        r0, n = c["r0"], output.shape[1]
        c["r0"] = None
        if r0 < c["L"]:          # a prompt pass: rows whose last attended column falls in it give step 0
            last = c["last"]
            rows = ((last >= r0) & (last < r0 + n)).nonzero().squeeze(1)
            if rows.numel():
                lg = output[rows, last[rows] - r0].detach().float()
                if self.keep:
                    if c["first"] is None:
                        c["first"] = torch.full((c["B"], output.shape[-1]), float("nan"), device=output.device)
                    c["first"][rows] = lg
                self._put_top2(c, 0, rows, lg)
        else:                     # decode pass s = r0 - L + 1 (the cache holds the prompt and s - 1 tokens)
            self._put_top2(c, r0 - c["L"] + 1, None, output[:, -1].detach().float())
        return None

    def _end(self, out):
        c = self._cur
        steps = sorted(c["top2"])
        tv = torch.stack([c["top2"][s][0] for s in steps], 1).cpu() if steps else torch.zeros(c["B"], 0, 2)
        ti = torch.stack([c["top2"][s][1] for s in steps], 1).cpu() if steps else torch.zeros(c["B"], 0, 2).long()
        lat = c["input_ids"] == self.lm.latent_id
        first = c["first"].cpu() if c["first"] is not None else None
        rows = []
        for b in range(c["B"]):
            toks = list(out.tokens[b])
            capped = bool(out.capped[b]) if out.capped is not None else False
            eos = len(toks) < c["max_new_tokens"] and not capped
            stream = toks + ([self.lm.eos_id] if eos else [])
            cols = lat[b].nonzero().squeeze(1)
            fed = (c["fed"][b, cols.to(c["fed"].device)].float().cpu() if c["fed"] is not None
                   else torch.zeros(0))
            rows.append({"prompt": c["input_ids"][b][c["attention_mask"][b].bool()].tolist(), "width": c["L"],
                         "tokens": toks, "stream": stream, "eos": eos, "hit_cap": len(toks) >= c["max_new_tokens"],
                         "fed": fed, "top2_values": tv[b, :len(stream)], "top2_ids": ti[b, :len(stream)],
                         "first_logits": first[b] if first is not None else None})
        self.calls.append({"mode": c["mode"], "width": c["L"], "prompt_passes": c["prompt_passes"],
                           "decode_passes": c["decode_passes"], "covered_once": bool((c["covered"] == 1).all()),
                           "rows": rows})

    def by_idx(self, records: list[dict], batch_size: int, mode: str | None = None) -> dict[int, dict]:
        """Rows keyed by example idx: evaluate's calls of `mode` follow `records` in chunks of batch_size; every row's
        attended prompt must equal its record's input_ids (else RuntimeError)."""
        calls = [c for c in self.calls if mode is None or c["mode"] == mode]
        chunks = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
        if len(calls) != len(chunks):
            raise RuntimeError(f"{len(calls)} generate calls in mode {mode!r} for {len(chunks)} batches")
        out = {}
        for call, chunk in zip(calls, chunks):
            if len(call["rows"]) != len(chunk):
                raise RuntimeError("a generate call's batch differs from evaluate's chunk")
            for row, rec in zip(call["rows"], chunk):
                if row["prompt"] != list(rec["input_ids"]):
                    raise RuntimeError(f"recorded prompt differs from record idx {rec['idx']} (order)")
                out[rec["idx"]] = {**row, "prompt_passes": call["prompt_passes"],
                                   "covered_once": call["covered_once"]}
        return out


def _margin(row, t):
    v = row["top2_values"]
    if t is None or t >= v.shape[0] or torch.isnan(v[t]).any():
        return None
    return float(v[t, 0] - v[t, 1])


def _top2(row, t):
    i = row["top2_ids"]
    return None if t is None or t >= i.shape[0] else "|".join(str(int(x)) for x in i[t])


# ------------------------------------------------------------------------------------------------ isolation
@torch.no_grad()
def _prefill(lm, batch, mode, device):
    """LatentLM._run on one collated batch, as generate's prefill (keep_cache=True): per row (logits at its attended
    prompt columns [n, V], fed-back latents [n_latent, d]). Attended columns are contiguous in both collations."""
    out = lm._run(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["position_ids"].to(device),
                  mode, None, keep_cache=True)
    am = batch["attention_mask"].bool()
    rows = []
    for b in range(am.shape[0]):
        cols = am[b].nonzero().squeeze(1)
        lo, hi = int(cols[0]), int(cols[-1]) + 1
        if hi - lo != cols.numel():
            raise ValueError("attended prompt columns are not contiguous")
        rows.append((out.logits[b, lo:hi], out.latents[b]))
    return rows


def _diff(d, name, a, b):
    d[f"logits_{name}"] = _maxabs(a[0], b[0])
    d[f"last_{name}"] = _maxabs(a[0][-1], b[0][-1])
    d[f"latents_{name}"] = _maxabs(a[1], b[1])


def _latent_col(input_ids, latent_id):
    lat = (input_ids == latent_id).any(0).nonzero()
    return int(lat[0]) if lat.numel() else None


def _pad_to(b, left, width, pad_id):
    """Prepend `left` and append up to `width` columns as the Collator pads (input pad, attention 0, position 0)."""
    B, L = b["input_ids"].shape
    right = width - L - left
    if left < 0 or right < 0:
        raise ValueError("a swap batch does not fit the batch's shape")

    def pad(x, v):
        return torch.cat([torch.full((B, left), v, dtype=x.dtype), x, torch.full((B, right), v, dtype=x.dtype)], 1)
    return {"input_ids": pad(b["input_ids"], pad_id), "attention_mask": pad(b["attention_mask"], 0),
            "position_ids": pad(b["position_ids"], 0)}


def _swap_batch(collate, chunk, r, pool, big, latent_id, pad_id):
    """Row r of `chunk` kept; every other row replaced by a record from `pool` (examples outside the batch, in the
    seeded order) that fits the batch's shape; padded to big's width and latent column, so row r's tokens, mask,
    positions and the tensor shapes are big's and only the companions' content differs. None when nothing fits."""
    width = big["input_ids"].shape[1]
    col = _latent_col(big["input_ids"], latent_id)

    def fits(rec):
        ids = rec["input_ids"]
        if col is None:
            return len(ids) <= width
        return latent_id in ids and ids.index(latent_id) <= col and len(ids) - ids.index(latent_id) + col <= width
    cand = [rec for rec in pool if fits(rec)]
    if not cand:
        return None
    new = [chunk[k] if k == r else cand[(k - (k > r)) % len(cand)] for k in range(len(chunk))]
    b = collate(new)
    sb = _pad_to(b, 0 if col is None else col - _latent_col(b["input_ids"], latent_id), width, pad_id)
    if not all(torch.equal(sb[k][r], big[k][r]) for k in ("input_ids", "attention_mask", "position_ids")):
        raise RuntimeError("the swap batch changed row i's layout")
    return sb


def isolation(lm, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str, perm: list[int]
              ) -> list[dict]:
    """7.10 item 5 on the read's path: per example (file order) the max absolute differences of the prompt logits,
    the last-position logits and the latents between B = 1 and B = batch_size ("nat", "perm"), the same-shape
    companion swap ("swap": nat batch against itself with the companions replaced; None for a batch of one or when no
    outside example fits) and the B = 1 repeat ("rep"); prompt length and padded widths."""
    was = lm.training
    lm.eval()
    mode = E.gen_mode(arm)
    records = prompt_records(tokenizer, examples, arm=arm, n_latent=n_latent)
    by = {r["idx"]: r for r in records}
    collate = Collator(tokenizer.pad_token_id, lm.latent_id)
    res = {r["idx"]: {"idx": r["idx"], "prompt_len": len(r["input_ids"])} for r in records}
    b1_first = {}
    orders = {"nat": [r["idx"] for r in records], "perm": [records[k]["idx"] for k in perm]}
    pool = []               # the swap's replacements per nat batch: every example outside it, in the seeded order
    for s in range(0, len(records), batch_size):
        inside = set(orders["nat"][s:s + batch_size])
        pool.append([by[i] for i in orders["perm"] if i not in inside])
    try:
        for name in SETS:
            order = orders[name]
            for s in range(0, len(order), batch_size):
                chunk = [by[i] for i in order[s:s + batch_size]]
                big = collate(chunk)
                big_rows = _prefill(lm, big, mode, device)
                for r, rec in enumerate(chunk):
                    i, d = rec["idx"], res[rec["idx"]]
                    one = _prefill(lm, collate([rec]), mode, device)[0]
                    d[f"width_{name}"] = int(big["input_ids"].shape[1])
                    _diff(d, name, one, big_rows[r])
                    if name == "nat":
                        b1_first[i] = (one[0][-1].float().cpu(), one[1].float().cpu())
                        sb = _swap_batch(collate, chunk, r, pool[s // batch_size], big, lm.latent_id,
                                         tokenizer.pad_token_id) if len(chunk) >= 2 else None
                        if sb is None:
                            d.update(logits_swap=None, last_swap=None, latents_swap=None)
                        else:
                            _diff(d, "swap", big_rows[r], _prefill(lm, sb, mode, device)[r])
                    else:
                        last1, lat1 = b1_first[i]
                        d["rep_last"] = _maxabs(one[0][-1].cpu(), last1)
                        d["rep_latents"] = _maxabs(one[1].cpu(), lat1)
                    del one
                del big_rows
    finally:
        lm.train(was)
    return [res[r["idx"]] for r in records]


def isolation_counts(rows: list[dict], tol: float = TOL) -> dict:
    """Per comparison: examples compared, over tol, nonzero, the max, and the idx over tol."""
    out = {}
    for k in ISO_KEYS:
        have = [(r["idx"], r[k]) for r in rows if r.get(k) is not None]
        out[k] = {"n": len(have), "over_tol": sum(v > tol for _, v in have), "nonzero": sum(v > 0 for _, v in have),
                  "max": max((v for _, v in have), default=None), "idx_over_tol": [i for i, v in have if v > tol]}
    return out


# ----------------------------------------------------------------------------------------------- generation
def generation(lm, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str,
               max_new_tokens: int, perm: list[int]) -> tuple[list[dict], dict]:
    """evaluate at B = 1, at B = batch_size in file order ("nat", the read's batching) and in the permuted order
    ("perm"), watched by a Recorder. Per example: emitted tokens, eos / cap, correctness under each (eval.py's
    answer extraction), and against B = 1 the first divergent stream position, the top-2 logit margin there and the
    top-2 token ids under both. Returns (rows in file order, {run: {idx: ExampleResult}})."""
    mode = E.gen_mode(arm)
    runs, results = {}, {}
    for name, exs, bs in (("b1", examples, 1), ("nat", examples, batch_size),
                          ("perm", [examples[k] for k in perm], batch_size)):
        recs = prompt_records(tokenizer, exs, arm=arm, n_latent=n_latent)
        with Recorder(lm) as rec:
            out = E.evaluate(lm, tokenizer, exs, arm=arm, n_latent=n_latent, batch_size=bs, device=device,
                             max_new_tokens=max_new_tokens)
        runs[name] = rec.by_idx(recs, bs, mode)
        results[name] = {r.idx: r for r in out}
    rows = []
    for ex in examples:
        i = ex.idx
        one = runs["b1"][i]
        d = {"idx": i, "prompt_len": len(one["prompt"]), "emitted_b1": len(one["tokens"]), "hit_cap_b1": one["hit_cap"],
             "correct_b1": results["b1"][i].correct}
        for name in SETS:
            row = runs[name][i]
            t = first_divergence(one["stream"], row["stream"])
            d.update({f"gen_width_{name}": row["width"], f"emitted_{name}": len(row["tokens"]),
                      f"hit_cap_{name}": row["hit_cap"], f"correct_{name}": results[name][i].correct,
                      f"diverges_{name}": t is not None, f"div_pos_{name}": t,
                      f"margin_b1_{name}": _margin(one, t), f"margin_{name}": _margin(row, t),
                      f"top2_b1_{name}": _top2(one, t), f"top2_{name}": _top2(row, t)})
        rows.append(d)
    return rows, results


def generation_counts(rows: list[dict], results: dict, n_boot: int = 0) -> dict:
    """Diverging generations per companion set with every divergence's position and margins; correctness under
    each run (accuracy, and its percentile-bootstrap CI when n_boot > 0)."""
    out = {}
    for name in SETS:
        div = [r for r in rows if r[f"diverges_{name}"]]
        out[name] = {"diverging": len(div), "n": len(rows), "idx": [r["idx"] for r in div],
                     "divergences": [{"idx": r["idx"], "pos": r[f"div_pos_{name}"],
                                      "margin_b1": r[f"margin_b1_{name}"], "margin_batch": r[f"margin_{name}"],
                                      "top2_b1": r[f"top2_b1_{name}"], "top2_batch": r[f"top2_{name}"],
                                      "width_batch": r[f"gen_width_{name}"], "prompt_len": r["prompt_len"],
                                      "correct_b1": r["correct_b1"], "correct_batch": r[f"correct_{name}"]}
                                     for r in div],
                     "correct_differs": sum(r["correct_b1"] != r[f"correct_{name}"] for r in rows),
                     "correct_differs_idx": [r["idx"] for r in rows if r["correct_b1"] != r[f"correct_{name}"]]}
    acc = {}
    for name, res in results.items():
        vals = list(res.values())
        s = E.summarize(vals, n_boot=n_boot) if n_boot > 0 else {"n": len(vals),
                                                                 "accuracy": float(np.mean([r.correct for r in vals]))}
        acc[name] = {"correct": int(sum(r.correct for r in vals)), "n": len(vals), "accuracy": s["accuracy"],
                     **({"accuracy_ci95": s["accuracy_ci95"]} if "accuracy_ci95" in s else {})}
    out["correctness"] = acc
    out["hit_cap"] = {name: sum(r[f"hit_cap_{name}"] for r in rows) for name in ("b1", *SETS)}
    return out


# ------------------------------------------------------------------------------------------------- scramble
def scramble(lm, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str,
             max_new_tokens: int, scramble_seed: int = 0, tol: float = TOL, results_out: dict | None = None
             ) -> tuple[list[dict], list[dict], dict]:
    """7.10 item 4 with the real model: evaluate(intervention=None) and evaluate(intervention="scrambled"), each
    watched by a Recorder. Returns (per-example rows, per (example, rank) rows, summary with "bug" reasons).
    results_out, when a dict, receives the two passes' ExampleResult lists as "own" and "scr" (the donor-answer
    scorer reads their pred / gold strings; nothing else changes)."""
    if arm not in E.LATENT_ARMS:
        raise ValueError(f"the scrambled pass needs a latent-feedback arm, not {arm!r}")
    records = prompt_records(tokenizer, examples, arm=arm, n_latent=n_latent)
    with Recorder(lm, keep_first_logits=True) as rec_own:
        own = E.evaluate(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size, device=device,
                         max_new_tokens=max_new_tokens, return_latents=True)
    with Recorder(lm, keep_first_logits=True) as rec:
        scr = E.evaluate(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size, device=device,
                         max_new_tokens=max_new_tokens, intervention="scrambled", return_latents=True,
                         scramble_seed=scramble_seed)
    if results_out is not None:
        results_out.update(own=own, scr=scr)
    own_rows = rec_own.by_idx(records, batch_size, "coconut")
    clean = rec.by_idx(records, batch_size, "coconut")       # the scrambled call's own pass: the bank's source
    fed = rec.by_idx(records, batch_size, "scrambled")
    donors = E.scramble_donors({i: clean[i]["fed"].shape[0] for i in clean}, seed=scramble_seed)   # evaluate's call
    own_res, scr_res = {r.idx: r for r in own}, {r.idx: r for r in scr}
    rows, ranks = [], []
    for ex in examples:
        i = ex.idx
        j = donors[i]
        a, o = fed[i]["fed"], clean[i]["fed"]
        l2 = (a.double() - o.double()).norm(dim=-1)
        cos = F.cosine_similarity(a.double(), o.double(), dim=-1)
        for k in range(a.shape[0]):
            ranks.append({"idx": i, "rank": k, "donor": j, "l2": float(l2[k]), "cos": float(cos[k]),
                          "own_norm": float(o[k].double().norm()), "fed_norm": float(a[k].double().norm())})
        fo, fs = clean[i]["first_logits"], fed[i]["first_logits"]
        t = first_divergence(clean[i]["stream"], fed[i]["stream"])
        k_cmp = len(clean[i]["stream"]) if t is None else t
        top1 = (clean[i]["top2_values"][:k_cmp, 0] - fed[i]["top2_values"][:k_cmp, 0]).abs()
        rows.append({"idx": i, "n_latent": int(a.shape[0]), "donor": j, "donor_is_self": j == i,
                     "fed_equals_donor": torch.equal(a, clean[j]["fed"]),
                     "fed_equals_collected": torch.equal(a, scr_res[i].latents.float()),
                     "fed_min_l2": float(l2.min()) if l2.numel() else None,
                     "fed_max_cos": float(cos.max()) if cos.numel() else None,
                     "first_maxabs_own_scr": _maxabs(fo, fs), "first_argmax_own": int(fo.argmax()),
                     "first_argmax_scr": int(fs.argmax()), "own_repeat_latents": _maxabs(own_rows[i]["fed"], o),
                     "own_repeat_first": _maxabs(own_rows[i]["first_logits"], fo),
                     "stream_div_own_scr": t, "top1_maxabs_own_scr": float(top1.max()) if top1.numel() else None,
                     "correct_own": own_res[i].correct, "correct_scr": scr_res[i].correct})
    l2_all = np.array([r["l2"] for r in ranks])
    cos_all = np.array([r["cos"] for r in ranks])
    changed = [r["idx"] for r in rows if r["first_maxabs_own_scr"] is not None and r["first_maxabs_own_scr"] > tol]
    passes = {m: sorted({c["prompt_passes"] for rc in (rec_own, rec) for c in rc.calls if c["mode"] == m})
              for m in ("coconut", "scrambled")}
    covered = all(c["covered_once"] for rc in (rec, rec_own) for c in rc.calls)
    bug = []
    zero = int((l2_all == 0).sum())
    if zero:
        bug.append(f"{zero} fed-vector distances are 0 (the scrambled pass fed an example its own latent)")
    selfish = sum(r["donor_is_self"] for r in rows)
    if selfish:
        bug.append(f"{selfish} examples are their own donor")
    for key, what in (("fed_equals_donor", "the donor's own latent"), ("fed_equals_collected", "evaluate's latents")):
        k = sum(not r[key] for r in rows)
        if k:
            bug.append(f"{k} examples' fed vectors are not {what} bit for bit")
    if not covered:
        bug.append("a prompt column was not fed exactly once in some generate call")
    if passes["coconut"] != [1 + n_latent] or passes["scrambled"] != [1]:
        bug.append(f"prompt passes per call {passes}, expected coconut [{1 + n_latent}] and scrambled [1]")
    summ = {"n": len(rows), "n_latent": n_latent, "scramble_seed": scramble_seed, "tol": tol,
            "fed_vectors": {"pairs": int(l2_all.size), "zero_distance": zero,
                            "min_l2": float(l2_all.min()) if l2_all.size else None,
                            "median_l2": float(np.median(l2_all)) if l2_all.size else None,
                            "max_cos": float(cos_all.max()) if cos_all.size else None,
                            "mean_cos": float(cos_all.mean()) if cos_all.size else None,
                            "donor_is_self": sum(r["donor_is_self"] for r in rows),
                            "fed_equals_donor": sum(r["fed_equals_donor"] for r in rows),
                            "fed_equals_collected": sum(r["fed_equals_collected"] for r in rows),
                            "prompt_passes": passes, "covered_once": covered,
                            "verdict": "BUG" if bug else "ok"},
            "first_answer_logits": {"changed": len(changed), "fraction": len(changed) / max(1, len(rows)),
                                    "idx_changed": changed,
                                    "max_abs_diff_min": min((r["first_maxabs_own_scr"] for r in rows), default=None),
                                    "max_abs_diff_max": max((r["first_maxabs_own_scr"] for r in rows), default=None),
                                    "argmax_changed": sum(r["first_argmax_own"] != r["first_argmax_scr"]
                                                          for r in rows)},
            "streams": {"differ": sum(r["stream_div_own_scr"] is not None for r in rows),
                        "correct_own": sum(r["correct_own"] for r in rows),
                        "correct_scr": sum(r["correct_scr"] for r in rows),
                        "correct_differs": sum(r["correct_own"] != r["correct_scr"] for r in rows)},
            "own_repeat": {"latents_max": max((r["own_repeat_latents"] or 0.0 for r in rows), default=None),
                           "first_logits_max": max((r["own_repeat_first"] or 0.0 for r in rows), default=None)},
            "bug": bug}
    return rows, ranks, summ


# ---------------------------------------------------------------------------------------------- the two modes
def _merge(*tables):
    out = {}
    for t in tables:
        for r in t:
            out.setdefault(r["idx"], {}).update(r)
    return list(out.values())


def cot_batch(lm, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str,
              max_new_tokens: int, perm_seed: int = 0, tol: float = TOL, n_boot: int = 0) -> tuple[list[dict], dict]:
    """A12.3 on the arm's generation path (one numerics mode): isolation, generation and correctness. Returns (rows
    per example, summary)."""
    perm = permutation(len(examples), perm_seed)
    iso = isolation(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size, device=device,
                    perm=perm)
    gen, results = generation(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size,
                              device=device, max_new_tokens=max_new_tokens, perm=perm)
    return _merge(iso, gen), {"isolation": isolation_counts(iso, tol), "generation": generation_counts(gen, results,
                                                                                                        n_boot)}


def latent_path(lm, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str,
                max_new_tokens: int, perm_seed: int = 0, scramble_seed: int = 0, tol: float = TOL, n_boot: int = 0
                ) -> tuple[list[dict], list[dict], dict]:
    """A11.3 (1) (one numerics mode): the scrambled pass's fed vectors and first-answer logits, then isolation and
    generation on the latent path. Returns (rows per example, rows per example and rank, summary; "bug" lists every
    failed fed-vector check)."""
    sc_rows, ranks, sc = scramble(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size,
                                  device=device, max_new_tokens=max_new_tokens, scramble_seed=scramble_seed, tol=tol)
    perm = permutation(len(examples), perm_seed)
    iso = isolation(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size, device=device,
                    perm=perm)
    gen, results = generation(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size,
                              device=device, max_new_tokens=max_new_tokens, perm=perm)
    summ = {"scramble": sc, "isolation": isolation_counts(iso, tol), "generation": generation_counts(gen, results,
                                                                                                     n_boot),
            "bug": list(sc["bug"])}
    return _merge(sc_rows, iso, gen), ranks, summ


def flags(results: dict, batch_size: int, tol: float = TOL) -> list[str]:
    """Sentences for the critic from per-numerics-mode summaries ({"off": ..., "on": ...}): the A12.3 outcome-B
    conditions (at tf32 off, B = 1 against B = batch_size beyond tol or a diverging generation; at either numerics,
    the same-shape swap beyond tol, i.e. row j changing row i) and every BUG. Facts only; the critic rules."""
    out = []
    what = {"logits": "prompt logits", "last": "first-answer-token logits", "latents": "fed-back latents"}
    for m, s in results.items():
        iso = s["isolation"]
        for q in what:
            for name in SETS:
                c = iso[f"{q}_{name}"]
                if m == "off" and c["over_tol"]:
                    out.append(f"tf32 off: {c['over_tol']} of {c['n']} examples' {what[q]} differ by more than {tol} "
                               f"between B = 1 and B = {batch_size} ({name} batches) (7.10 item 5; A12.3 outcome B)")
            c = iso[f"{q}_swap"]
            if c["over_tol"]:
                out.append(f"tf32 {m}: row j changes row i: {c['over_tol']} of {c['n']} examples' {what[q]} move by "
                           f"more than {tol} when only the companions' content changes (A12.3 outcome B)")
        for name in SETS:
            g = s["generation"][name]
            if m == "off" and g["diverging"]:
                out.append(f"tf32 off: {g['diverging']} of {g['n']} greedy generations diverge between B = 1 and "
                           f"B = {batch_size} ({name} batches) (A12.3 outcome B)")
        out += [f"tf32 {m}: BUG: {b}" for b in s.get("bug", [])]
    return out


# --------------------------------------------------------------------------------------------- band_logits
# A6.2 rule 3's strict-fp32 logits check (critic entry 2026-10-20T12:40Z, the registered diagnostic, checks (1) to (4);
# A14.1; the 10:51Z row part (3)). The Recorder above follows the HF path only (a latent_band model never calls the
# backbone); PathRecorder follows LatentLM.generate on either path. Nothing here decodes on its own: generation is
# LatentLM.generate on batches collated by latent.data.Collator from the prompts evaluate builds (prompt_records over
# the whole split, so record 0's tokenization check is the one every read ran), scored as evaluate scores.
BAND_NUMERICS = ("strict", "registered")     # the caller's numerics sets: strict fp32, the read's registered set
BAND_DTYPES = {torch.float32: "fp32", torch.float64: "float64"}   # the strict set's weight dtype and its label


class PathRecorder:
    """Watches LatentLM.generate on the HF path or on a latent_band path. Step 0 is the vector generate takes its
    first token from: LatentLM._run's logits at each row's last attended prompt column (the first-answer-token
    logits; with latents, the column after the last latent). Step s >= 1 is the LM head's output of decode pass s (one
    head call per decode pass on both paths: _pass on the HF path, _band_decode on the band path). One entry per
    generate call in `calls`: padded width and per row the emitted stream (tokens, then eos when the row stopped on
    it), `first` (the full step-0 vector, float cpu), the top-2 values / ids of every step, `step0_consistent` (the
    first emitted token attains the step-0 maximum) and the fed-back latents (generate's return_latents). Recorded
    tensors keep float64 on a float64 model (band_logits dtype=float64) and are float32 otherwise."""

    def __init__(self, lm):
        self.lm = lm
        self.calls: list[dict] = []
        self._cur = None
        self._hook = None

    def __enter__(self):
        gen, run = self.lm.generate, self.lm._run

        def generate(input_ids, attention_mask, position_ids, **kw):
            B, L = input_ids.shape
            cols = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
            last = torch.where(attention_mask.bool(), cols, torch.full_like(cols, -1)).max(1).values   # as _generate
            self._cur = {"B": B, "L": L, "last": last, "max_new_tokens": int(kw["max_new_tokens"]), "first": None,
                         "steps": [], "decoding": False, "runs": 0}
            try:
                out = gen(input_ids, attention_mask, position_ids, **kw)
                self._end(out)
            finally:
                self._cur = None
            return out

        def _run(*args, **kw):
            out = run(*args, **kw)
            c = self._cur
            if c is not None:
                c["runs"] += 1
                lg = _keep(out.logits[torch.arange(c["B"], device=out.logits.device), c["last"]])
                c["first"], c["steps"], c["decoding"] = lg, [lg.topk(2, dim=-1)], True
            return out
        self.lm.generate = generate          # instance attributes: generate's self._run(...) goes through them
        self.lm._run = _run
        self._hook = self.lm.lm_head.register_forward_hook(self._head)
        return self

    def __exit__(self, *exc):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self.lm.__dict__.pop("generate", None)
        self.lm.__dict__.pop("_run", None)
        return False

    def _head(self, module, inputs, output):
        c = self._cur
        if c is not None and c["decoding"]:
            c["steps"].append(_keep(output[:, -1]).topk(2, dim=-1))
        return None

    def _end(self, out):
        c = self._cur
        if c["runs"] != 1:
            raise RuntimeError(f"{c['runs']} LatentLM._run calls in one generate call (the recorder expects 1)")
        tv = torch.stack([v for v, _ in c["steps"]], 1).cpu()
        ti = torch.stack([i for _, i in c["steps"]], 1).cpu()
        first = c["first"].cpu()
        rows = []
        for b in range(c["B"]):
            toks = list(out.tokens[b])
            capped = bool(out.capped[b]) if out.capped is not None else False
            eos = len(toks) < c["max_new_tokens"] and not capped
            stream = toks + ([self.lm.eos_id] if eos else [])
            ok = not stream or bool(first[b, stream[0]] == first[b].max())
            rows.append({"width": c["L"], "tokens": toks, "stream": stream, "eos": eos,
                         "hit_cap": len(toks) >= c["max_new_tokens"], "first": first[b], "step0_consistent": ok,
                         "top2_values": tv[b, :len(stream)], "top2_ids": ti[b, :len(stream)],
                         "latents": _keep(out.latents[b]).cpu() if out.latents is not None else None})
        self.calls.append({"width": c["L"], "B": c["B"], "rows": rows})


def band_model(lm) -> LatentLM:
    """Band [0, L] over lm's own weights (A6.2 rule 1's denominator): the LatentLM build_model makes for a config with
    latent_band [0, L] (run_eval.py --set latent_band=[0,12] at L = 12), sharing lm.base_causallm (no copy)."""
    if lm.latent_band is not None:
        raise ValueError("band_logits compares the HF path with band [0, L]: give the full-depth (HF-path) model")
    return LatentLM(lm.base_causallm, lm.latent_id, lm.start_id, lm.end_id, lm.eos_id,
                    latent_band=(0, len(lm.backbone.h)))


def generate_rows(lm, tokenizer, records: list[dict], examples_by_idx: dict, *, mode: str, device: str,
                  max_new_tokens: int) -> dict[int, dict]:
    """One LatentLM.generate call on `records` collated as evaluate collates a batch (greedy, return_latents), watched
    by a PathRecorder. Per idx: the recorder's row, the batch size B, the prompt length and `correct` (evaluate's
    scoring: decode prompt + tokens with skip_special_tokens, answer_extract against answer_normalize)."""
    b = Collator(tokenizer.pad_token_id, lm.latent_id)(records)
    with PathRecorder(lm) as rec:
        lm.generate(b["input_ids"].to(device), b["attention_mask"].to(device), b["position_ids"].to(device),
                    max_new_tokens=max_new_tokens, mode=mode, scramble_bank=None, return_latents=True)
    (call,) = rec.calls
    out = {}
    for row, r in zip(call["rows"], records):
        text = tokenizer.decode(list(r["input_ids"]) + row["tokens"], skip_special_tokens=True)
        out[r["idx"]] = {**row, "B": len(records), "prompt_len": len(r["input_ids"]),
                         "correct": answer_extract(text) == answer_normalize(examples_by_idx[r["idx"]].answer)}
    return out


def _compare(a: dict, b: dict, na: str, nb: str) -> dict:
    """Two generations of one example: first-answer-token logits and fed latents (max abs difference), the step-0
    argmax and top-2 margin under each, the first divergent stream position with the top-2 margin and ids there under
    each, emitted counts, widths and correctness."""
    t = first_divergence(a["stream"], b["stream"])
    return {"first_maxabs": _maxabs(a["first"], b["first"]), "latents_maxabs": _maxabs(a["latents"], b["latents"]),
            f"first_argmax_{na}": int(a["first"].argmax()), f"first_argmax_{nb}": int(b["first"].argmax()),
            f"first_margin_{na}": _margin(a, 0), f"first_margin_{nb}": _margin(b, 0),
            "diverges": t is not None, "div_pos": t, f"margin_{na}": _margin(a, t), f"margin_{nb}": _margin(b, t),
            f"top2_{na}": _top2(a, t), f"top2_{nb}": _top2(b, t), f"emitted_{na}": len(a["tokens"]),
            f"emitted_{nb}": len(b["tokens"]), f"width_{na}": a["width"], f"width_{nb}": b["width"],
            f"B_{na}": a["B"], f"B_{nb}": b["B"], f"correct_{na}": a["correct"], f"correct_{nb}": b["correct"],
            "step0_consistent": a["step0_consistent"] and b["step0_consistent"]}


def paths_check(lm, band, tokenizer, batch: list[dict], examples_by_idx: dict, *, mode: str, device: str,
                max_new_tokens: int) -> list[dict]:
    """Checks (1) and (2): `batch` generated once on the HF path (lm) and once on band [0, L] (band) at the batch's
    shape; per example the comparison of the two (_compare, "hf" against "band")."""
    kw = dict(mode=mode, device=device, max_new_tokens=max_new_tokens)
    hf = generate_rows(lm, tokenizer, batch, examples_by_idx, **kw)
    bd = generate_rows(band, tokenizer, batch, examples_by_idx, **kw)
    return [{"idx": r["idx"], "prompt_len": len(r["input_ids"]), **_compare(hf[r["idx"]], bd[r["idx"]], "hf", "band")}
            for r in batch]


def shapes_check(band, tokenizer, records: list[dict], targets: list[int], examples_by_idx: dict, *, batch_size: int,
                 mode: str, device: str, max_new_tokens: int, keep: dict | None = None) -> list[dict]:
    """Check (3): per target idx, band [0, L] at B = 1 against B = batch_size in the read's own batch (the file-order
    chunk of `records` of batch_size holding it, generated once per chunk); _compare "b1" against "batch". `keep`
    (a dict) receives the B = 1 rows by idx (check (1b) compares them with the HF path at B = 1)."""
    kw = dict(mode=mode, device=device, max_new_tokens=max_new_tokens)
    pos = {r["idx"]: k for k, r in enumerate(records)}
    big = {}
    for k in sorted({pos[i] // batch_size for i in targets}):
        big.update(generate_rows(band, tokenizer, records[k * batch_size:(k + 1) * batch_size], examples_by_idx, **kw))
    rows = []
    for i in targets:
        one = generate_rows(band, tokenizer, [records[pos[i]]], examples_by_idx, **kw)[i]
        if keep is not None:
            keep[i] = one
        rows.append({"idx": i, "prompt_len": len(records[pos[i]]["input_ids"]),
                     "batch_start": pos[i] // batch_size * batch_size, **_compare(one, big[i], "b1", "batch")})
    return rows


def band_b1_rows(band, tokenizer, records: list[dict], targets: list[int], examples_by_idx: dict, *, mode: str,
                 device: str, max_new_tokens: int) -> dict[int, dict]:
    """Band [0, L] at B = 1 on each target idx (one generate call per target's record, as shapes_check makes its
    B = 1 rows): check (1b)'s band rows when check (3) is not run."""
    pos = {r["idx"]: k for k, r in enumerate(records)}
    return {i: generate_rows(band, tokenizer, [records[pos[i]]], examples_by_idx, mode=mode, device=device,
                             max_new_tokens=max_new_tokens)[i] for i in targets}


def swap_check(band, tokenizer, batch: list[dict], pool: list[dict], *, mode: str, device: str) -> list[dict]:
    """Check (4): A14.1's same-shape companion swap on band [0, L] at B = len(batch): per row i, the prefill
    (LatentLM._run, as generate's) of the batch against the batch with every companion replaced by a record of `pool`
    (examples outside the batch, in the seeded order) that fits its shape (_swap_batch: row i's tokens, mask,
    positions and the tensor shapes fixed): max abs differences of the attended prompt logits ("logits"), the
    first-answer-token logits ("last") and the fed latents ("latents"); "rep": the same batch computed twice."""
    collate = Collator(tokenizer.pad_token_id, band.latent_id)
    big = collate(batch)
    base = _prefill(band, big, mode, device)
    again = _prefill(band, big, mode, device)
    rows = []
    for r, rec in enumerate(batch):
        d = {"idx": rec["idx"], "prompt_len": len(rec["input_ids"]), "width": int(big["input_ids"].shape[1])}
        _diff(d, "rep", base[r], again[r])
        sb = (_swap_batch(collate, batch, r, pool, big, band.latent_id, tokenizer.pad_token_id) if len(batch) >= 2
              else None)
        if sb is None:
            d.update(logits_swap=None, last_swap=None, latents_swap=None)
        else:
            _diff(d, "swap", base[r], _prefill(band, sb, mode, device)[r])
        rows.append(d)
    return rows


def paths_b1_check(lm, tokenizer, records: list[dict], targets: list[int], band_b1: dict, examples_by_idx: dict, *,
                   mode: str, device: str, max_new_tokens: int) -> list[dict]:
    """Check (1b): per target idx, the HF path at B = 1 against band [0, L] at B = 1 (band_b1: check (3)'s B = 1
    rows, shapes_check's `keep`); _compare "hf" against "band"."""
    pos = {r["idx"]: k for k, r in enumerate(records)}
    rows = []
    for i in targets:
        hf = generate_rows(lm, tokenizer, [records[pos[i]]], examples_by_idx, mode=mode, device=device,
                           max_new_tokens=max_new_tokens)[i]
        rows.append({"idx": i, "prompt_len": len(records[pos[i]]["input_ids"]),
                     **_compare(hf, band_b1[i], "hf", "band")})
    return rows


class DtypeWatch:
    """Forward hooks on every module of a model: the floating dtypes of every module output seen (the activations),
    beside the floating dtypes of its parameters and buffers (the weights)."""

    def __init__(self, model):
        self.model, self.seen, self._hooks = model, set(), []

    def _add(self, x):
        if torch.is_tensor(x):
            if x.is_floating_point():
                self.seen.add(str(x.dtype).replace("torch.", ""))
        elif isinstance(x, dict):
            for v in x.values():
                self._add(v)
        elif isinstance(x, (tuple, list)):
            for v in x:
                self._add(v)

    def __enter__(self):
        self._hooks = [m.register_forward_hook(lambda mod, i, o: self._add(o)) for m in self.model.modules()]
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        return False

    def weights(self) -> list[str]:
        return sorted({str(t.dtype).replace("torch.", "") for t in (*self.model.parameters(), *self.model.buffers())
                       if t.is_floating_point()})


def _quantity(rows: list[dict], key: str, tol: float) -> dict:
    """Examples compared, the max (NaN when any value is NaN) and its idx, and the idx over tol (NaN counts over)."""
    have = [(r["idx"], r[key]) for r in rows if r.get(key) is not None]
    nan = [i for i, v in have if v != v]
    fin = [(i, v) for i, v in have if v == v]
    top = max(fin, key=lambda x: x[1]) if fin else None
    over = [i for i, v in have if not v <= tol]
    return {"n": len(have), "max": float("nan") if nan else (top[1] if top else None),
            "idx_max": nan[0] if nan else (top[0] if top else None), "over_tol": len(over), "idx_over_tol": over,
            "nan": len(nan)}


def _within(q: dict) -> bool:
    return q["n"] > 0 and q["over_tol"] == 0


def _gen_summary(rows: list[dict], na: str, nb: str, tol: float) -> dict:
    div = [r for r in rows if r["diverges"]]
    keep = ("idx", "div_pos", f"margin_{na}", f"margin_{nb}", f"top2_{na}", f"top2_{nb}", f"first_margin_{na}",
            f"first_margin_{nb}", f"correct_{na}", f"correct_{nb}", f"width_{na}", f"width_{nb}", "prompt_len")
    return {"n": len(rows), "first_answer_logits": _quantity(rows, "first_maxabs", tol),
            "latents": _quantity(rows, "latents_maxabs", tol),
            "first_argmax_differs": sum(r[f"first_argmax_{na}"] != r[f"first_argmax_{nb}"] for r in rows),
            "diverging": len(div), "idx_diverging": [r["idx"] for r in div],
            "divergences": [{k: r[k] for k in keep if k in r} for r in div],
            "correct": {na: sum(r[f"correct_{na}"] for r in rows), nb: sum(r[f"correct_{nb}"] for r in rows)},
            "correct_differs": sum(r[f"correct_{na}"] != r[f"correct_{nb}"] for r in rows),
            "idx_correct_differs": [r["idx"] for r in rows if r[f"correct_{na}"] != r[f"correct_{nb}"]],
            "step0_inconsistent": [r["idx"] for r in rows if not r["step0_consistent"]],
            "widths": sorted({r[f"width_{nb}"] for r in rows})}


def _flags_now(device: str) -> dict:
    """TF32 flags and autocast state as torch reports them (recorded at the start and the end of each numerics set)."""
    dt = device.split(":")[0]
    on = bool(torch.is_autocast_enabled(dt))
    return {"cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "autocast_enabled": on,
            "autocast_dtype": str(torch.get_autocast_dtype(dt)).replace("torch.", "") if on else None}


BAND_CHECKS = ("1", "1b", "2", "3", "4")     # band_logits' checks in record order (run_diag.py --checks)
BAND_NOT_RUN = "not run (--checks)"          # the verdict of a check the caller did not request (no flag line)


def band_checks(checks=None, *, strict_only: bool = False, paths_b1: bool = False) -> list[str]:
    """The checks a band_logits run makes, in BAND_CHECKS order. None: every check the settings make ((1); (1b) with
    paths_b1; (2) unless strict_only; (3); (4)), the behaviour before --checks. Otherwise the listed ones, refused
    (ValueError) when empty, outside BAND_CHECKS, (1b) without paths_b1 or (2) with strict_only."""
    made = [c for c in BAND_CHECKS if (c != "1b" or paths_b1) and (c != "2" or not strict_only)]
    if checks is None:
        return made
    want = [str(c).strip() for c in checks]
    bad = [c for c in want if c not in BAND_CHECKS]
    if not want or bad:
        raise ValueError(f"checks {want!r}: a non-empty list from {list(BAND_CHECKS)}")
    if "1b" in want and not paths_b1:
        raise ValueError("check (1b) runs with paths_b1 (run_diag.py --strict-only) only")
    if "2" in want and strict_only:
        raise ValueError("check (2) runs under the registered set, which strict_only does not run")
    return [c for c in BAND_CHECKS if c in want]


def _release(device: str) -> None:
    """Between band_logits' checks: collect what the finished check left unreferenced and, on cuda, return the caching
    allocator's free blocks (torch.cuda.empty_cache), so the next check starts from what the last one released.
    Changes no number."""
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def band_logits(lm, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str,
                max_new_tokens: int, numerics: dict, extra_idx=(), perm_seed: int = 0, tol: float = TOL,
                dtype=torch.float32, strict_only: bool = False, paths_b1: bool = False, checks=None,
                on_check=None) -> tuple[list[dict], dict]:
    """A6.2 rule 3's four checks (critic entry 2026-10-20T12:40Z) on the full-depth checkpoint `lm` (HF path) against
    band [0, L] over the same weights. `numerics` maps "strict" and "registered" to zero-argument callables returning
    a context manager that sets (and asserts) that numerics set and yields its read-back state; checks (1), (3) and
    (4) run under "strict", then (2), (3) and (4) under "registered" ((3) there is printed beside; (4)'s line holds at
    each); the strict set is refused (RuntimeError) unless torch reads back TF32 off, "highest", no autocast and fp32
    weights, and either set is refused when its flags move during its checks. The batch is the split's first
    batch_size examples in file order (the read's first batch); (3) adds extra_idx, each at B = batch_size in its own
    file-order batch. `dtype` is the weight dtype the strict set requires (float32: the registered check; float64: the
    same comparison with kernel-order noise near 1e-12, which tells kernel-order noise from a path difference; the
    caller casts the model); `strict_only` runs the strict set alone (`numerics` then maps "strict" only; (2) and the
    registered columns of (3) and (4) are not run), and float64 requires it (the registered set is autocast over fp32);
    with strict_only the floating dtypes of the weights and of every module output during the checks are recorded
    ("dtypes") and must all be `dtype` (RuntimeError otherwise). `paths_b1` adds check (1b) under the strict set: the
    HF path at B = 1 against band [0, L] at B = 1 (check (3)'s B = 1 rows) on (3)'s targets.
    `checks` (band_checks; default None = every check the settings make, the record as before) runs the listed checks
    only, in the order above; a check not listed is recorded with verdict BAND_NOT_RUN, no line and no flag line
    ((1b) without (3) generates band [0, L]'s B = 1 rows itself, as (3) does). Between checks the finished check's
    rows are dropped once consumed and memory is released (_release; no number changes). `on_check(rows, summary)`,
    when given, is called after each check with the rows so far and the summary of the checks finished so far
    (`complete` false; a check run under both sets is finished after the second), so a caller can write the record
    incrementally; the flags and dtypes refusals are applied at each call as at the set's end.
    Returns (rows: one per check, numerics set and
    example; summary: per check its quantities, maxima, counts and verdict, the numerics read back, "bug",
    "checks_requested" and "complete" true)."""
    if arm not in E.LATENT_ARMS:
        raise ValueError(f"band_logits needs a latent-feedback arm {E.LATENT_ARMS}, not {arm!r}")
    if dtype not in BAND_DTYPES:
        raise ValueError(f"dtype {dtype} is not one of {sorted(map(str, BAND_DTYPES))}")
    if dtype != torch.float32 and not strict_only:
        raise ValueError(f"a {BAND_DTYPES[dtype]} run takes the strict set alone (strict_only): the registered set is "
                         f"autocast over fp32 and has no {BAND_DTYPES[dtype]} form")
    sets = ("strict",) if strict_only else BAND_NUMERICS
    if set(numerics) != set(sets):
        raise ValueError(f"numerics must map exactly {sets}")
    req = band_checks(checks, strict_only=strict_only, paths_b1=paths_b1)
    lab = BAND_DTYPES[dtype]
    band = band_model(lm)
    L = band.latent_band[1]
    mode = E.gen_mode(arm)
    records = prompt_records(tokenizer, examples, arm=arm, n_latent=n_latent)
    by_ex = {ex.idx: ex for ex in examples}
    pos = {r["idx"]: k for k, r in enumerate(records)}
    missing = [i for i in extra_idx if i not in pos]
    if missing:
        raise ValueError(f"extra idx {missing} not in the split (idx 0 to {len(records) - 1})")
    batch = records[:batch_size]
    inside = [r["idx"] for r in batch]
    targets = inside + [i for i in dict.fromkeys(extra_idx) if i not in inside]
    pool = [records[k] for k in permutation(len(records), perm_seed) if records[k]["idx"] not in set(inside)]
    gen = dict(mode=mode, device=device, max_new_tokens=max_new_tokens)
    res = {k: {} for k in sets}         # per set: "p" (check (1) / (2)), "p1" ((1b)), "s" ((3)), "w" ((4)): rows
    states = {}

    def rows_now() -> list[dict]:
        out = []
        for k in sets:
            r = res[k]
            out += [{"check": "1" if k == "strict" else "2", "numerics": k, **x} for x in r.get("p", [])]
            out += [{"check": "1b", "numerics": k, **x} for x in r.get("p1", [])]
            out += [{"check": "3", "numerics": k, **x} for x in r.get("s", [])]
            out += [{"check": "4", "numerics": k, **x} for x in r.get("w", [])]
        return out

    def finished() -> list[str]:
        have = {"1": "p" in res["strict"], "1b": "p1" in res["strict"],
                "2": "registered" in res and "p" in res["registered"],
                "3": all("s" in res[k] for k in sets), "4": all("w" in res[k] for k in sets)}
        return [c for c in req if have[c]]

    def summarize(states_now: dict, dtypes_now, complete: bool) -> dict:
        done = finished()

        def shapes(s):
            g = _gen_summary(s, "b1", "batch", tol)
            g["of_batch"] = {"n": sum(r["idx"] in inside for r in s),
                             "diverging": sum(r["diverges"] for r in s if r["idx"] in inside)}
            g["extra"] = {str(r["idx"]): {k: r[k] for k in ("batch_start", "B_batch", "width_b1", "width_batch",
                                                            "first_maxabs", "latents_maxabs", "first_margin_b1",
                                                            "first_margin_batch", "first_argmax_b1",
                                                            "first_argmax_batch", "diverges", "div_pos", "margin_b1",
                                                            "margin_batch", "top2_b1", "top2_batch", "emitted_b1",
                                                            "emitted_batch", "correct_b1", "correct_batch")}
                          for r in s if r["idx"] in extra_idx}
            return g

        def swaps(w):
            return {"n": len(w), **{f"{q}_{c}": _quantity(w, f"{q}_{c}", tol) for c in ("swap", "rep")
                                    for q in ("last", "logits", "latents")}}
        c1 = _gen_summary(res["strict"]["p"], "hf", "band", tol) if "1" in done else None
        c2 = _gen_summary(res["registered"]["p"], "hf", "band", tol) if "2" in done else None
        c3 = {k: shapes(res[k]["s"]) for k in sets} if "3" in done else None
        c4 = {k: swaps(res[k]["w"]) for k in sets} if "4" in done else None
        c1b = None
        if "1b" in done:
            p1 = res["strict"]["p1"]
            c1b = _gen_summary(p1, "hf", "band", tol)
            c1b["of_batch"] = {"n": sum(r["idx"] in inside for r in p1),
                               "first_answer_logits": _quantity([r for r in p1 if r["idx"] in inside], "first_maxabs",
                                                                tol)}
            v1b = "pass" if _within(c1b["first_answer_logits"]) else "BUG"
        bug, flags = [], []
        if c1 is not None:
            v1 = "pass" if _within(c1["first_answer_logits"]) else "BUG"
            q = c1["first_answer_logits"]
            flags.append(f"check (1), strict {lab}: max |first-answer-token logits| HF path vs band [0, {L}] = "
                         f"{q['max']:.3e} over {q['n']} examples (idx {q['idx_max']}), line <= {tol}: {v1}; "
                         f"{c1['diverging']} of {c1['n']} greedy generations diverge between the two paths")
            if v1 != "pass":
                bug.append(f"check (1): {q['over_tol']} of {q['n']} examples' first-answer-token logits differ by more "
                           f"than {tol} between the HF path and band [0, {L}] at strict {lab} (max {q['max']:.3e})")
        if c1b is not None:
            q = c1b["first_answer_logits"]
            flags.append(f"check (1b), strict {lab}: max |first-answer-token logits| HF path vs band [0, {L}] at B = 1 "
                         f"= {q['max']:.3e} over {q['n']} examples (idx {q['idx_max']}), line <= {tol}: {v1b}; latents "
                         f"max {c1b['latents']['max']:.3e}; {c1b['first_argmax_differs']} argmax differences; "
                         f"{c1b['diverging']} of {c1b['n']} greedy generations diverge")
            if v1b != "pass":
                bug.append(f"check (1b): {q['over_tol']} of {q['n']} examples' first-answer-token logits differ by more "
                           f"than {tol} between the HF path and band [0, {L}] at B = 1 at strict {lab} (max "
                           f"{q['max']:.3e})")
        if strict_only:
            flags.append("check (2), registered numerics: not run (strict set alone)")
        elif c2 is not None:
            q = c2["first_answer_logits"]
            flags.append(f"check (2), registered numerics: max |first-answer-token logits| HF path vs band [0, {L}] = "
                         f"{q['max']:.3e}; {c2['diverging']} of {c2['n']} greedy generations diverge (no line)")
        if c3 is not None:
            for k in sets:
                g = c3[k]
                flags.append(f"check (3), {k}: band [0, {L}] B = 1 vs B = {batch_size}: {g['diverging']} of {g['n']} "
                             f"greedy generations diverge ({g['of_batch']['diverging']} of the batch's "
                             f"{g['of_batch']['n']}; idx {g['idx_diverging']}); max |first-answer-token logits| "
                             f"{g['first_answer_logits']['max']:.3e}" + ("" if k == "strict" else " (beside)"))
            v3 = c3["strict"]["n"] > 0 and c3["strict"]["diverging"] == 0
            if not v3:
                bug.append(f"check (3): {c3['strict']['diverging']} of {c3['strict']['n']} greedy generations diverge "
                           f"between B = 1 and B = {batch_size} on band [0, {L}] at strict {lab} (idx "
                           f"{c3['strict']['idx_diverging']})")
        if c4 is not None:
            v4 = all(_within(c4[k][f"{q}_swap"]) for k in sets for q in ("last", "logits", "latents"))
            for k in sets:
                flags.append(f"check (4), {k}: same-shape companion swap on band [0, {L}] at B = {len(batch)}: max "
                             + ", ".join(f"{q} {c4[k][q + '_swap']['max']:.3e}" if c4[k][q + "_swap"]["max"] is not None
                                         else f"{q} none" for q in ("last", "logits", "latents"))
                             + f" (line <= {tol}); batch repeated: last {c4[k]['last_rep']['max']:.3e}")
                for qn in ("last", "logits", "latents"):
                    g = c4[k][f"{qn}_swap"]
                    if not _within(g):
                        bug.append(f"check (4), {k}: {g['over_tol']} of {g['n']} examples' {qn} move by more than "
                                   f"{tol} when only the companions change (idx {g['idx_over_tol']})" if g["n"] else
                                   f"check (4), {k}: no swap batch fits ({qn})")
        for name, g in (("(1)", c1), ("(1b)", c1b), ("(2)", c2), ("(3) strict", (c3 or {}).get("strict")),
                        ("(3) registered", (c3 or {}).get("registered"))):
            if g is not None and g["step0_inconsistent"]:
                bug.append(f"check {name}: the recorder's step-0 maximum is not the first emitted token on idx "
                           f"{g['step0_inconsistent']} (instrument)")
        ch = {}
        if c1 is not None:
            ch["1"] = {"numerics": "strict", "line": f"<= {tol}", "verdict": v1, **c1}
        elif "1" not in req:
            ch["1"] = {"numerics": "strict", "line": None, "verdict": BAND_NOT_RUN}
        if c1b is not None:
            ch["1b"] = {"numerics": "strict", "line": f"<= {tol}", "verdict": v1b, "B": 1, **c1b}
        elif paths_b1 and "1b" not in req:
            ch["1b"] = {"numerics": "strict", "line": None, "verdict": BAND_NOT_RUN, "B": 1}
        if strict_only:
            ch["2"] = {"numerics": "registered", "line": None, "verdict": "not run (strict set alone)"}
        elif c2 is not None:
            ch["2"] = {"numerics": "registered", "line": None, "verdict": "printed (no line)", **c2}
        elif "2" not in req:
            ch["2"] = {"numerics": "registered", "line": None, "verdict": BAND_NOT_RUN}
        n3 = "strict (line)" + ("" if strict_only else "; registered beside")
        if c3 is not None:
            ch["3"] = {"numerics": n3, "line": f"0 diverging at strict {lab}",
                       "verdict": f"0 diverging at strict {lab}" if v3 else
                       f"{c3['strict']['diverging']} diverging at strict {lab}", **c3}
        elif "3" not in req:
            ch["3"] = {"numerics": n3, "line": None, "verdict": BAND_NOT_RUN}
        n4 = "strict" if strict_only else "strict and registered"
        if c4 is not None:
            ch["4"] = {"numerics": n4, "line": f"<= {tol} at each",
                       "verdict": f"within {tol} at each" if v4 else f"over {tol}", **c4}
        elif "4" not in req:
            ch["4"] = {"numerics": n4, "line": None, "verdict": BAND_NOT_RUN}
        return {"band": [0, L], "n_layer": L, "mode": mode, "n_latent": n_latent, "tol": tol,
                "batch_size": batch_size, "dtype": str(dtype).replace("torch.", ""), "sets": list(sets),
                **({"dtypes": dtypes_now} if strict_only else {}),
                "batch": {"rule": "the split's first batch_size examples in file order (the read's first batch)",
                          "idx": inside},
                "extra_idx": list(extra_idx), "perm_seed": perm_seed, "swap_pool": len(pool), "numerics": states_now,
                "checks": ch, "flags": flags, "bug": bug, "checks_requested": list(req), "checks_run": done,
                "complete": complete}

    was = (lm.training, band.training)
    lm.eval()
    band.eval()
    dtypes = None
    try:
        for name in sets:
            with numerics[name]() as st:
                start, t0 = _flags_now(device), time.time()
                if name == "strict" and (start["cuda.matmul.allow_tf32"] or start["cudnn.allow_tf32"]
                                         or start["float32_matmul_precision"] != "highest"
                                         or start["autocast_enabled"]
                                         or next(lm.parameters()).dtype != dtype):
                    raise RuntimeError(f"the strict set is not strict {lab} ({lab} weights, TF32 off, 'highest', no "
                                       f"autocast): {start}, weights {next(lm.parameters()).dtype}")
                watch = DtypeWatch(lm.base_causallm) if strict_only else contextlib.nullcontext()
                r = res[name]

                def seen():                     # the set's refusals (flags moved; not `dtype` throughout)
                    dt = None
                    if strict_only:
                        want = [str(dtype).replace("torch.", "")]
                        dt = {"weights": watch.weights(), "activations": sorted(watch.seen)}
                        if dt["weights"] != want or dt["activations"] != want:
                            raise RuntimeError(f"the strict set is not {lab} throughout: {dt}")
                    now = _flags_now(device)
                    if now != start:
                        raise RuntimeError(f"numerics {name!r} changed during the checks: {start} -> {now}")
                    return dt, now

                def step():                     # after each check: release, then the partial record
                    _release(device)
                    if on_check is not None:
                        dt, now = seen()
                        sts = {**states, name: {**dict(st), "torch_at_start": start, "torch_at_end": now,
                                                "seconds": round(time.time() - t0, 3)}}
                        on_check(rows_now(), summarize(sts, dt, False))
                with watch:
                    if ("1" if name == "strict" else "2") in req:
                        r["p"] = paths_check(lm, band, tokenizer, batch, by_ex, **gen)
                        step()
                    want_b1 = name == "strict" and "1b" in req
                    b1 = {} if want_b1 else None
                    if "3" in req:
                        r["s"] = shapes_check(band, tokenizer, records, targets, by_ex, batch_size=batch_size,
                                              keep=b1, **gen)
                        step()
                    elif want_b1:
                        b1 = band_b1_rows(band, tokenizer, records, targets, by_ex, **gen)
                    if want_b1:
                        r["p1"] = paths_b1_check(lm, tokenizer, records, targets, b1, by_ex, **gen)
                        del b1                  # (3)'s B = 1 rows: consumed by (1b)
                        step()
                    if "4" in req:
                        r["w"] = swap_check(band, tokenizer, batch, pool, mode=mode, device=device)
                        step()
                dt, end = seen()
                if strict_only:
                    dtypes = dt
                states[name] = {**dict(st), "torch_at_start": start, "torch_at_end": end,
                                "seconds": round(time.time() - t0, 3)}
    finally:
        lm.train(was[0])
        band.train(was[1])
    return rows_now(), summarize(states, dtypes, True)


# --------------------------------------------------------------------------------------------- donor_answer
# A6.1 rule 4 (B)'s directional read on GSM-50k, the donor-answer read (design note
# governance/designs/a6_1_directional_read.md, rev230; registered by critic entry 2026-10-20T17:36Z, which run_diag.py
# names as DONOR_ROW). The pass is `scramble` unchanged (evaluate(None) and evaluate("scrambled", scramble_seed) under
# Recorders with the fed-vector checks); the scorer is a pure function of the two passes' ExampleResults (their pred
# and gold strings), the donor map and the valid records' steps and answers: no model, no bank, no neighbouring row.
#   class C (ruling 1): the donor j = pi(i) has step min(n_steps(j), 3) result equal to its answer
#       (state_targets.answer_in_last_target on the donor's record alone; 295 of 500 on the valid file of record);
#   ambiguous (ruling 2): gold(i) == gold(j) after answer_normalize; excluded; scored S = C minus ambiguous (289);
#   d_own / d_scr: the own-pass / scrambled-pass prediction equals the donor's gold (R_own, R_dir over S);
#   changed: the scrambled prediction differs from the own-pass prediction (both answer_normalize'd; an empty
#       extraction is the empty string); g_chg = changed and not d_scr (ruling 3, the garbage of record: rule 5's
#       second clause fires iff G_chg > R_dir on the point fractions);
#   g_own / g_scr: neither gold (the gold-literal reading G_scr = 1 - acc_scr(S) - R_dir, and the excess reading
#       Delta_g = G_scr - G_own = drop_S - Delta_dir; both printed beside, deciding nothing);
#   intervals (ruling 4): 6.2's form Delta +- 1.96 sqrt(Var_ex), Var_ex = the variance (ddof 1, read_screen's K2) of
#       the draw means over n_boot paired draws of the scored indices with replacement, numpy default_rng(seed), one
#       index matrix (drawn in chunks as latent.eval._boot_ci draws) applied to every boolean; percentile beside.
DONOR_CLASS_STEPS = 3       # ruling 1: step min(n_steps, 3) (GSM-50k's final regime: 6 latents at c_thought 2)
DONOR_CHUNK = 1000          # the draws in chunks of 1000 rows, as latent.eval._boot_ci and read_screen draw them
DONOR_PASS_LINE = "directional read present iff Delta_dir > 0 and the lower bound > 0"     # 17:36Z (2), verbatim
DONOR_OUTCOMES = ("BUG", "held: identification check failed", "directional read present",
                  "directional read absent (interval includes 0)")                       # 17:36Z (2): the words
DONOR_THIN = 10             # 17:36Z (2): a present outcome on fewer than 10 net examples is labelled thin
DONOR_GARBAGE_LABELS = {    # 17:36Z ruling 3, verbatim
    "chg": "changed-output reading: G_chg > R_dir; rule 5's comparison of record",
    "literal": "gold-literal reading: 1 minus acc_scr(S) minus R_dir; fires at any accuracy under 50%; tests the "
               "accuracy, not the direction; decides nothing",
    "excess": "excess reading: Delta_g = drop_S minus Delta_dir; decides nothing"}
# examples.csv columns after the numerics set (the note's 2.4 with ruling 3's `changed` and `g_chg`): ids, booleans
# and numbers only, never a pred, gold or text string
DONOR_COLUMNS = ("idx", "donor", "donor_is_self", "fed_equals_donor", "fed_equals_collected", "fed_min_l2",
                 "fed_max_cos", "in_class", "ambiguous", "scored", "own_n_steps", "donor_n_steps",
                 "donor_last_sup_step", "correct_own", "correct_scr", "d_own", "d_scr", "changed", "g_chg", "g_own",
                 "g_scr", "scr_eq_any_donor_step", "scr_eq_any_own_step", "donor_gold_in_own_steps",
                 "first_maxabs_own_scr", "first_argmax_own", "first_argmax_scr", "stream_div_own_scr", "emitted_own",
                 "emitted_scr")
DONOR_BOOLS = ("d_scr", "d_own", "g_chg", "changed", "g_scr", "g_own", "correct_own", "correct_scr")


class _NoTokens:
    """record_targets' tokenizer for the class rule: the rule reads the step results, never their token lists, so no
    tokenizer enters it (and a test tokenizer that grows its vocabulary on encode is never touched)."""

    @staticmethod
    def encode(text, add_special_tokens=False):
        return []


def donor_in_class(ex, n_latent_steps: int = DONOR_CLASS_STEPS) -> bool:
    """Ruling 1's class test on ONE record (the donor's): its step min(n_steps, n_latent_steps) result, the text after
    the last "=" of a "<<expr=c>>" step, normalised, equals its normalised answer (state_targets.answer_in_last_target;
    a record with no step, or a last supervised step not of that form, is outside). GSM records only."""
    from latent.state_targets import answer_in_last_target, record_targets, task_kind
    if task_kind(ex) != "gsm":
        raise ValueError(f"record {getattr(ex, 'idx', '?')}: the donor-answer class is A6.1 rule 4 (B)'s GSM rule")
    return bool(answer_in_last_target(record_targets(ex, _NoTokens()), ex.answer, n_latent_steps))


def step_results(ex) -> set:
    """The normalised results of every "<<expr=c>>" step of a GSM record (a step of another form gives none)."""
    from latent.state_targets import gsm_step_result
    out = set()
    for st in ex.steps:
        r = gsm_step_result(st)
        if r is not None and answer_normalize(r):
            out.add(answer_normalize(r))
    return out


def donor_answer_rows(examples, own, scr, donors: dict, *, n_latent_steps: int = DONOR_CLASS_STEPS) -> list[dict]:
    """The pure scorer (one numerics set): per example i in file order, with donor j = donors[i], the class, ambiguity
    and scored flags, step counts, and the own / scrambled predictions against gold(i) and gold(j) (see the section
    comment). own / scr: the evaluate(None) and evaluate("scrambled") ExampleResults (lists or {idx: result}); every
    pred and gold is compared after answer_normalize and no string leaves this function. Refuses (ValueError)
    mismatched idx sets, a donor outside the split, and a result whose gold or correct disagrees with the record."""
    by = {ex.idx: ex for ex in examples}
    own = own if isinstance(own, dict) else {r.idx: r for r in own}
    scr = scr if isinstance(scr, dict) else {r.idx: r for r in scr}
    if set(own) != set(by) or set(scr) != set(by) or set(donors) != set(by):
        raise ValueError("own results, scrambled results, donors and examples must cover the same idx")
    if any(donors[i] not in by for i in donors):
        raise ValueError("a donor outside the split")
    cls = {i: donor_in_class(by[i], n_latent_steps) for i in by}
    steps = {i: step_results(by[i]) for i in by}
    rows = []
    for ex in examples:
        i, j = ex.idx, donors[ex.idx]
        gi, gj = answer_normalize(ex.answer), answer_normalize(by[j].answer)
        po, ps = answer_normalize(own[i].pred), answer_normalize(scr[i].pred)
        if own[i].gold != gi or scr[i].gold != gi:
            raise ValueError(f"idx {i}: a result's gold is not the record's normalised answer")
        co, cs, do, ds = po == gi, ps == gi, po == gj, ps == gj
        if co != bool(own[i].correct) or cs != bool(scr[i].correct):
            raise ValueError(f"idx {i}: correct differs from pred == gold (the eval's scoring)")
        amb = gi == gj
        nj = len(by[j].steps)
        changed = ps != po
        rows.append({"idx": i, "donor": j, "in_class": cls[j], "ambiguous": amb, "scored": cls[j] and not amb,
                     "own_n_steps": len(ex.steps), "donor_n_steps": nj,
                     "donor_last_sup_step": min(nj, int(n_latent_steps)), "correct_own": co, "correct_scr": cs,
                     "d_own": do, "d_scr": ds, "changed": changed, "g_chg": changed and not ds,
                     "g_own": not co and not do, "g_scr": not cs and not ds, "scr_eq_any_donor_step": ps in steps[j],
                     "scr_eq_any_own_step": ps in steps[i], "donor_gold_in_own_steps": gj in steps[i],
                     "emitted_own": int(own[i].emitted_tokens), "emitted_scr": int(scr[i].emitted_tokens)})
    return rows


def donor_boot(cols: dict, n_boot: int, seed: int, chunk: int = DONOR_CHUNK) -> dict:
    """Paired example bootstrap: one index matrix (numpy default_rng(seed); chunks of `chunk` rows of
    rng.integers(0, m, size=(k, m)), as latent.eval._boot_ci draws) applied to every per-example column of `cols`
    ({name: float array over the same m examples}). Returns {name: the n_boot draw means}."""
    m = len(next(iter(cols.values())))
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in cols}
    for s0 in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s0)
        ix = rng.integers(0, m, size=(k, m))
        for name, v in cols.items():
            out[name][s0:s0 + k] = v[ix].mean(1)
    return out


def _interval(point: float, draws) -> dict:
    """6.2's form point +- 1.96 sqrt(Var_ex) (Var_ex: ddof 1 over the draw means) and the percentile [2.5, 97.5]."""
    se = float(np.sqrt(draws.var(ddof=1)))
    plo, phi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    return {"se": se, "lo": point - 1.96 * se, "hi": point + 1.96 * se, "pct_lo": plo, "pct_hi": phi}


def _rates(rows: list[dict], n_boot: int, seed: int, chunk: int = DONOR_CHUNK) -> dict:
    """The donor-answer quantities over `rows` (a scored set or the complement): counts, fractions and the paired
    intervals of Delta_dir = R_dir - R_own, Delta_g = G_scr - G_own and G_chg - R_dir from one set of draws."""
    m = len(rows)
    k = {c: sum(bool(r[c]) for r in rows) for c in DONOR_BOOLS}
    out = {"n": m, "k_dir": k["d_scr"], "k_own": k["d_own"], "k_changed": k["changed"], "k_g_chg": k["g_chg"],
           "k_g_scr": k["g_scr"], "k_g_own": k["g_own"], "k_correct_own": k["correct_own"],
           "k_correct_scr": k["correct_scr"]}
    if m == 0:
        return {**out, **{x: None for x in ("r_dir", "r_own", "delta_dir", "g_chg", "g_scr", "g_own", "delta_g",
                                             "acc_own_S", "acc_scr_S", "drop_S", "chg_minus_dir", "se", "lo", "hi",
                                             "pct_lo", "pct_hi", "se_g", "lo_g", "hi_g", "pct_lo_g", "pct_hi_g",
                                             "se_chg", "lo_chg", "hi_chg", "pct_lo_chg", "pct_hi_chg")}}
    out.update(r_dir=k["d_scr"] / m, r_own=k["d_own"] / m, delta_dir=(k["d_scr"] - k["d_own"]) / m,
               g_chg=k["g_chg"] / m, g_scr=k["g_scr"] / m, g_own=k["g_own"] / m,
               delta_g=(k["g_scr"] - k["g_own"]) / m, acc_own_S=k["correct_own"] / m,
               acc_scr_S=k["correct_scr"] / m, drop_S=(k["correct_own"] - k["correct_scr"]) / m,
               chg_minus_dir=(k["g_chg"] - k["d_scr"]) / m)
    if m < 2:
        return {**out, **{x: None for x in ("se", "lo", "hi", "pct_lo", "pct_hi", "se_g", "lo_g", "hi_g", "pct_lo_g",
                                             "pct_hi_g", "se_chg", "lo_chg", "hi_chg", "pct_lo_chg", "pct_hi_chg")}}
    a = {c: np.array([float(r[c]) for r in rows]) for c in ("d_scr", "d_own", "g_scr", "g_own", "g_chg")}
    dr = donor_boot({"dir": a["d_scr"] - a["d_own"], "g": a["g_scr"] - a["g_own"], "chg": a["g_chg"] - a["d_scr"]},
                    n_boot, seed, chunk)
    out.update(_interval(out["delta_dir"], dr["dir"]))
    for tag, pt in (("g", out["delta_g"]), ("chg", out["chg_minus_dir"])):
        out.update({f"{k_}_{tag}": v for k_, v in _interval(pt, dr[tag]).items()})
    return out


def donor_answer_outcome(s: dict, bug: list, identification: str | None) -> str:
    """The record's outcome word (17:36Z (2)): BUG (the scramble bug list), held (identification, ruling 9), else
    DONOR_PASS_LINE on the 6.2-form interval; an absent read carries its sign in the line, not in the word."""
    if bug:
        return DONOR_OUTCOMES[0]
    if identification == "held":
        return DONOR_OUTCOMES[1]
    if s.get("lo") is not None and s["k_dir"] > s["k_own"] and s["lo"] > 0:
        return DONOR_OUTCOMES[2]
    return DONOR_OUTCOMES[3]


def donor_answer_summary(rows: list[dict], n_boot: int = 10_000, seed: int = 0, *,
                         n_latent_steps: int = DONOR_CLASS_STEPS, chunk: int = DONOR_CHUNK) -> dict:
    """One numerics set's record block over the scorer's rows: the class, the ambiguous pairs, the scored set S and
    its quantities (_rates), the three garbage readings (garbage_exceeds = G_chg > R_dir is rule 5's of record), the
    accuracies over every example (they equal the none / scrambled reads'), the complement class and the secondary
    diagnostics (deciding nothing). The outcome word needs the bug list and the identification: donor_answer_outcome."""
    rows = sorted(rows, key=lambda r: r["idx"])         # the draws index the scored set in idx (file) order
    n = len(rows)
    C = [r for r in rows if r["in_class"]]
    S = [r for r in rows if r["scored"]]
    amb = [r["idx"] for r in C if r["ambiguous"]]
    rs_ = _rates(S, n_boot, seed, chunk)
    out = {"class_rule": f"donor's step min(n_steps, {n_latent_steps}) result equals the donor's answer "
                         f"(state_targets.answer_in_last_target on the donor's record, n_latent_steps "
                         f"{n_latent_steps}); ambiguous pairs (own gold = donor gold) excluded",
           "n": n, "n_class": len(C), "n_ambiguous": len(amb), "ambiguous_idx": amb, "n_scored": len(S),
           "n_ambiguous_all": sum(r["ambiguous"] for r in rows), "pass_line": DONOR_PASS_LINE,
           **{k: v for k, v in rs_.items() if k != "n"}, "n_boot": n_boot, "boot_seed": seed}
    ok = rs_["r_dir"] is not None
    out["garbage_exceeds"] = bool(ok and rs_["k_g_chg"] > rs_["k_dir"])                # G_chg > R_dir, same n
    out["garbage_literal_exceeds"] = bool(ok and rs_["k_g_scr"] > rs_["k_dir"])
    out["garbage_excess_exceeds"] = bool(ok and rs_["k_g_scr"] - rs_["k_g_own"] > rs_["k_dir"] - rs_["k_own"])
    out["garbage_labels"] = dict(DONOR_GARBAGE_LABELS)
    k_net = rs_["k_dir"] - rs_["k_own"]
    out["thin"] = bool(k_net < DONOR_THIN)
    out["k_net"] = k_net
    out["acc_own"] = sum(r["correct_own"] for r in rows) / n if n else None
    out["acc_scr"] = sum(r["correct_scr"] for r in rows) / n if n else None
    out["k_acc_own"] = sum(r["correct_own"] for r in rows)
    out["k_acc_scr"] = sum(r["correct_scr"] for r in rows)
    out["drop"] = (out["k_acc_own"] - out["k_acc_scr"]) / n if n else None
    comp = [r for r in rows if not r["in_class"]]
    out["complement"] = {"rule": "donors outside the class (4 or more steps, or a last supervised step not of the "
                                 "<<expr=c>> form); diagnostic, decides nothing (section 10)",
                         "n_ambiguous": sum(r["ambiguous"] for r in comp), **_rates(comp, n_boot, seed, chunk)}
    out["secondary"] = {"decides": "nothing (section 10)",
                        "any_donor_step_rate": sum(r["scr_eq_any_donor_step"] for r in S) / len(S) if S else None,
                        "any_own_step_rate": sum(r["scr_eq_any_own_step"] for r in S) / len(S) if S else None,
                        "k_any_donor_step": sum(r["scr_eq_any_donor_step"] for r in S),
                        "k_any_own_step": sum(r["scr_eq_any_own_step"] for r in S),
                        "donor_gold_in_own_steps": sum(r["donor_gold_in_own_steps"] for r in C)}
    return out


def donor_answer(lm, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str,
                 max_new_tokens: int, scramble_seed: int = 0, n_boot: int = 10_000, boot_seed: int = 0,
                 n_latent_steps: int = DONOR_CLASS_STEPS, tol: float = TOL) -> tuple[list[dict], list[dict], dict]:
    """The donor-answer read (one numerics set): `scramble` unchanged, then the pure scorer over the two passes'
    results. Bug checks beyond scramble's: the donor map must equal scramble_donors({idx: n_latent}, scramble_seed)
    (the registered derangement: whole set, one class) and scramble's correct_own / correct_scr must equal the
    scorer's. Returns (rows in DONOR_COLUMNS order, scramble's per-rank rows, {"donor_answer", "scramble", "bug"})."""
    res: dict = {}
    sc_rows, ranks, sc = scramble(lm, tokenizer, examples, arm=arm, n_latent=n_latent, batch_size=batch_size,
                                  device=device, max_new_tokens=max_new_tokens, scramble_seed=scramble_seed, tol=tol,
                                  results_out=res)
    donors = {r["idx"]: r["donor"] for r in sc_rows}
    bug = list(sc["bug"])
    want = E.scramble_donors({ex.idx: n_latent for ex in examples}, seed=scramble_seed)
    if donors != want:
        k = sum(donors[i] != want[i] for i in want)
        bug.append(f"{k} examples' donors are not scramble_donors(seed {scramble_seed}) over the whole set (one "
                   f"class of {len(want)} at {n_latent} latents)")
    da = {r["idx"]: r for r in donor_answer_rows(examples, res["own"], res["scr"], donors,
                                                 n_latent_steps=n_latent_steps)}
    off = [r["idx"] for r in sc_rows if (r["correct_own"], r["correct_scr"]) != (da[r["idx"]]["correct_own"],
                                                                                da[r["idx"]]["correct_scr"])]
    if off:
        bug.append(f"{len(off)} examples' correct_own / correct_scr differ between scramble and the scorer")
    rows = []
    for r in sc_rows:
        m = {**r, **da[r["idx"]]}
        rows.append({c: m[c] for c in DONOR_COLUMNS})
    summ = donor_answer_summary(list(da[r["idx"]] for r in sc_rows), n_boot, boot_seed,
                                n_latent_steps=n_latent_steps)
    summ["scramble_seed"] = scramble_seed
    return rows, ranks, {"donor_answer": summ, "scramble": sc, "bug": bug}


__all__ = ["Recorder", "isolation", "isolation_counts", "generation", "generation_counts", "scramble", "cot_batch",
           "latent_path", "flags", "first_divergence", "permutation", "prompt_records", "TOL", "PathRecorder",
           "band_model", "generate_rows", "paths_check", "shapes_check", "swap_check", "band_logits", "BAND_NUMERICS",
           "BAND_DTYPES", "paths_b1_check", "DtypeWatch", "donor_answer", "donor_answer_rows", "donor_answer_summary",
           "donor_answer_outcome", "donor_in_class", "donor_boot", "step_results", "DONOR_CLASS_STEPS",
           "DONOR_COLUMNS", "DONOR_PASS_LINE", "DONOR_OUTCOMES", "DONOR_GARBAGE_LABELS"]
