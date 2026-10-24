"""Training-only auxiliary losses of the Phase 2 cards: the C2 state decoder and loss, the C3 latent-dependence hinge.
Nothing here is imported by model.py or eval.py: generation and evaluation never see these modules.

C2 (docs/cards/C2_verifiable_state_supervision.md "Mechanism"; registration A3.4 C2). `StateDecoder` g is a 2-layer
MLP with GELU and hidden 768 on the fed-back latent vector of step t plus a learned step-index embedding: layer 1
d -> 768 (+ step embedding), GELU, layer 2 768 -> d per head; the heads are read out through the language model's
embedding matrix, which the decoder receives **detached** (tied to the embedding matrix in value, with no gradient into
it, so the state loss reaches the language model only through the latent; the card's leak answer, tested):
  * ProsQA head: multi-label sigmoid over the record's symbols; symbol s scores <head(z), mean_k wte[tok_k(s)]> + b with
    tok(s) = encode(" " + name); binary cross-entropy over the unmasked symbol slots;
  * GSM heads: 4 parallel softmax heads over the vocabulary (logits = head_k(z) @ wte^T), cross-entropy per result
    token, masked beyond the result's length.
L_state = the mean over unmasked targets of the batch; total = CE + lambda L_state (lambda 0.1, registered).

C3 (docs/cards/C3_latent_dependence_loss.md "Mechanism"; registration A3.4 C3 with the critic's C3-foreign null).
Per example i with latents and answer tokens (labels != -100: everything after <|end-latent|>), CE_x(i) = the mean
cross-entropy per answer token under pass x. The bank of a scrambled pass is `bank[i, j] = latents[pi(i), j]` (own
pass's latents, detached) with pi a seeded derangement of the batch within latent-count classes (pi(i) != i, asserted).
  * C3:          hinge_i = max(0, m - (CE_scr(i) - CE_true(i)))
  * C3-foreign:  hinge_i = max(0, m - (CE_B(i) - CE_A(i))) with two banks from derangements A, B drawn with different
                 seeds and A(i) != B(i) for every hinged row (asserted); the answer loss is still CE_true.
Loss = CE_true + beta * mean_i hinge_i (beta 1, m 0.2 nat per token, registered). Rows in a class too small for the
derangement(s) (1 member; 2 for the foreign pair) carry no hinge. The training seeds come from
`train_derangement_seed`, which never equals the evaluation seed (`EVAL_SCRAMBLE_SEED`, latent.eval's default 0).
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent.state_targets import supervised_steps

EVAL_SCRAMBLE_SEED = 0              # latent.eval.evaluate(scramble_seed=0): the registered evaluation derangement
_TRAIN_SEED_BASE = 0xC3_5EED_0000   # training derangements live far from the evaluation seed


class StateDecoder(nn.Module):
    """C2 decoder g (see the module docstring). Saved separately from the language model (train.py)."""

    def __init__(self, d_model: int, hidden: int = 768, n_token_heads: int = 4, max_steps: int = 32):
        super().__init__()
        self.max_steps = max_steps
        self.inp = nn.Linear(d_model, hidden)
        self.step = nn.Embedding(max_steps, hidden)
        self.act = nn.GELU()
        self.sym_head = nn.Linear(hidden, d_model)
        self.sym_bias = nn.Parameter(torch.zeros(1))
        self.tok_heads = nn.ModuleList([nn.Linear(hidden, d_model) for _ in range(n_token_heads)])

    def trunk(self, z, step):
        return self.act(self.inp(z) + self.step(step.clamp(max=self.max_steps - 1)))

    def symbol_logits(self, z, step, sym_emb, sym_mask):
        """z [N, d], step [N] (0-based), sym_emb [N, S, d] (detached), sym_mask [N, S] -> logits [N, S]."""
        q = self.sym_head(self.trunk(z, step))
        lg = torch.einsum("nd,nsd->ns", q, sym_emb.to(q.dtype)) + self.sym_bias
        return lg.masked_fill(~sym_mask, 0.0)

    def token_logits(self, z, step, emb):
        """z [N, d], step [N], emb [V, d] (detached) -> logits [N, n_heads, V]."""
        h = self.trunk(z, step)
        return torch.stack([head(h) for head in self.tok_heads], 1) @ emb.to(h.dtype).t()

    def flops_per_target(self, kind: str, V: int, S: int) -> float:
        """Forward FLOPs of one decoded latent (2 x multiply-adds), for the training-FLOP log."""
        d, hid = self.inp.in_features, self.inp.out_features
        trunk = 2 * d * hid
        if kind == "prosqa":
            return float(trunk + 2 * hid * d + 2 * S * d)
        return float(trunk + len(self.tok_heads) * (2 * hid * d + 2 * V * d))


def _symbol_embeddings(tgts: list, S: int, emb: torch.Tensor):
    """For each record target, the mean of the (detached) embedding rows of each symbol's tokens, padded to S
    symbols -> [U, S, d] (one gather)."""
    K = max(max((len(t) for t in tg["symbol_tokens"]), default=1) for tg in tgts)
    ids = torch.zeros(len(tgts), S, max(1, K), dtype=torch.long)
    m = torch.zeros(len(tgts), S, max(1, K))
    for u, tg in enumerate(tgts):
        for s, toks in enumerate(tg["symbol_tokens"]):
            ids[u, s, :len(toks)] = torch.tensor(toks, dtype=torch.long)
            m[u, s, :len(toks)] = 1.0
    ids, m = ids.to(emb.device), m.to(emb.device, emb.dtype)
    return (emb[ids] * m.unsqueeze(-1)).sum(2) / m.sum(2, keepdim=True).clamp(min=1.0)


def state_loss(decoder: StateDecoder, latents, idx, targets: dict, *, c_thought: int, emb: torch.Tensor):
    """C2 L_state for one batch. latents: list per row of [n_latent, d] fed-back vectors (not detached: the loss
    trains the language model through them); idx: dataset idx per row; targets: idx -> latent.state_targets
    record targets (the donor's for C2-shuffled); emb: the embedding matrix, detached here. Returns
    (loss or None, n_supervised_latents, n_target_units)."""
    emb = emb.detach()
    items = []
    for r, (lat, i) in enumerate(zip(latents, idx)):
        tgt = targets[i]
        for t, j in supervised_steps(tgt["n_steps"], lat.shape[0], c_thought):
            if tgt["kind"] == "gsm" and tgt["tokens"][t - 1] is None:
                continue
            items.append((r, t, j, tgt))
    if not items:
        return None, 0, 0
    z = torch.stack([latents[r][j] for r, _, j, _ in items])
    step = torch.tensor([t - 1 for _, t, _, _ in items], device=z.device)
    if items[0][3]["kind"] == "prosqa":
        S = max(tg["n_symbols"] for *_, tg in items)
        uniq, where = [], {}
        for *_, tg in items:
            if id(tg) not in where:
                where[id(tg)] = len(uniq)
                uniq.append(tg)
        sym = _symbol_embeddings(uniq, S, emb)
        lab = torch.zeros(len(items), S, device=z.device)
        mask = torch.zeros(len(items), S, dtype=torch.bool, device=z.device)
        for n, (_, t, _, tg) in enumerate(items):
            mask[n, :tg["n_symbols"]] = True
            mask[n, tg["root"]] = False
            lab[n, tg["frontiers"][t - 1]] = 1.0
        rows = torch.tensor([where[id(tg)] for *_, tg in items], device=z.device)
        lg = decoder.symbol_logits(z, step, sym[rows], mask)
        bce = F.binary_cross_entropy_with_logits(lg.float(), lab, reduction="none")
        units = int(mask.sum())
        return (bce * mask).sum() / max(1, units), len(items), units
    H = len(decoder.tok_heads)
    lab = torch.full((len(items), H), -100, dtype=torch.long, device=z.device)
    for n, (_, t, _, tg) in enumerate(items):
        toks = tg["tokens"][t - 1]
        lab[n, :len(toks)] = torch.tensor(toks[:H], device=z.device)
    lg = decoder.token_logits(z, step, emb)
    units = int((lab != -100).sum())
    return F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100), len(items), units


# ----------------------------------------------------------------------------------------------------------- C3
def train_derangement_seed(seed: int, epoch: int, step: int, stream: int = 0) -> int:
    """C3: per-batch derangement seed, reseeded per epoch (and per batch), stream 0 = the scrambled bank (bank A of
    C3-foreign), 1 = bank B. Never equals EVAL_SCRAMBLE_SEED."""
    s = _TRAIN_SEED_BASE + ((int(seed) * 1_000_003 + int(epoch)) * 1_000_033 + int(step)) * 4 + int(stream)
    assert s != EVAL_SCRAMBLE_SEED
    return s


def _sattolo(n, rng):
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        p[i], p[j] = p[j], p[i]
    return p


def batch_derangement(counts, seed: int, avoid=None):
    """pi over batch rows: within each latent-count class (count > 0) a seeded derangement (Sattolo; pi(i) != i).
    With `avoid` (another pi) also pi(i) != avoid(i), redrawn up to 100 times (C3-foreign). Returns (pi, valid) with
    valid[i] False for rows without latents or in a class too small (1 member; 2 with `avoid`)."""
    rng = random.Random(seed)
    B = len(counts)
    pi, valid = list(range(B)), [False] * B
    classes = {}
    for i, c in enumerate(counts):
        if c > 0:
            classes.setdefault(int(c), []).append(i)
    for c in sorted(classes):
        mem = classes[c]
        if len(mem) < (3 if avoid is not None else 2):
            continue
        for _ in range(100):
            perm = _sattolo(len(mem), rng)
            if avoid is None or all(mem[perm[k]] != avoid[mem[k]] for k in range(len(mem))):
                break
        else:
            continue
        for k, i in enumerate(mem):
            pi[i], valid[i] = mem[perm[k]], True
    assert all(pi[i] != i for i in range(B) if valid[i])
    if avoid is not None:
        assert all(pi[i] != avoid[i] for i in range(B) if valid[i])
    return pi, valid


def build_bank(latents, pi, valid) -> torch.Tensor:
    """bank[i, j] = latents[pi(i)][j], detached; rows without a valid donor keep their own (detached) latents so the
    pass is defined (they carry no hinge)."""
    n_max = max(1, max(x.shape[0] for x in latents))
    ref = next(x for x in latents if x.shape[0] > 0) if any(x.shape[0] for x in latents) else latents[0]
    bank = ref.new_zeros(len(latents), n_max, ref.shape[-1])
    for i in range(len(latents)):
        src = latents[pi[i]] if valid[i] else latents[i]
        if src.shape[0]:
            bank[i, :src.shape[0]] = src.detach()
    return bank.detach()


def per_example_ce(logits, labels):
    """Per-row mean cross-entropy over answer tokens (labels != -100, shifted as the model's loss) -> (ce [B], n [B])."""
    lg = logits[:, :-1].float()
    lab = labels[:, 1:]
    tok = F.cross_entropy(lg.transpose(1, 2), lab, ignore_index=-100, reduction="none")
    m = (lab != -100)
    n = m.sum(1)
    return (tok * m).sum(1) / n.clamp(min=1), n


def hinge(ce_ref, ce_alt, valid, margin: float):
    """C3 hinge, mean over valid rows of max(0, margin - (ce_alt - ce_ref)); None when no row is valid."""
    if not bool(valid.any()):
        return None
    h = torch.relu(margin - (ce_alt - ce_ref))
    return h[valid].mean()


__all__ = ["StateDecoder", "state_loss", "train_derangement_seed", "batch_derangement", "build_bank",
           "per_example_ce", "hinge", "EVAL_SCRAMBLE_SEED"]
