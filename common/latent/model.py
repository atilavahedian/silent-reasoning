"""LatentLM: a vectorized port of facebookresearch/coconut (MIT) `coconut.py` (class Coconut).

Modes of LatentLM.forward / LatentLM.generate:
  * "coconut": Coconut.forward. The last-layer hidden state (after the final LayerNorm, as `hidden_states[-1]` in the
    reference) at position j-1 becomes the input embedding at latent position j. One sequential pass per latent rank,
    the KV cache of the previous passes is reused (sliced to the compute range as the reference does), the per-pass
    logits are concatenated, and the loss is CE on shifted labels. The fed-back hidden state is NOT detached (as in the
    reference). The embedding tensor is rebuilt with an out-of-place `index_put` on the latent positions instead of
    the reference's nested Python lists of 1-d tensors; values and gradients are identical (tests/test_model.py).
  * "pause": one pass; <|latent|> positions keep their learned embedding (the reference's coconut=False path for a
    run with latent tokens, i.e. a plain causal LM over the same token sequence).
  * "scrambled": the vector at latent rank j of example i is `scramble_bank[i, j]` (built by eval.py from other
    examples). It never depends on example i's hidden states, so the whole sequence runs in one pass, which is
    numerically the same computation as the multi-pass loop with fixed inputs.

The reference requires (and the reference collator guarantees) that the latents of a batch are contiguous and that
the first latent sits at the same column in every example that has one; LatentLM raises ValueError otherwise, where
the reference would index out of range or silently feed the wrong state.

C1b (docs/cards/C1b_partial_depth_latents.md, "Mechanism" and "Implementation notes"; registration A3.4):
`LatentLM(..., latent_band=(a, b))` runs only blocks [a, b) at latent positions. The input to block a at latent
position j is h_b(j-1), the residual stream after block b-1 at the previous position (HF's hidden_states[b]; for b = L
that is the post-ln_f state the full-depth path feeds back), so consecutive latents recur through the band. With a = 0
the fed-back vector takes the token embedding's place (position embedding and embedding dropout applied, as in mode
"coconut"), so latent_band=(0, L) is the full-depth Coconut forward (a test). Blocks outside the band are **skipped** at
latent positions (A3.4: no computation, not a mask over computed junk); their key/value caches hold zeros there and a
per-block 4D boolean mask removes those columns for every query, so later tokens see the latents only inside the band.
Position ids are unchanged. The band path is a manual GPT-2 forward over the block's own submodules (ln_1, c_attn,
scaled_dot_product_attention, c_proj, ln_2, mlp; dropout as HF in train mode) with a per-block KV list; it supports
the GPT-2 family only. A batch without latents, and latent_band=None, use the unchanged HF path above (bit-identical).
Logits at latent positions are zeros in the band path (no LM head there; their labels are -100 in every arm).
Per-pass wall-clock (synchronised) is recorded when `self.timing` is a list (generate(time_steps=True)).

KV cache: transformers 4.x returns either the legacy tuple of (k, v) per layer (older 4.x) or a `Cache` object
(`DynamicCache`, 4.36+ and the only form GPT-2 returns from about 4.50; the Mac lock 4.57.6 converts tuples and returns
DynamicCache). `_crop_cache` handles both.
"""
from __future__ import annotations

from collections import namedtuple

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn import CrossEntropyLoss

Outputs = namedtuple("Outputs", ["loss", "logits", "inputs_embeds", "latents"])
GenOutputs = namedtuple("GenOutputs", ["tokens", "emitted_tokens", "latent_steps", "sequential_passes", "latents",
                                       "timing", "capped"], defaults=(None, None))
_RunOut = namedtuple("_RunOut", ["logits", "inputs_embeds", "latents", "cache", "n_latent"])

MODES = ("coconut", "pause", "scrambled")
TRANSFORMERS_VERSION = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())


def _crop_cache(cache, n: int):
    """Mirrors the reference's `[(k[:, :, :n, :], v[:, :, :n, :]) for k, v in kv_cache]` for both cache formats:
    a legacy tuple is sliced into a new tuple; a `Cache` object (DynamicCache) is cropped in place when longer."""
    if cache is None:
        return None
    if hasattr(cache, "get_seq_length") and hasattr(cache, "crop"):
        if cache.get_seq_length() > n:
            cache.crop(n)
        return cache
    return tuple((k[:, :, :n, :], v[:, :, :n, :]) for k, v in cache)


class LatentLM(nn.Module):
    """Mirrors coconut.py Coconut (same constructor arguments; the wrapped model is `self.base_causallm`)."""

    def __init__(self, base_causallm, latent_id, start_id, end_id, eos_id, latent_band=None):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_id = int(latent_id)
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.eos_id = int(eos_id)
        self.latent_band = None if latent_band is None else _check_band(base_causallm, latent_band)
        self.timing = None        # a list collects (kind, n_columns, seconds) per pass; generate(time_steps=True)
        self.band_debug = None    # a list collects ("mask", block, in_band, mask) / ("block", block, kind, rows)
        # Review fix (docs/review_phase2_code.md): nn.Module starts in train mode whatever the wrapped model's mode,
        # so a wrapper over an .eval() base reported training=True, the band path's attention dropout followed that
        # flag, and evaluate() / probe.patch restored "train" to the base afterwards. The wrapper inherits the mode.
        self.train(base_causallm.training)

    @property
    def layer_fraction(self) -> float:
        """C1b "Math and accounting": (b - a) / L, the share of blocks a latent step runs; 1.0 without a band."""
        if self.latent_band is None:
            return 1.0
        a, b = self.latent_band
        return (b - a) / len(self.backbone.h)

    # Accessors instead of attributes so state_dict() holds only base_causallm.* (the reference also registered an
    # `embedding` alias; loaders drop it, see train.load_state).
    @property
    def embedding(self):
        """Mirrors Coconut.embedding (the wrapped model's input embeddings)."""
        return self.base_causallm.get_input_embeddings()

    @property
    def backbone(self):
        """Ours: the transformer body; with lm_head it equals the causal LM's forward (hidden_states[-1], logits)."""
        return self.base_causallm.base_model  # GPT2Model for GPT2LMHeadModel, Qwen3Model for Qwen3ForCausalLM

    @property
    def lm_head(self):
        """Ours: the output head (tied to the input embeddings in GPT-2)."""
        return self.base_causallm.get_output_embeddings()

    def _pass(self, embeds, attention_mask, position_ids, cache, use_cache):
        out = self.backbone(inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_values=cache, use_cache=use_cache)
        h = out.last_hidden_state  # == outputs.hidden_states[-1] of the causal LM (post final norm)
        return h, self.lm_head(h), (out.past_key_values if use_cache else None)

    def _latent_layout(self, input_ids):
        lat = input_ids == self.latent_id
        n_lat = lat.sum(1)
        max_n = int(n_lat.max().item()) if lat.numel() else 0
        if max_n == 0:
            return lat, n_lat, 0, 0, None
        rank = lat.long().cumsum(1) - 1
        first = int(lat.any(0).nonzero()[0].item())
        cols = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        if not bool(((cols - first == rank) | ~lat).all().item()):
            raise ValueError("latent tokens must be contiguous and start at the same column in every example that has "
                             "them (the reference MyCollator left-pads to guarantee this)")
        if first == 0:
            raise ValueError("a latent token at column 0 has no preceding hidden state")
        return lat, n_lat, max_n, first, rank

    def _run(self, input_ids, attention_mask, position_ids, mode, scramble_bank, keep_cache) -> _RunOut:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        L = input_ids.shape[1]
        embeds = self.embedding(input_ids)
        lat, n_lat, max_n, first, rank = self._latent_layout(input_ids)
        if self.latent_band is not None and max_n > 0:
            if mode == "pause":
                raise ValueError("latent_band applies to the latent-feedback modes (coconut, scrambled), not pause")
            return self._run_band(embeds, attention_mask, position_ids, mode, scramble_bank, lat, n_lat, max_n,
                                  first, rank)

        if mode == "scrambled" and max_n > 0:
            if scramble_bank is None or scramble_bank.shape[0] != input_ids.shape[0] or scramble_bank.shape[1] < max_n:
                raise ValueError("mode 'scrambled' needs scramble_bank of shape [batch, >= max latents, hidden]")
            bi, pi = lat.nonzero(as_tuple=True)
            vals = scramble_bank.to(device=embeds.device, dtype=embeds.dtype)[bi, rank[bi, pi]]
            embeds = embeds.index_put((bi, pi), vals)

        if mode in ("pause", "scrambled") or max_n == 0:
            t0 = self._tick()
            _, logits, cache = self._pass(embeds, attention_mask, position_ids, None, keep_cache)
            self._tock("prefill", L, t0)
            return _RunOut(logits, embeds, self._collect(embeds, n_lat, first, max_n), cache, n_lat)

        # mode == "coconut": Coconut.forward with ranges (0, first), (first, first+1), ..., (first+max_n-1, L)
        logits, cache = [], None
        r0, r1 = 0, first
        for p in range(max_n):
            t0 = self._tick()
            h, lg, cache = self._pass(embeds[:, r0:r1], attention_mask[:, :r1], position_ids[:, r0:r1],
                                      _crop_cache(cache, r0), True)
            self._tock("prefill" if p == 0 else "latent", r1 - r0, t0)
            logits.append(lg)
            pos = first + p  # every example with > p latents has its rank-p latent here
            bi = (n_lat > p).nonzero().squeeze(1)
            pi = torch.full_like(bi, pos)
            embeds = embeds.index_put((bi, pi), h[bi, pos - 1 - r0].to(embeds.dtype))
            r0, r1 = r1, (L if p + 1 >= max_n else r1 + 1)
        t0 = self._tick()
        _, lg, cache = self._pass(embeds[:, r0:r1], attention_mask[:, :r1], position_ids[:, r0:r1],
                                  _crop_cache(cache, r0), keep_cache)
        self._tock("latent_last", r1 - r0, t0)
        logits.append(lg)
        return _RunOut(torch.cat(logits, dim=-2), embeds, self._collect(embeds, n_lat, first, max_n), cache, n_lat)

    # ------------------------------------------------------------------------------------------------ timing
    def _tick(self):
        if self.timing is None:
            return None
        _sync(self.lm_head.weight.device)
        return time.perf_counter()

    def _tock(self, kind, n_cols, t0):
        if t0 is not None:
            _sync(self.lm_head.weight.device)
            self.timing.append((kind, int(n_cols), time.perf_counter() - t0))

    # ------------------------------------------------------------------------- C1b band path (GPT-2 family)
    def _band_masks(self, state, r0, r1):
        """Per-block 4D boolean masks for queries [r0, r1) over keys [0, r1): causal, attended, and (outside the
        band) not a latent column. A query row with no admissible key (a pad) attends to itself only."""
        cols = torch.arange(r0, r1, device=state["ok_in"].device)
        causal = torch.arange(r1, device=cols.device).unsqueeze(0) <= cols.unsqueeze(1)
        out = {}
        for in_band, ok in ((True, state["ok_in"]), (False, state["ok_out"])):
            m = causal.unsqueeze(0) & ok[:, :r1].unsqueeze(1)
            empty = ~m.any(-1)
            if bool(empty.any()):
                diag = torch.zeros_like(m)
                diag[:, torch.arange(r1 - r0), cols] = True
                m = m | (diag & empty.unsqueeze(-1))
            out[in_band] = m.unsqueeze(1)
        return out

    def _block(self, i, x, mask, kv):
        """One GPT-2 block (GPT2Block: ln_1, attention, residual, ln_2, mlp, residual) on hidden x [B, n, d] with a
        cached (k, v) of earlier columns and a boolean mask [B, 1, n, S]. Returns (x, (k, v)) with the new columns."""
        blk = self.backbone.h[i]
        B, n, d = x.shape
        H = self.base_causallm.config.n_head
        q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(d, dim=2)
        q, k, v = (t.view(B, n, H, d // H).transpose(1, 2) for t in (q, k, v))
        if kv is not None:
            k, v = torch.cat([kv[0].to(k.dtype), k], 2), torch.cat([kv[1].to(v.dtype), v], 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=blk.attn.attn_dropout.p if self.training else 0.0)
        x = x + blk.attn.resid_dropout(blk.attn.c_proj(o.transpose(1, 2).reshape(B, n, d)))
        x = x + blk.mlp(blk.ln_2(x))
        if self.band_debug is not None:
            self.band_debug.append(("mask", i, None, mask.detach()))
        return x, (k, v)

    def _band_full(self, state, embeds, position_ids, r0, r1):
        """Every block over the non-latent columns [r0, r1) (prefix, the tail after the latents, decode steps).
        Returns (h_final post ln_f, h_b) for the range."""
        a, b = self.latent_band
        bb = self.backbone
        x = bb.drop(embeds[:, r0:r1] + bb.wpe(position_ids[:, r0:r1]))
        masks = self._band_masks(state, r0, r1)
        hb = None
        for i in range(len(bb.h)):
            x, state["kv"][i] = self._block(i, x, masks[a <= i < b], state["kv"][i])
            if self.band_debug is not None:
                self.band_debug.append(("block", i, "full", x.shape[0]))
            if i == b - 1:
                hb = x
        hf = bb.ln_f(x)
        return hf, (hf if b == len(bb.h) else hb)

    def _band_column(self, state, embeds, position_ids, j, latcol, vals):
        """Column j: rows with a latent there run blocks [a, b) only, from the injected vector `vals` (at block a's
        input; with a = 0 it replaces the token embedding); other rows run every block. Non-band blocks append a
        zero key/value column for latent rows (never attended). Returns (logits [B, 1, V], h_b [B, d])."""
        a, b = self.latent_band
        bb = self.backbone
        n_layer = len(bb.h)
        B = embeds.shape[0]
        R = latcol.nonzero().squeeze(1)
        N = (~latcol).nonzero().squeeze(1)
        masks = self._band_masks(state, j, j + 1)
        pos = position_ids[:, j:j + 1]
        inj = vals.to(embeds.dtype).unsqueeze(1)
        e = embeds[:, j:j + 1]
        if a == 0:
            e = e.index_put((R,), inj[R])
        x = bb.drop(e + bb.wpe(pos))                    # rows N use it for blocks < a; for a == 0 every row does
        if a > 0:
            xN = x[N]
        for i in range(n_layer):
            in_band = a <= i < b
            if in_band:
                if i == a and a > 0:
                    full = x.new_zeros(B, 1, x.shape[-1]).index_put((N,), xN.to(x.dtype))
                    x = full.index_put((R,), inj[R].to(full.dtype))
                x, state["kv"][i] = self._block(i, x, masks[True], state["kv"][i])
                if self.band_debug is not None:
                    self.band_debug.append(("block", i, "column", B))
                if i == b - 1:
                    hb_all = x
                    xN = x[N]
                continue
            kc, vc = state["kv"][i]
            ck, cv = kc.new_zeros(kc.shape[:2] + (1, kc.shape[-1])), vc.new_zeros(vc.shape[:2] + (1, vc.shape[-1]))
            if N.numel():
                xN, (kN, vN) = self._block(i, xN, masks[False][N], (kc[N], vc[N]))
                if self.band_debug is not None:
                    self.band_debug.append(("block", i, "column", int(N.numel())))
                ck = ck.index_put((N,), kN[:, :, -1:].to(ck.dtype))
                cv = cv.index_put((N,), vN[:, :, -1:].to(cv.dtype))
            state["kv"][i] = (torch.cat([kc, ck], 2), torch.cat([vc, cv], 2))
        logits = hb_all.new_zeros(B, 1, self.lm_head.weight.shape[0])
        if N.numel():
            hfN = bb.ln_f(xN)
            lgN = self.lm_head(hfN)
            logits = logits.to(lgN.dtype).index_put((N,), lgN)
        hb = bb.ln_f(hb_all) if b == n_layer else hb_all
        if N.numel() and b == n_layer:
            hb = hb.index_put((N,), hfN)
        return logits, hb[:, 0]

    def _run_band(self, embeds, attention_mask, position_ids, mode, scramble_bank, lat, n_lat, max_n, first, rank):
        """C1b forward: prefix [0, first) through every block, one column pass per latent rank (band only for latent
        rows), then the tail through every block. mode "scrambled" injects scramble_bank[i, rank] instead of the
        band output of the previous position (the k = 8 scrambled diagnostic, A3.4)."""
        if mode == "scrambled":
            if scramble_bank is None or scramble_bank.shape[0] != embeds.shape[0] or scramble_bank.shape[1] < max_n:
                raise ValueError("mode 'scrambled' needs scramble_bank of shape [batch, >= max latents, hidden]")
            bank = scramble_bank.to(device=embeds.device, dtype=embeds.dtype)
        L = embeds.shape[1]
        am = attention_mask.bool()
        state = {"kv": [None] * len(self.backbone.h), "ok_in": am, "ok_out": am & ~lat}
        t0 = self._tick()
        hf, hb = self._band_full(state, embeds, position_ids, 0, first)
        self._tock("prefill", first, t0)
        logits = [self.lm_head(hf)]
        prev = hb[:, first - 1]
        for p in range(max_n):
            j = first + p
            latcol = lat[:, j]
            vals = bank[:, p] if mode == "scrambled" else prev
            embeds = embeds.index_put((latcol.nonzero().squeeze(1), torch.full_like(latcol.nonzero().squeeze(1), j)),
                                      vals[latcol].to(embeds.dtype))
            t0 = self._tick()
            lg, prev = self._band_column(state, embeds, position_ids, j, latcol, vals)
            self._tock("latent", 1, t0)
            logits.append(lg.to(logits[0].dtype))
        if first + max_n < L:
            t0 = self._tick()
            hf, _ = self._band_full(state, embeds, position_ids, first + max_n, L)
            self._tock("latent_last", L - first - max_n, t0)
            logits.append(self.lm_head(hf))
        return _RunOut(torch.cat(logits, dim=-2), embeds, self._collect(embeds, n_lat, first, max_n), state, n_lat)

    def _band_decode(self, state, tok, pos):
        """One decode step on the band cache: the new token runs every block; outside the band it cannot see
        latent columns (state masks)."""
        ones = torch.ones(tok.shape[0], 1, dtype=torch.bool, device=tok.device)
        state["ok_in"] = torch.cat([state["ok_in"], ones], 1)
        state["ok_out"] = torch.cat([state["ok_out"], ones], 1)
        S = state["ok_in"].shape[1]
        embeds = self.embedding(tok).unsqueeze(1)
        bb = self.backbone
        x = bb.drop(embeds + bb.wpe(pos.unsqueeze(1)))
        a, b = self.latent_band
        for i in range(len(bb.h)):
            ok = state["ok_in"] if a <= i < b else state["ok_out"]
            x, state["kv"][i] = self._block(i, x, ok[:, None, None, :S], state["kv"][i])
        return self.lm_head(bb.ln_f(x)), state

    @staticmethod
    def _collect(embeds, n_lat, first, max_n):
        if max_n == 0:
            return [embeds[i, :0] for i in range(embeds.shape[0])]
        return [embeds[i, first:first + int(n)] for i, n in enumerate(n_lat.tolist())]

    def forward(self, input_ids, attention_mask, labels, position_ids, *, mode: str = "coconut",
                scramble_bank: torch.Tensor | None = None, return_latents: bool = False) -> Outputs:
        """Mirrors Coconut.forward (mode "coconut"); "pause" mirrors the plain causal-LM forward the reference uses
        when coconut=False; "scrambled" is ours. Loss: CrossEntropyLoss on logits[..., :-1] vs labels[..., 1:]."""
        run = self._run(input_ids, attention_mask, position_ids, mode, scramble_bank, keep_cache=False)
        loss = None
        if labels is not None:
            shift_logits = run.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return Outputs(loss=loss, logits=run.logits, inputs_embeds=run.inputs_embeds,
                       latents=run.latents if return_latents else None)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, position_ids, *, max_new_tokens: int, mode: str = "coconut",
                 scramble_bank=None, return_latents: bool = False, time_steps: bool = False) -> GenOutputs:
        """Mirrors Coconut.generate (greedy; first token from the forward's last logits, then one pass per token,
        stop at eos) batched over right-padded prompts with a KV cache instead of the reference's batch-size-1 full
        recomputation. The reference appends the first token even when it is eos and keeps going; here eos stops an
        example at any step (see report). tokens excludes the prompt and the eos. time_steps=True synchronises the
        device around every pass and returns GenOutputs.timing (summarize_timing: wall-clock per latent step and per
        emitted-token step, C1b A3.4); False (the default) adds nothing to the computation."""
        B, L = input_ids.shape
        dev = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        if time_steps:
            self.timing = []
        try:
            return self._generate(input_ids, attention_mask, position_ids, max_new_tokens, mode, scramble_bank,
                                  return_latents)
        finally:
            self.timing = None

    def _generate(self, input_ids, attention_mask, position_ids, max_new_tokens, mode, scramble_bank,
                  return_latents):
        B, L = input_ids.shape
        dev = input_ids.device
        run = self._run(input_ids, attention_mask, position_ids, mode, scramble_bank, keep_cache=True)
        cache = run.cache
        band = isinstance(cache, dict)
        am = attention_mask.to(torch.long)
        cols = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        last = torch.where(am.bool(), cols, torch.full_like(cols, -1)).max(1).values
        if bool((last < 0).any().item()):
            raise ValueError("every prompt needs at least one attended token")
        rows = torch.arange(B, device=dev)
        tok = run.logits[rows, last].argmax(-1)
        pos = position_ids[rows, last] + 1
        out_tokens = [[] for _ in range(B)]
        done = [False] * B
        capped = [False] * B
        cfg = self.base_causallm.config
        n_pos = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None)
        for t in range(max_new_tokens):
            for i, x in enumerate(tok.tolist()):
                if not done[i]:
                    if x == self.eos_id:
                        done[i] = True
                    else:
                        out_tokens[i].append(x)
            if all(done) or t == max_new_tokens - 1:
                break
            if n_pos is not None and bool((pos >= n_pos).any()):
                # Ours: a row whose next token would sit past the model's last position stops there (the reference
                # would fail on the position index); finished rows are fed a clamped position, their output unused.
                for i, over in enumerate((pos >= n_pos).tolist()):
                    if over and not done[i]:
                        done[i] = capped[i] = True
                if all(done):
                    break
                pos = pos.clamp(max=n_pos - 1)
            t0 = self._tick()
            if band:
                lg, cache = self._band_decode(cache, tok, pos)
            else:
                am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device=dev)], dim=1)
                _, lg, cache = self._pass(self.embedding(tok).unsqueeze(1), am, pos.unsqueeze(1), cache, True)
            self._tock("token", 1, t0)
            tok = lg[:, -1].argmax(-1)
            pos = pos + 1
        steps = run.n_latent.tolist() if mode in ("coconut", "scrambled") else [0] * B
        emitted = [len(t) for t in out_tokens]
        lat = [x.detach() for x in run.latents] if return_latents else None
        return GenOutputs(tokens=out_tokens, emitted_tokens=emitted, latent_steps=steps,
                          sequential_passes=[1 + s + e for s, e in zip(steps, emitted)], latents=lat,
                          timing=summarize_timing(self.timing) if self.timing is not None else None, capped=capped)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def summarize_timing(timing) -> dict:
    """Ours (C1b A3.4: the layer fraction must be visible in time). Per-pass records (kind, columns, seconds) of one
    generate call -> totals and means in ms: prefill, latent single-column passes ("latent"; the coconut path's last
    pass also carries <|end-latent|> and is "latent_last"), and emitted-token passes ("token")."""
    out = {}
    for kind in ("prefill", "latent", "latent_last", "token"):
        xs = [s for k, _, s in timing if k == kind]
        out[f"{kind}_passes"] = len(xs)
        out[f"{kind}_ms_total"] = 1000.0 * sum(xs)
        out[f"{kind}_ms_mean"] = (1000.0 * sum(xs) / len(xs)) if xs else None
    return out


def _check_band(model, band):
    """C1b: validates latent_band = (a, b) with 0 <= a < b <= n_layer on a GPT-2-family model whose attention the
    manual band forward reproduces (scaled by 1/sqrt(head_dim), no layer-index scaling, no upcast reordering)."""
    a, b = (int(x) for x in band)
    bb = model.base_model
    if not all(hasattr(bb, k) for k in ("h", "wte", "wpe", "ln_f", "drop")):
        raise NotImplementedError("latent_band is implemented for the GPT-2 family only")
    cfg = model.config
    if (getattr(cfg, "scale_attn_by_inverse_layer_idx", False) or getattr(cfg, "reorder_and_upcast_attn", False)
            or not getattr(cfg, "scale_attn_weights", True) or getattr(cfg, "add_cross_attention", False)):
        raise NotImplementedError("latent_band needs standard GPT-2 attention")
    if not 0 <= a < b <= len(bb.h):
        raise ValueError(f"latent_band {band} must satisfy 0 <= a < b <= n_layer = {len(bb.h)}")
    return (a, b)


def init_latent_embeddings(model, tokenizer, ids, init_token: str = "<<"):
    """Mirrors run.py: resize_token_embeddings(len(tokenizer)) and copy the embedding (and lm_head row, a no-op when
    tied) of the token "<<" into <|latent|>, <|start-latent|>, <|end-latent|>. Falls back to eos when the vocabulary
    has no "<<" token. Never shrinks a padded vocabulary (Qwen3)."""
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    try:
        target = tokenizer.convert_tokens_to_ids(init_token)
    except KeyError:  # tokenizers without the token and without an unk id
        target = None
    unk = getattr(tokenizer, "unk_token_id", None)
    if target is None or (unk is not None and target == unk) or target >= len(tokenizer):
        target = tokenizer.eos_token_id
    emb = model.get_input_embeddings().weight.data
    head = model.get_output_embeddings()
    for t in (ids["latent_id"], ids["start_id"], ids["end_id"]):
        emb[t] = emb[target]
        if head is not None and head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
            head.weight.data[t] = head.weight.data[target]
    return target


def load_base(model_id_or_path, dtype=torch.float32, device="cpu", *, resize: bool = True):
    """Mirrors run.py model/tokenizer setup: AutoModelForCausalLM / AutoTokenizer.from_pretrained, pad = eos, add the
    three latent tokens in order, and (resize=True, the reference's non-CoT arms) resize embeddings and initialise the
    new rows from "<<". The reference does not resize for cot / no_cot; pass resize=False there. Returns
    (model, tokenizer); the latent ids come from tokenizer.convert_tokens_to_ids."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from latent.data import add_latent_tokens

    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    ids = add_latent_tokens(tokenizer)
    if resize:
        init_latent_embeddings(model, tokenizer, ids)
    return model.to(device), tokenizer


__all__ = ["LatentLM", "Outputs", "GenOutputs", "load_base", "init_latent_embeddings", "MODES", "summarize_timing"]
