"""Compute accounting for latent-reasoning runs: FLOPs, sequential passes, emitted tokens.

Registered convention: think-bet registration section 6 as implemented in
experiments/2026-10-12-think-bet/code/think/lm.py (flops_per_token, gen_flops), extended to GPT-2. Notation:
N_ne = non-embedding parameters, V = vocab rows of the output head, d = hidden size, n_L = layers,
d_a = n_head * head_dim (attention width).

* N_ne is summed from the model's parameter tensors (not from a closed form): every parameter except the input
  token embedding (wte), the learned position embedding (wpe) and the output head. GPT-2 ties its head to wte, so
  the head is charged only through the V d term; an untied head is charged the same way and is likewise excluded
  from N_ne. Biases and LayerNorms are inside N_ne (think/lm.py's closed form omitted them and assumed a 3-matrix
  SwiGLU MLP; GPT-2 has a 2-matrix MLP with n_inner = 4d, which the parameter count picks up automatically).
* Training, per sequence of T positions: 3 x [2 (N_ne + V d) T + 2 n_L d_a T^2] (forward x 3; the Coconut
  multi-pass forward reuses the KV cache, so its FLOPs equal one full pass up to the recomputed attention, which is
  not counted, as in think/lm.py).
* Prefill of p positions: 2 N_ne p + 2 V d head_positions + 2 n_L d_a p^2.
* n sequential single-position passes after a prefix of p: 2 N_ne n + 2 V d head_steps + 2 n_L d_a [(p+n)^2 - p^2].
  A latent step needs no logits (head 0); an emitted token needs one head row.

inference_record follows the binding position table of the registration (section 5.3; INTERFACE.md v2), with
|q| = question tokens including the trailing newline and E' = emitted tokens before eos (the eos pass is inside the
E' head steps, because the pass at the last emitted position produces it):

| arm | prefill p (head positions) | steps after prefill n (head steps) | latent_steps | sequential passes |
|---|---|---|---|---|
| cot, no_cot | |q| (1) | E' (E') | 0 | 1 + E' |
| coconut, coconut_no_curriculum, scrambled (L latents; L = 0 is zero-latent) | |q| + 1 (0) | L + 1 + E' (1 + E') | L | 1 + L + E' |
| pause (L pause tokens) | |q| + L + 2 (1) | E' (E') | 0 | 1 + E' |

So the head is evaluated E' + 1 times: once at the last prompt position (the first answer token; for coconut that is
the <|end-latent|> step) and once per emitted token (the last one yields eos); latent steps evaluate no head.
Registration worked example (GPT-2 124M dims, V = 50,260; coconut, |q| = 250, L = 6, E' = 8): 47,248,760,832 FLOPs,
15 sequential passes (a test).

C1b (docs/cards/C1b_partial_depth_latents.md, "Math and accounting"; registration A3.4): `layer_fraction` f = (b - a)/L
(default 1.0, which leaves every formula and every result unchanged). A latent step runs only the band, so
flops_steps(..., layer_fraction=f) charges 2 N_ne f per step and the attention term 2 n_L d_a f [(p+n)^2 - p^2] (the
band's layers only); inference_record applies it to the L latent steps of the coconut row and charges the <end> and
emitted steps as before (their attention over the latent columns is charged at every layer although the non-band
layers mask them: the conservative side). Sequential passes are unchanged. Training (flops_train_per_sequence with
latent_positions and layer_fraction) scales the 2 N_ne term at latent positions only; the head and the T^2 attention
term are charged in full (again the conservative side). f is rationalised with limit_denominator(1000), so k/L band
fractions are charged exactly (the worked example at 8/12 and 4/12 is a test).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

import torch


@dataclass(frozen=True)
class Dims:
    N_ne: int   # non-embedding parameters (all but wte, wpe and the output head)
    V: int      # rows of the output head (vocab size after adding special tokens)
    d: int      # hidden size
    n_L: int    # transformer blocks
    d_a: int    # n_head * head_dim


_Q_PATTERNS = (
    (re.compile(r"(^|\.)h\.(\d+)\.attn\.c_attn\.weight$"), "gpt2"),          # Conv1D [d, 3 d_a]
    (re.compile(r"(^|\.)layers\.(\d+)\.self_attn\.q_proj\.weight$"), "linear"),  # Linear [d_a, d] (Qwen/Llama)
)


def _unwrap(m):
    while hasattr(m, "base_causallm"):  # LatentLM / Coconut wrappers
        m = m.base_causallm
    return m


def _as_model(config_or_model):
    if isinstance(config_or_model, torch.nn.Module):
        return _unwrap(config_or_model)
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):  # shapes only, no memory, no download
        return AutoModelForCausalLM.from_config(config_or_model)


def model_dims(config_or_model) -> Dims:
    """Dims from parameter shapes (mirrors think/lm.py flops_per_token's quantities, measured instead of derived).
    Accepts an HF causal-LM, a wrapper exposing .base_causallm (LatentLM, Coconut), or an HF config (instantiated on
    the meta device). V is the head's row count, so call it after the special tokens are added and embeddings
    resized if the run uses them."""
    model = _as_model(config_or_model)
    emb_in = model.get_input_embeddings()
    emb_out = model.get_output_embeddings()
    excluded = {id(emb_in.weight)}
    if emb_out is not None:
        excluded.add(id(emb_out.weight))
        if getattr(emb_out, "bias", None) is not None:
            excluded.add(id(emb_out.bias))
    for mod in model.modules():  # position embeddings (GPT-2 wpe) and any other lookup table
        if isinstance(mod, torch.nn.Embedding):
            excluded.add(id(mod.weight))
    n_ne = sum(p.numel() for p in model.parameters() if id(p) not in excluded)  # parameters() dedups tied weights
    head_w = emb_out.weight if emb_out is not None else emb_in.weight
    V, d = int(head_w.shape[0]), int(head_w.shape[1])

    layers, d_a = set(), None
    for name, p in model.named_parameters():
        for pat, kind in _Q_PATTERNS:
            m = pat.search(name)
            if m:
                layers.add(int(m.group(2)))
                width = p.shape[1] // 3 if kind == "gpt2" else p.shape[0]
                d_a = int(width) if d_a is None else d_a
    cfg = model.config
    if not layers:  # unknown family: fall back to the config
        n_L = int(getattr(cfg, "n_layer", None) or cfg.num_hidden_layers)
        n_head = int(getattr(cfg, "n_head", None) or cfg.num_attention_heads)
        d_a = n_head * int(getattr(cfg, "head_dim", None) or d // n_head)
    else:
        n_L = len(layers)
    return Dims(N_ne=int(n_ne), V=V, d=d, n_L=n_L, d_a=int(d_a))


def _frac(layer_fraction) -> Fraction:
    f = Fraction(layer_fraction).limit_denominator(1000)
    if not 0 < f <= 1:
        raise ValueError(f"layer_fraction must be in (0, 1], got {layer_fraction}")
    return f


def flops_train_per_sequence(dims: Dims, T: int, latent_positions: int = 0, layer_fraction: float = 1.0) -> float:
    """think/lm.py flops_per_token(...)["train_per_token"] summed over a sequence:
    3 * [2 (N_ne + V d) T + 2 n_L d_a T^2]. C1b: with layer_fraction f < 1 the latent_positions of the sequence charge
    2 N_ne f instead of 2 N_ne (head and attention unchanged)."""
    if layer_fraction == 1.0 or not latent_positions:
        return 3.0 * (2.0 * (dims.N_ne + dims.V * dims.d) * T + 2.0 * dims.n_L * dims.d_a * float(T) ** 2)
    f, n = _frac(layer_fraction), int(latent_positions)
    body = 2 * dims.N_ne * ((T - n) + f * n) + 2 * dims.V * dims.d * T + 2 * dims.n_L * dims.d_a * T * T
    return float(3 * body)


def flops_prefill(dims: Dims, p: int, head_positions: int = 1) -> float:
    """think/lm.py gen_prefix_flops plus the head rows actually computed:
    2 N_ne p + 2 V d head_positions + 2 n_L d_a p^2."""
    return (2.0 * dims.N_ne * p + 2.0 * dims.V * dims.d * head_positions
            + 2.0 * dims.n_L * dims.d_a * float(p) ** 2)


def flops_steps(dims: Dims, p: int, n_steps: int, head_steps: int, layer_fraction: float = 1.0) -> float:
    """think/lm.py gen_flops for n sequential single-position passes after a cached prefix of p:
    2 N_ne n + 2 V d head_steps + 2 n_L d_a [(p+n)^2 - p^2]. Latent steps contribute 0 to head_steps.
    C1b: layer_fraction f scales the 2 N_ne n term and the attention term (only the band's layers run)."""
    if layer_fraction == 1.0:
        return (2.0 * dims.N_ne * n_steps + 2.0 * dims.V * dims.d * head_steps
                + 2.0 * dims.n_L * dims.d_a * (float(p + n_steps) ** 2 - float(p) ** 2))
    f = _frac(layer_fraction)
    return float(f * 2 * dims.N_ne * n_steps + 2 * dims.V * dims.d * head_steps
                 + f * 2 * dims.n_L * dims.d_a * ((p + n_steps) ** 2 - p ** 2))


@dataclass
class InferenceRecord:
    prefill_tokens: int
    latent_steps: int
    emitted_tokens: int
    sequential_passes: int
    flops: float
    reference_flops: float = 0.0     # registration 5.5 (c), amendment A6.4: the reference's own cost (secondary)


LATENT_FEEDBACK_ARMS = ("coconut", "coconut_no_curriculum", "scrambled")
NO_LATENT_ARMS = ("cot", "no_cot")


def inference_record(dims: Dims, arm, q_len: int, latent_steps: int, emitted_tokens: int | None = None,
                     layer_fraction: float = 1.0) -> InferenceRecord:
    """Per-example inference cost by the registration's position table (section 5.3; module docstring). This is
    think/lm.py gen_flops (shared prefix q = 0) with latent positions added to the body and attention terms and not
    to the head term.

    arm: cot | no_cot | coconut | coconut_no_curriculum | scrambled | pause. q_len = |q|. latent_steps = L (for pause:
    the number of pause tokens, which sit in the prefill; the record reports latent_steps 0). emitted_tokens = E'.

    Compatibility: the v1 positional call inference_record(dims, p, latent_steps, emitted_tokens), with p the prompt
    length excluding sequentially fed latents, is still accepted (non-str second argument); it charges
    prefill(p, head 1) + L latent steps (head 0) + E' emitted steps (head 1 each), which equals the table for every arm
    when p is |q| (cot), |q| + 2 (coconut: <start> and <end> counted in p) or |q| + L + 2 (pause).

    layer_fraction (C1b): for the latent-feedback rows, the L latent steps are charged flops_steps(..., f) and the
    1 + E' steps after them as before; 1.0 is the registered table exactly."""
    if not isinstance(arm, str):  # v1 call: (dims, p, latent_steps, emitted_tokens)
        p, L, E = int(arm), int(q_len), int(latent_steps)
        flops = (flops_prefill(dims, p, head_positions=1) + flops_steps(dims, p, L, head_steps=0)
                 + flops_steps(dims, p + L, E, head_steps=E))
        return InferenceRecord(prefill_tokens=p, latent_steps=L, emitted_tokens=E, sequential_passes=1 + L + E,
                               flops=flops, reference_flops=reference_cost_flops(dims, p, 1 + L + E))
    if emitted_tokens is None:
        raise TypeError("inference_record(dims, arm, q_len, latent_steps, emitted_tokens)")
    q, L, E = int(q_len), int(latent_steps), int(emitted_tokens)
    if arm in NO_LATENT_ARMS:
        if L != 0:
            raise ValueError(f"arm {arm} has no latent steps")
        p, head_p, n, head_n, lat = q, 1, E, E, 0
    elif arm in LATENT_FEEDBACK_ARMS:
        p, head_p, n, head_n, lat = q + 1, 0, L + 1 + E, 1 + E, L
    elif arm == "pause":
        p, head_p, n, head_n, lat = q + L + 2, 1, E, E, 0
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if layer_fraction != 1.0:
        if arm not in LATENT_FEEDBACK_ARMS:
            raise ValueError(f"layer_fraction applies to latent-feedback arms, not {arm!r}")
        flops = (flops_prefill(dims, p, head_positions=head_p)
                 + flops_steps(dims, p, L, head_steps=0, layer_fraction=layer_fraction)
                 + flops_steps(dims, p + L, 1 + E, head_steps=head_n))
        return InferenceRecord(prefill_tokens=p, latent_steps=lat, emitted_tokens=E, sequential_passes=1 + lat + E,
                               flops=flops, reference_flops=reference_cost_flops(dims, p, 1 + lat + E))
    flops = flops_prefill(dims, p, head_positions=head_p) + flops_steps(dims, p, n, head_steps=head_n)
    return InferenceRecord(prefill_tokens=p, latent_steps=lat, emitted_tokens=E, sequential_passes=1 + lat + E,
                           flops=flops, reference_flops=reference_cost_flops(dims, p, 1 + lat + E))


def reference_cost_flops(dims: Dims, p0: int, passes: int) -> float:
    """Registration 5.5 (c), amendment A6.4 (review F7): the reference code's own inference cost, the head charged at
    every position and no KV cache. Pass k (k = 0 .. passes - 1) recomputes the whole prefix of length p0 + k with the
    head at every position: sum over k of flops_prefill(dims, p0 + k, head_positions = p0 + k). p0 is the arm's 5.3
    prefill length (|q| for cot / no_cot, |q| + 1 for the latent-feedback arms, |q| + L + 2 for pause) and `passes`
    its 5.3 sequential passes (1 + L + E'; L = 0 for cot / no_cot / pause). No layer fraction: the reference has no
    band. Secondary accounting only: never the verdict (5.5)."""
    return float(sum(flops_prefill(dims, p0 + k, head_positions=p0 + k) for k in range(int(passes))))


__all__ = ["Dims", "model_dims", "flops_train_per_sequence", "flops_prefill", "flops_steps", "InferenceRecord",
           "inference_record", "reference_cost_flops"]
