"""latent.accounting against a hand computation on a 2-layer toy GPT-2 and against torch's parameter counts."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import accounting as A


def toy_config(**kw):
    return GPT2Config(n_layer=2, n_embd=8, n_head=2, vocab_size=11, n_positions=16, **kw)


# Hand computation for toy_config (d = 8, n_inner = 4d = 32, V = 11, n_L = 2, d_a = 2 heads x 4 = 8):
#   per block: ln_1 2d = 16; attn.c_attn d*3d + 3d = 192 + 24 = 216; attn.c_proj d*d + d = 64 + 8 = 72;
#              ln_2 2d = 16; mlp.c_fc d*4d + 4d = 256 + 32 = 288; mlp.c_proj 4d*d + d = 256 + 8 = 264
#              block total = 16 + 216 + 72 + 16 + 288 + 264 = 872
#   2 blocks = 1744; ln_f 2d = 16  ->  N_ne = 1760
#   excluded: wte V*d = 88 (also the tied head), wpe 16*8 = 128; total parameters 1760 + 88 + 128 = 1976
HAND = dict(N_ne=1760, V=11, d=8, n_L=2, d_a=8)
VD = 11 * 8  # 88


def test_toy_dims_by_hand():
    torch.manual_seed(0)
    m = GPT2LMHeadModel(toy_config())
    assert sum(p.numel() for p in m.parameters()) == 1976
    assert A.model_dims(m) == A.Dims(**HAND)
    assert A.model_dims(toy_config()) == A.Dims(**HAND)  # from the config, on the meta device


def test_toy_flops_by_hand():
    dims = A.Dims(**HAND)
    # train, T = 5: 3 * [2 (1760 + 88) 5 + 2*2*8*25] = 3 * [18480 + 800] = 57840
    assert A.flops_train_per_sequence(dims, 5) == 57840
    # prefill p = 4, one head row: 2*1760*4 + 2*88*1 + 2*2*8*16 = 14080 + 176 + 512 = 14768
    assert A.flops_prefill(dims, 4) == 14768
    assert A.flops_prefill(dims, 4, head_positions=0) == 14768 - 176
    # 3 steps after p = 4, one emitting: 2*1760*3 + 2*88*1 + 2*2*8*(49 - 16) = 10560 + 176 + 1056 = 11792
    assert A.flops_steps(dims, 4, 3, head_steps=1) == 11792
    # 3 latent steps (no head): 11792 - 176 = 11616
    assert A.flops_steps(dims, 4, 3, head_steps=0) == 11616
    # registration 5.3, coconut, |q| = 3, L = 3 latents, E' = 2 emitted tokens:
    #   prefill p = |q| + 1 = 4 with 0 head positions: 2*1760*4 + 0 + 2*2*8*4^2 = 14080 + 512 = 14592
    #   n = L + 1 + E' = 6 steps with 1 + E' = 3 head steps: 2*1760*6 + 2*88*3 + 2*2*8*(10^2 - 4^2)
    #     = 21120 + 528 + 2688 = 24336;  total 38928; passes 1 + 3 + 2 = 6
    r = A.inference_record(dims, "coconut", 3, 3, 2)
    assert r.flops == 38928
    assert (r.prefill_tokens, r.latent_steps, r.emitted_tokens, r.sequential_passes) == (4, 3, 2, 6)
    # cot, |q| = 7, E' = 5: prefill 2*1760*7 + 2*88*1 + 2*2*8*49 = 24640 + 176 + 1568 = 26384;
    #   5 steps, 5 heads: 2*1760*5 + 2*88*5 + 2*2*8*(144 - 49) = 17600 + 880 + 3040 = 21520; total 47904
    #   = think/lm.py gen_flops with q = 0 and E' + 1 head rows: 2 N_ne (7+5) + 2 Vd (5+1) + 2 n_L d_a 12^2
    r = A.inference_record(dims, "cot", 7, 0, 5)
    assert r.flops == 47904 == 2 * 1760 * 12 + 2 * VD * 6 + 2 * 2 * 8 * 144
    assert (r.prefill_tokens, r.latent_steps, r.sequential_passes) == (7, 0, 6)
    # pause, |q| = 3, L = 3 pauses, E' = 2: prefill 3+3+2 = 8 with 1 head: 2*1760*8 + 176 + 32*64 = 28160 + 176
    #   + 2048 = 30384; 2 steps, 2 heads: 2*1760*2 + 352 + 32*(100 - 64) = 7040 + 352 + 1152 = 8544; total 38928,
    #   equal to coconut with L = 3 (same 10 positions, same 3 head rows) but in 1 + E' = 3 sequential passes
    r = A.inference_record(dims, "pause", 3, 3, 2)
    assert r.flops == 30384 + 8544 == 38928
    assert (r.prefill_tokens, r.latent_steps, r.sequential_passes) == (8, 0, 3)
    # zero-latent: coconut with L = 0
    r = A.inference_record(dims, "coconut", 3, 0, 2)
    assert (r.prefill_tokens, r.latent_steps, r.sequential_passes) == (4, 0, 3)


def test_registration_worked_example():
    """Registration 5.1: GPT-2 124M dims (GPT2Config defaults, V = 50,260), ProsQA coconut, |q| = 250, L = 6, E' = 8.
    prefill (head 0) = 2 x 85,056,000 x 251 + 2 x 12 x 768 x 251^2 = 43,859,346,432
    steps n = 15, head 9 = 2 x 85,056,000 x 15 + 2 x 38,599,680 x 9 + 2 x 12 x 768 x (266^2 - 251^2) = 3,389,414,400
    total 47,248,760,832 FLOPs, 15 sequential passes."""
    dims = A.model_dims(GPT2Config(vocab_size=50260))
    assert dims == A.Dims(N_ne=85_056_000, V=50_260, d=768, n_L=12, d_a=768)
    assert A.flops_prefill(dims, 251, head_positions=0) == 43_859_346_432
    assert A.flops_steps(dims, 251, 15, head_steps=9) == 3_389_414_400
    r = A.inference_record(dims, "coconut", 250, 6, 8)
    assert r.flops == 47_248_760_832
    assert (r.sequential_passes, r.emitted_tokens, r.latent_steps, r.prefill_tokens) == (15, 8, 6, 251)
    for arm in ("coconut_no_curriculum", "scrambled"):
        assert A.inference_record(dims, arm, 250, 6, 8) == r


@pytest.mark.parametrize("arm,q,L,E", [("cot", 40, 0, 30), ("no_cot", 40, 0, 5), ("coconut", 40, 6, 8),
                                       ("coconut", 40, 0, 8), ("pause", 40, 6, 8)])
def test_v1_positional_call_equals_table(arm, q, L, E):
    """eval.py's v1 call passes p = prompt length minus sequentially fed latents; it must charge the same FLOPs."""
    dims = A.model_dims(GPT2Config(vocab_size=50260))
    v2 = A.inference_record(dims, arm, q, L, E)
    p = {"cot": q, "no_cot": q, "coconut": q + 2, "pause": q + L + 2}[arm]
    v1 = A.inference_record(dims, p, 0 if arm == "pause" else L, E)
    assert v1.flops == v2.flops and v1.sequential_passes == v2.sequential_passes


def test_bad_arm():
    dims = A.Dims(**HAND)
    with pytest.raises(ValueError):
        A.inference_record(dims, "cot", 3, 2, 1)
    with pytest.raises(ValueError):
        A.inference_record(dims, "nope", 3, 0, 1)


def test_n_ne_matches_torch_numel_tied_and_resized():
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=3, n_embd=16, n_head=4, vocab_size=50, n_positions=32))
    assert m.lm_head.weight is m.transformer.wte.weight  # GPT-2 ties the head to wte
    expected = sum(p.numel() for n, p in m.named_parameters()
                   if not n.startswith(("transformer.wte.", "transformer.wpe.")))
    dims = A.model_dims(m)
    assert dims.N_ne == expected
    assert (dims.V, dims.d, dims.n_L, dims.d_a) == (50, 16, 3, 16)
    m.resize_token_embeddings(53)  # the three latent special tokens
    d2 = A.model_dims(m)
    assert d2.V == 53 and d2.N_ne == expected


def test_untied_head_charged_through_vd_only():
    m = GPT2LMHeadModel(toy_config(tie_word_embeddings=False))
    assert m.lm_head.weight is not m.transformer.wte.weight
    assert A.model_dims(m).N_ne == HAND["N_ne"]


def test_wrapper_unwrapped():
    class Wrap(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_causallm = base
    assert A.model_dims(Wrap(GPT2LMHeadModel(toy_config()))) == A.Dims(**HAND)


@pytest.mark.parametrize("cfg,expected", [
    (dict(), A.Dims(N_ne=85_056_000, V=50257, d=768, n_L=12, d_a=768)),                      # gpt2 (124M)
    (dict(n_layer=24, n_embd=1024, n_head=16), A.Dims(302_311_424, 50257, 1024, 24, 1024)),  # gpt2-medium (355M)
    (dict(n_layer=36, n_embd=1280, n_head=20), A.Dims(708_390_400, 50257, 1280, 36, 1280)),  # gpt2-large (774M)
])
def test_gpt2_family_from_config(cfg, expected):
    """Registration 1.1: N_ne from parameter shapes (meta device, no weights). 124M: 124,439,808 parameters -
    wte 38,597,376 - wpe 786,432 = 85,056,000."""
    assert A.model_dims(GPT2Config(**cfg)) == expected


def test_c1b_layer_fraction_worked_example():
    """C1b "Math and accounting" (registration A3.4): the 5.1 worked example with the six latent steps run through a
    band of 8/12 and 4/12 of the blocks. By hand (124M dims, |q| = 250, L = 6, E' = 8):
      prefill (head 0)              2 x 85,056,000 x 251 + 2 x 12 x 768 x 251^2          = 43,859,346,432
      6 latent steps, full depth    2 x 85,056,000 x 6 + 2 x 12 x 768 x (257^2 - 251^2) = 1,020,672,000 + 56,180,736
      <end> + 8 emitted (head 9)    2 x 85,056,000 x 9 + 2 x 38,599,680 x 9 + 18,432 x (266^2 - 257^2)
                                    = 1,531,008,000 + 694,794,240 + 86,759,424
      f = 8/12: latent steps 680,448,000 + 37,453,824; total 46,889,809,920 (= 47,248,760,832 - 358,950,912)
      f = 4/12: latent steps 340,224,000 + 18,726,912; total 46,530,859,008 (= 47,248,760,832 - 717,901,824)
    Sequential passes stay 15; layer_fraction 1 is the registered table bit for bit."""
    from fractions import Fraction
    dims = A.model_dims(GPT2Config(vocab_size=50260))
    full = A.inference_record(dims, "coconut", 250, 6, 8)
    assert A.inference_record(dims, "coconut", 250, 6, 8, layer_fraction=1.0) == full
    assert full.flops == 47_248_760_832
    assert A.flops_steps(dims, 251, 6, 0, layer_fraction=8 / 12) == 680_448_000 + 37_453_824
    assert A.flops_steps(dims, 251, 6, 0, layer_fraction=Fraction(4, 12)) == 340_224_000 + 18_726_912
    for f, want in ((8 / 12, 46_889_809_920), (Fraction(8, 12), 46_889_809_920), (4 / 12, 46_530_859_008)):
        r = A.inference_record(dims, "coconut", 250, 6, 8, layer_fraction=f)
        assert r.flops == want
        assert (r.sequential_passes, r.latent_steps, r.emitted_tokens, r.prefill_tokens) == (15, 6, 8, 251)
        assert A.inference_record(dims, "scrambled", 250, 6, 8, layer_fraction=f) == r
    assert A.flops_steps(dims, 251, 15, 9, layer_fraction=1.0) == A.flops_steps(dims, 251, 15, 9) == 3_389_414_400
    with pytest.raises(ValueError):
        A.inference_record(dims, "cot", 250, 0, 8, layer_fraction=0.5)
    with pytest.raises(ValueError):
        A.flops_steps(dims, 251, 6, 0, layer_fraction=0.0)


def test_c1b_training_flops_scale_latent_positions_only():
    dims = A.Dims(**HAND)
    # T = 5 with 2 latent positions at f = 1/2: 3 * [2 * 1760 * (3 + 1) + 2 * 88 * 5 + 2 * 2 * 8 * 25]
    assert A.flops_train_per_sequence(dims, 5, latent_positions=2, layer_fraction=0.5) == 3 * (14080 + 880 + 800)
    assert A.flops_train_per_sequence(dims, 5, latent_positions=2, layer_fraction=1.0) == 57840
    assert A.flops_train_per_sequence(dims, 5) == 57840


def test_reference_cost_accounting_hand_example():
    """Registration 5.5 (c), amendment A6.4: the critic's hand example on the 5.1 worked example (GPT-2 124M dims,
    V = 50,260, p0 = 251, L = 6, E' = 8, K = 15 passes over prefixes of length 251 to 265, head at every position):
    975,503,738,880 FLOPs, 20.65x the primary 47,248,760,832. The record carries it as reference_flops; the primary
    flops and passes are unchanged."""
    dims = A.model_dims(GPT2Config(vocab_size=50260))
    assert A.reference_cost_flops(dims, 251, 15) == 975_503_738_880
    r = A.inference_record(dims, "coconut", 250, 6, 8)
    assert r.reference_flops == 975_503_738_880 and r.flops == 47_248_760_832 and r.sequential_passes == 15
    assert abs(r.reference_flops / r.flops - 20.65) < 0.01
    # cot: p0 = |q|, K = 1 + E'; pause: p0 = |q| + L + 2, K = 1 + E'; the band path charges the reference without a band
    assert A.inference_record(dims, "cot", 250, 0, 8).reference_flops == A.reference_cost_flops(dims, 250, 9)
    assert A.inference_record(dims, "pause", 250, 6, 8).reference_flops == A.reference_cost_flops(dims, 258, 9)
    assert A.inference_record(dims, "coconut", 250, 6, 8, layer_fraction=8 / 12).reference_flops == 975_503_738_880
    # the v1 positional call (p = |q| + 1 for coconut) gives the same reference cost
    assert A.inference_record(dims, 251, 6, 8).reference_flops == 975_503_738_880
    # one pass of one position with the head equals flops_prefill with the head at that position
    assert A.reference_cost_flops(dims, 1, 1) == A.flops_prefill(dims, 1, head_positions=1)

