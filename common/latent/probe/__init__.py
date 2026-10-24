"""latent.probe: readout (R1 logit lens, R2 linear probes, R3 decoder interface) and causal faithfulness (F1
counterfactual patching, F2 scrambled drop, F3 probe-direction edits) for LatentLM checkpoints, plus the synthetic
reachability positive control (plan section 6.5, registration section 7.9, theory memo T3 and T4).

Modules: stats (bootstrap and binomial helpers), extract (latents and per-step state targets), tasks (task
adapters: prosqa, reach, gsm; minimal-pair rules; edit specs), linear (R1, R2, R3 scoring), patch (F1, F2 and the
bank mechanism), edit (F3 and the load-bearing fraction), report (JSON and Markdown per checkpoint), synthetic
(the T3 positive control).
"""
