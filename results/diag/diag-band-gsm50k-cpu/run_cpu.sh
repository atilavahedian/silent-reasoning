cd <home>/hinton
export PYTHONPATH=common:code
date +%s >> $SP/band64/stamps.txt; $SP/venv/bin/python code/run_diag.py --config configs/screen/gsm50k_coconut.json --checkpoint $SP/band64/ckpt/ckpt_best_final.pt --mode band_logits --set 'device="cpu"' --set seed=0 --set 'name="gsm50k_coconut_s0"' --set 'val_path="$SP/band64/data/gsm_valid.json"' --base-config $SP/band64/gpt2_cfg --dtype float64 --strict-only --out $SP/band64/gsm50k_coconut_s0_valid_band_logits_cpu_float64 --commit rev210 > $SP/band64/run64.log 2>&1
echo EXIT $? >> $SP/band64/run64.log
date +%s >> $SP/band64/stamps.txt; $SP/venv/bin/python code/run_diag.py --config configs/screen/gsm50k_coconut.json --checkpoint $SP/band64/ckpt/ckpt_best_final.pt --mode band_logits --set 'device="cpu"' --set seed=0 --set 'name="gsm50k_coconut_s0"' --set 'val_path="$SP/band64/data/gsm_valid.json"' --base-config $SP/band64/gpt2_cfg --dtype float32 --strict-only --out $SP/band64/gsm50k_coconut_s0_valid_band_logits_cpu_float32 --commit rev210 > $SP/band64/run32.log 2>&1
echo EXIT $? >> $SP/band64/run32.log
date +%s >> $SP/band64/stamps.txt
