#!/usr/bin/env bash
set -euo pipefail

python data/posttrain_sft/prepare.py --input=data/posttrain_sft/sample.jsonl --out_dir=data/posttrain_sft
python sft.py config/sft_smoke.py

python chat_eval.py config/chat_eval_smoke.py --out_dir=out-sft-smoke --run_name=sft_smoke_eval

python data/posttrain_dpo/prepare.py --input=data/posttrain_dpo/sample_preferences.jsonl --out_dir=data/posttrain_dpo
python dpo.py config/dpo_smoke.py

python chat_eval.py config/chat_eval_smoke.py --out_dir=out-dpo-smoke --run_name=dpo_smoke_eval

python verified_rollout.py config/verified_rollout_smoke.py --out_dir=out-dpo-smoke --samples_per_prompt=2
python data/posttrain_sft/prepare.py --input=data/posttrain_verified/accepted.jsonl --out_dir=data/posttrain_verified
python sft.py config/verified_ft_smoke.py

python chat_eval.py config/chat_eval_smoke.py --out_dir=out-verified-smoke --run_name=verified_smoke_eval
