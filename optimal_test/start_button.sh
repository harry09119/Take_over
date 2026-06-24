#!/bin/bash

python compare_greedy_vs_exhaustive_local_exact_8tiles.py \
  --root . \
  --packing packing.py \
  --base-oracle ctf_optimality_oracle_dp_lb.py \
  --row-exact ctf_row_pairing_exact_oracle_memo.py \
  --worker-script exact_order_worker_8tiles.py \
  --seed 4 \
  --density 0.20 \
  --workers 10 \
  --pair-timeout 300 \
  --order-timeout 1200 \
  --output-dir exact_8tiles_seed4 \
  --resume

python compare_existing_exact_hungarian_beam_8tiles.py \
  --packing packing.py \
  --seed 4 \
  --density 0.20 \
  --beam-width 4 \
  --row-option-limit 8 \
  --exact-summary exact_8tiles_seed20/summary.json \
  --output-dir final_8tiles_comparison
