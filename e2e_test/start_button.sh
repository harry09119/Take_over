#!/usr/bin/env bash

#!/bin/bash

MODEL=$1

python pack_wgt.py --model "$MODEL"
python run_non_gemm.py --model "$MODEL"

for METHOD in 0 1 2 3
do
  python run_gemm.py --model "$MODEL" --method_idx "$METHOD"
done

python look_result.py --models "$MODEL"
