#!/bin/bash

echo "Sweeping Density"
python3 sweep_by_list.py --sweep keep --values 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

echo "Sweeping Matrix Shape"
python3 sweep_by_list.py --sweep ratio --values 0.125 0.25 0.5 1 2 4 8

echo "Sweeping Mux Size"
python3 sweep_by_list.py --sweep mux --values 4 8 12 16

echo "Sweeping SA Size"
python3 sweep_by_list.py --sweep s --values 32 64 128 256

echo "Results"
python3 look_result.py --json sweep_keep.json sweep_ratio.json sweep_mux.json sweep_s.json

