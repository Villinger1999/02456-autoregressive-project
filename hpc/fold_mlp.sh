#!/bin/bash
#BSUB -J fold_mlp
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 02:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/fold_mlp.out
#BSUB -e hpc/logs/fold_mlp.err

set -eu

cd "/dtu/blackhole/1d/187261/02456-autoregressive-project"
source hpc/env.sh

uv run --group eval python eval/eval.py \
    --samples results/sequences/mlp_generated.csv \
    --reference data/train.csv \
    --out results/eval/mlp \
    --fold