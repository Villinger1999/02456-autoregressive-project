#!/bin/bash
#BSUB -J plot_mlp
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 00:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/plot_mlp.out
#BSUB -e hpc/logs/plot_mlp.err

set -eu
cd "/dtu/blackhole/1d/187261/02456-autoregressive-project"
source hpc/env.sh

uv run --group eval python eval/plot_structure.py \
    results/eval/mlp/pdb \
    --limit 10 \
    --out results/figures/mlp