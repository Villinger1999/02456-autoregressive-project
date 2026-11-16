#!/bin/bash
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -o hpc/logs/train.out
#BSUB -e hpc/logs/train.err

# Submit your training run to a GPU queue:   bsub < hpc/train.sh

cd "/dtu/blackhole/1d/187261/02456-autoregressive-project"
source hpc/env.sh
uv run python src/protgen/models/mlp.py