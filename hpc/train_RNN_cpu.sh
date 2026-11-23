#!/bin/bash
#BSUB -q hpc
#BSUB -J train
#BSUB -n 4
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o hpc/logs/train_%J.out
#BSUB -e hpc/logs/train_%J.err

# Submit your training run to the CPU queue, from the repository root:   bsub < hpc/train.sh
# For a GPU run instead: set -q to a GPU queue (gpuv100, gpua100, ...) and add
#   #BSUB -gpu "num=1:mode=exclusive_process"
#
# Run `bash hpc/setup_env.sh` once before this, to install the environment.
# %J in the log names is the LSF job id, so each run keeps its own logs.
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu

# LSF copies this script to a spool directory before running it, so $0 does not point into the
# repository. The job starts in the directory it was submitted from; LS_SUBCWD names it too.
cd "${LS_SUBCWD:-$PWD}"
source hpc/env.sh   # uv on the PATH, caches off the home quota

echo "job ${LSB_JOBID:-local} on $(hostname), started $(date)"

# Let torch use exactly the cores LSF gave the job (-n above), not every core on the node.
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-4}"

# -u: unbuffered output, so the per-epoch prints show up in the log while the job runs.
uv run python -u -m protgen.models.RNN_v

echo "finished $(date)"
