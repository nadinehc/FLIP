#!/bin/bash
#SBATCH --job-name=flip-sweep
#SBATCH --output=out/sweep/slurm-%j.out
#SBATCH --error=out/sweep/slurm-%j.out
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Unattended launch of the FLIP vs FLIP+G sweep on a SLURM cluster.
#
#   sbatch sweep/submit_sweep.sbatch                 # full sweep
#   sbatch sweep/submit_sweep.sbatch --smoke         # pipeline check first
#   TAG=main sbatch sweep/submit_sweep.sbatch
#
# Any extra arguments are forwarded to sweep/run_sweep.py. Adjust the SBATCH
# directives above (partition, account, time) to the cluster's requirements --
# they are left generic on purpose.
#
# The sweep is resumable: re-submitting the same TAG skips completed steps, so
# hitting the wall clock is recoverable.

set -euo pipefail

TAG="${TAG:-main}"
PYTHON="${PYTHON:-python}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p out/sweep

# Load the project environment here, e.g.:
#   module load cuda/11.3
#   source activate flip
if [[ -n "${FLIP_ENV_SETUP:-}" ]]; then
    # shellcheck disable=SC1090
    source "${FLIP_ENV_SETUP}"
fi

# Fail fast and loudly if the environment was not activated: inside a batch
# job no conda env is active by default, and the sweep would otherwise run
# every step against an interpreter with no torch.
if ! "${PYTHON}" -c "import torch" 2>/dev/null; then
    echo "ERROR: '${PYTHON}' cannot import torch." >&2
    echo "Activate the project env in this script (e.g. 'conda activate flip')" >&2
    echo "or point FLIP_ENV_SETUP at a script that does." >&2
    exit 1
fi

echo "host:    $(hostname)"
echo "tag:     ${TAG}"
echo "python:  $(${PYTHON} --version 2>&1)"
nvidia-smi || echo "warning: nvidia-smi unavailable"

exec "${PYTHON}" sweep/run_sweep.py --tag "${TAG}" --python "${PYTHON}" "$@"
