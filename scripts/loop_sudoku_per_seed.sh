#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# seeds=($(seq 1 1 5))
seeds=($(seq 6 1 10))
# seeds=(6)

for seed in "${seeds[@]}"; do
    jobname="sudoku_seed${seed}"
    sbatch --job-name=$jobname scripts/sudoku_per_seed.sbatch $seed
    echo "Submitted: $jobname"
done

