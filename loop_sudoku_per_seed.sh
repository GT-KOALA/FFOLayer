#!/bin/bash

# seeds=($(seq 1 1 5))
seeds=($(seq 4 1 5))

for seed in "${seeds[@]}"; do
    jobname="sudoku_seed${seed}"
    sbatch --job-name=$jobname sudoku_per_seed.sbatch $seed
    echo "Submitted: $jobname"
done

