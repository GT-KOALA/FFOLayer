#!/bin/bash

# seeds=($(seq 1 3))
seeds=(1)

for seed in "${seeds[@]}"; do
    jobname="sudoku_seed${seed}"
    sbatch --job-name=$jobname sudoku_per_seed.sbatch $seed
    echo "Submitted: $jobname"
done

