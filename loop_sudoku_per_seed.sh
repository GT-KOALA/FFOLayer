#!/bin/bash

seeds=($(seq 1 3))  # generates 1 2 3 ... 10

for seed in "${seeds[@]}"; do
    jobname="sudoku_seed${seed}"
    sbatch --job-name=$jobname sudoku_per_seed.sbatch $seed
    echo "Submitted: $jobname"
done

