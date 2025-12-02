#!/bin/bash


seeds=($(seq 1 3))   # ← ARRAY, not string

for seed in "${seeds[@]}"; do
    jobname="syn_seed${seed}"
    sbatch --job-name=$jobname synthetic_per_seed.sbatch $seed
    echo "Submitted: $jobname"
done


