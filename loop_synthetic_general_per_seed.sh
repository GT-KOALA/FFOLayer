#!/bin/bash


seeds=($(seq 1))   # ← ARRAY, not string

ydims=($(seq 200 100 1000))

for ydim in "${ydims[@]}"; do
  for seed in "${seeds[@]}"; do
    jobname="syn_y${ydim}_s${seed}"
    sbatch --job-name="$jobname" synthetic_general_per_seed.sbatch "$seed" "$ydim"
    echo "Submitted: $jobname"
  done
done
