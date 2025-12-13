#!/bin/bash


seeds=($(seq 1))   # ← ARRAY, not string

ydims=($(seq 200 100 1000))

for seed in "${seeds[@]}"; do
  for ydim in "${ydims[@]}"; do
    jobname="syn_y${ydim}_s${seed}"
    sbatch --job-name="$jobname" synthetic_per_seed_gpu.sbatch "$seed" "$ydim"
    echo "Submitted: $jobname"
  done
done


