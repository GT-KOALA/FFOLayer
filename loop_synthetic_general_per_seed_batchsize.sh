#!/bin/bash


seeds=(1)   # ← ARRAY, not string

ydims=(200)
batchSizes=(1 2 4 8 16 32)

for ydim in "${ydims[@]}"; do
  for seed in "${seeds[@]}"; do
    for batchSize in "${batchSizes[@]}"; do
      jobname="syn_y${ydim}_s${seed}_b${batchSize}"
      sbatch --job-name="$jobname" synthetic_general_per_seed.sbatch "$seed" "$ydim" "$batchSize"
      echo "Submitted: $jobname"
    done
  done
done
