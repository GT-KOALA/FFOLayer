#!/bin/bash


seeds=($(seq 1))   # ← ARRAY, not string

ydims=(800)
batchSizes=(16 32)

for seed in "${seeds[@]}"; do
  for ydim in "${ydims[@]}"; do
    for batchSize in "${batchSizes[@]}"; do
      jobname="syn_y${ydim}_s${seed}_b${batchSize}"
      sbatch --job-name="$jobname" synthetic_per_seed.sbatch "$seed" "$ydim" "$batchSize"
      echo "Submitted: $jobname"
    done
  done
done


