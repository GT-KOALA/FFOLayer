#!/bin/bash


seeds=($(seq 1))   # ← ARRAY, not string

ydims=($(seq 200 100 1000))

batchSizes=8

for seed in "${seeds[@]}"; do
  for ydim in "${ydims[@]}"; do
    jobname="syn_y${ydim}_s${seed}_b${batchSize}"
    sbatch --job-name="$jobname" synthetic_per_seed.sbatch "$seed" "$ydim" "$batchSize"
    echo "Submitted: $jobname"
  done
done


