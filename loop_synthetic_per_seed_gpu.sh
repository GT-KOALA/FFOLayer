#!/bin/bash


seeds=($(seq 1 1 5))
# seeds=(1)
ydims=($(seq 200 100 1000))

batchSizes=(8)

for seed in "${seeds[@]}"; do
  for ydim in "${ydims[@]}"; do
    for batchSize in "${batchSizes[@]}"; do
      jobname="syn_y${ydim}_s${seed}_b${batchSize}_gpu"
      sbatch --job-name="$jobname" synthetic_per_seed_gpu.sbatch "$seed" "$ydim" "$batchSize"
      echo "Submitted: $jobname"
    done
  done
done


