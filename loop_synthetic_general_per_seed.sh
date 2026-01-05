#!/bin/bash


seeds=($(seq 1 5))
# seeds=(1)

ydims=($(seq 200 100 1000))
# ydims=($(seq 100 100 400))

batchSize=8

for ydim in "${ydims[@]}"; do
  for seed in "${seeds[@]}"; do
    jobname="syn_y${ydim}_s${seed}_b${batchSize}"
    sbatch --job-name="$jobname" synthetic_general_per_seed.sbatch "$seed" "$ydim" "$batchSize"
    echo "Submitted: $jobname"
  done
done
