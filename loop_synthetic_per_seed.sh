#!/bin/bash


seeds=($(seq 1 1))   # ← ARRAY, not string

ydims=($(seq 900 900))

# backward_eps_list=    #($(seq 0.001 0.00001 0.00000001))
backward_eps_list=($(python3 -c 'print(" ".join([str(x) for x in [0.001,0.00001,0.00000001]]))'))


for seed in "${seeds[@]}"; do
  for ydim in "${ydims[@]}"; do
    for backward_eps in "${backward_eps_list[@]}"; do
      jobname="syn_y${ydim}_s${seed}_tol${backward_eps}"
      sbatch --job-name="$jobname" synthetic_per_seed.sbatch "$seed" "$ydim" "$backward_eps"
      echo "Submitted: $jobname"
    done
  done
done


