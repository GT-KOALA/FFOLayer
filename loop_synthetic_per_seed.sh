#!/bin/bash


seeds=($(seq 1 10))   # ← ARRAY, not string

# ydims=($(seq 200 100 1000))
ydims=(900)

# backward_eps_list=    #($(seq 0.001 0.00001 0.00000001)) #0.001,0.00001,0.00000001, 0.000000000001
backward_eps_list=($(python3 -c 'print(" ".join([str(x) for x in [0.001,0.00001,0.00000001, 0.000000000001]]))'))

batch_sizes=(8)


for seed in "${seeds[@]}"; do
  for ydim in "${ydims[@]}"; do
    for backward_eps in "${backward_eps_list[@]}"; do
      for batchSize in "${batch_sizes[@]}"; do
        jobname="syn_y${ydim}_s${seed}_tol${backward_eps}_batch${batchSize}"
        sbatch --job-name="$jobname" synthetic_per_seed.sbatch "$seed" "$ydim" "$batchSize" "$backward_eps"
        echo "Submitted: $jobname"
      done
    done
  done
done


