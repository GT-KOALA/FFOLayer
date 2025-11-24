#!/bin/bash
#SBATCH -J bilevel_sudoku                       # job name
#SBATCH -A gts-kwang692-paid                  # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --array=1-50                      # run 1 jobs with seeds 1
#SBATCH --mem-per-cpu=16G                 # memory per core
#SBATCH -t 360                           # duration of the job
#SBATCH --output=/nethome/sho73/bilevel-layer_package/sudoku_sbatch_output/%j.out


methods=("ffocp_eq" "cvxpylayer" "lpgd" "ffoqp_eq" "qpth")

method_index=$(( (SLURM_ARRAY_TASK_ID - 1) / 10 ))
seed=$(( (SLURM_ARRAY_TASK_ID - 1) % 10 + 1 ))
method=${methods[$method_index]}

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Running method=$method, seed=$seed"

epochs=10
batchSize=8
ydim=200
python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=$method --seed=$seed --learn_constraint=0 --ydim=$ydim
