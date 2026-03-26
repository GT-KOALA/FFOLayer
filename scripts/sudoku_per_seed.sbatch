#!/bin/bash
#SBATCH -J bilevel_sudoku
#SBATCH -A gts-kwang692-paid                  # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --mem-per-cpu=16G                 # memory per core
#SBATCH -t 360                           # duration of the job
#SBATCH --cpus-per-task=16
#SBATCH --output=/storage/home/hcoda1/3/sho73/r-kwang692-0/sudoku_sbatch_output/%j.out

HOME=/storage/home/hcoda1/3/sho73/r-kwang692-0
REPO_PATH=$HOME/project/FFOLayer

cd $REPO_PATH
source $HOME/my_miniconda/etc/profile.d/conda.sh
conda activate bilevel


seed=$1

echo "Running seed=$seed"

epochs=1
batchSize=8
n=3
# methods=("ffocp_eq" "cvxpylayer" "lpgd" "ffoqp_eq" "qpth" "bpqp")
methods=("ffocp_eq")
for method in "${methods[@]}"; do
    echo "Running method=$method, seed=$seed"
    python sudoku/main_sudoku.py --method=$method --seed=$seed --epochs=$epochs --batch_size=$batchSize --n=$n
done