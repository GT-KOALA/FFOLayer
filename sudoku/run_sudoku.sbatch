#!/bin/bash
#SBATCH -J bilevel_sudoku                       # job name
#SBATCH -A gts-kwang692-paid                  # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --array=1                      # run 1 jobs with seeds 1
#SBATCH --mem-per-cpu=16G                 # memory per core
#SBATCH -t 360                           # duration of the job
#SBATCH --output=/nethome/sho73/bilevel-layer_package/sudoku_sbatch_output/%j.out

cd /nethome/sho73/bilevel-layer_package

source /nethome/sho73/miniconda3/etc/profile.d/conda.sh
conda activate bilevel

METHOD=ffocp_eq
N=2
EPOCHS=20
LR=0.1
BATCH_SIZE=32

# Use the array index as seed
SEED=$SLURM_ARRAY_TASK_ID

echo "METHOD=$METHOD, SEED=$SEED, N=$N, EPOCHS=$EPOCHS, LR=$LR, BATCH_SIZE=$BATCH_SIZE"


export GRB_LICENSE_FILE=/nethome/sho73/gurobi.lic


python sudoku/main_sudoku.py \
  --method $METHOD \
  --epochs $EPOCHS \
  --seed $SEED \
  --lr $LR \
  --batch_size $BATCH_SIZE \
  --n $N
