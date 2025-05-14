#!/bin/bash
#SBATCH -J bilevel                       # job name
#SBATCH -A gts-kwang692                  # account to which job is charged
#SBATCH -N 1 --ntasks-per-node=1         # number of nodes and cores per node required
#SBATCH --mem-per-cpu=8G                 # memory per core
#SBATCH -t 360                           # duration of the job
#SBATCH --output=/storage/home/hcoda1/9/kwang692/p-kwang692-0/constrained-bilevel-optimization/output/%j.out

cd /storage/home/hcoda1/9/kwang692/p-kwang692-0/ffoqp

module load anaconda3
conda activate bilevel
echo echo
echo $METHOD
echo $SEED
echo $YDIM
echo $EPS
echo $LR

# Running linear inequality
# Change this to main_bilinear.py for bilinear case
python3 main.py --method=$METHOD --seed=$SEED --ydim=$YDIM --lr=$LR --eps=$EPS
