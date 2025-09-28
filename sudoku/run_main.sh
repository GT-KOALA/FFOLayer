#!/bin/bash

ROOT=/nethome/sho73

cd $ROOT/bilevel_layer_project/bilevel-layer_package

source $ROOT/miniconda3/etc/profile.d/conda.sh
conda activate bilevel

N=2
SEED=1

######### FFOCP
METHOD=ffocp_eq
EPOCHS=30
LR=0.001
BATCH_SIZE=150

##########FFOQP
# METHOD=ffoqp_eq
# EPOCHS=18
# LR=0.1
# BATCH_SIZE=150


echo "METHOD=$METHOD, SEED=$SEED, N=$N, EPOCHS=$EPOCHS, LR=$LR, BATCH_SIZE=$BATCH_SIZE"


export GRB_LICENSE_FILE=/nethome/sho73/gurobi.lic

#python sudoku/main_sudoku.py --method $METHOD --epochs $EPOCHS --seed $SEED --lr $LR --batch_size $BATCH_SIZE --n $N

python sudoku/main_sudoku.py \
  --method $METHOD \
  --epochs $EPOCHS \
  --seed $SEED \
  --lr $LR \
  --batch_size $BATCH_SIZE \
  --n $N
