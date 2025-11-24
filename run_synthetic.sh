#!/bin/bash

batchSize=8
epochs=1
ydim=900
seed=3

python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffocp_eq --learn_constraint=0 --ydim=$ydim --seed=$seed
python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=lpgd --learn_constraint=0 --ydim=$ydim --seed=$seed
python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=cvxpylayer --learn_constraint=0 --ydim=$ydim --seed=$seed
python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffoqp_eq_schur --learn_constraint=0 --ydim=$ydim --seed=$seed
python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=qpth --learn_constraint=0 --ydim=$ydim --seed=$seed