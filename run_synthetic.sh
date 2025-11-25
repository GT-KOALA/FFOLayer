#!/bin/bash

batchSize=8
epochs=1
ydim=900
# seed=3


seed=$1

if [ -z "$seed" ]; then
  echo "Usage: $0 <seed>"
  exit 1
fi

echo "Running experiments with seed: $seed"


# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=cvxpylayer --learn_constraint=0 --ydim=$ydim --seed=$seed

# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffocp_eq --learn_constraint=0 --ydim=$ydim --seed=$seed


python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=lpgd --learn_constraint=0 --ydim=$ydim --seed=$seed

python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffoqp_eq_schur --learn_constraint=0 --ydim=$ydim --seed=$seed
python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=qpth --learn_constraint=0 --ydim=$ydim --seed=$seed