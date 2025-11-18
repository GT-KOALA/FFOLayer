#!/bin/bash

batchSize=1
epochs=10

python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffocp_eq --learn_constraint=0 --suffix=_not_learnable_1 --ydim=24

# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffocp_eq --learn_constraint=0 --suffix=_not_learnable_1 --ydim=10
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=qpth --learn_constraint=0 --suffix=_not_learnable_1 --ydim=10
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffoqp_eq_schur --learn_constraint=0 --suffix=_not_learnable_1 --ydim=10

# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffocp_eq --learn_constraint=0 --suffix=_not_learnable_1 --ydim=200
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=qpth --learn_constraint=0 --suffix=_not_learnable_1 --ydim=500
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffoqp_eq_schur --learn_constraint=0 --suffix=_not_learnable_1 --ydim=500

# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffocp_eq --learn_constraint=0 --suffix=_not_learnable_1 --ydim=1000
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=qpth --learn_constraint=0 --suffix=_not_learnable_1 --ydim=1000
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffoqp_eq_schur --learn_constraint=0 --suffix=_not_learnable_1 --ydim=1000

# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffocp_eq --learn_constraint=0 --suffix=_not_learnable_1 --ydim=2000
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=qpth --learn_constraint=0 --suffix=_not_learnable_1 --ydim=2000
# python synthetic_task/main_synthetic.py --batch_size=$batchSize --epochs=$epochs --method=ffoqp_eq_schur --learn_constraint=0 --suffix=_not_learnable_1 --ydim=2000











# python synthetic_task/main_synthetic.py --batch_size=32 --epochs=20 --method=ffocp_eq --learn_constraint=1 --suffix=_learnable_1
# python synthetic_task/main_synthetic.py --batch_size=32 --epochs=20 --method=ffoqp_eq_schur --learn_constraint=1 --suffix=_learnable_1