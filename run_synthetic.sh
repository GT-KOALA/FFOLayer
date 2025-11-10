#!/bin/bash

python synthetic_task/main_synthetic.py --batch_size=32 --epochs=20 --method=cvxpylayer --learn_constraint=0 --suffix=_not_learnable_1
# python synthetic_task/main_synthetic.py --batch_size=32 --epochs=20 --method=ffoqp_eq_schur --learn_constraint=0 --suffix=_not_learnable_1

# python synthetic_task/main_synthetic.py --batch_size=32 --epochs=20 --method=ffocp_eq --learn_constraint=1 --suffix=_learnable_1
# python synthetic_task/main_synthetic.py --batch_size=32 --epochs=20 --method=ffoqp_eq_schur --learn_constraint=1 --suffix=_learnable_1