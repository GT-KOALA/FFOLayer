#!/bin/bash

python main.py --task ffoqp_eq_cst --nRuns 1 --cuda_device 0 --seed 0 --chunk_size 10
python main.py --task ffoqp_eq_cst_pdipm --nRuns 1 --cuda_device 0 --seed 0 --chunk_size 10
python main.py --task ffoqp_eq_cst_parallelize --nRuns 1 --cuda_device 0 --seed 0 --chunk_size 10
python main.py --task ffoqp --nRuns 1 --cuda_device 0 --seed 0 --chunk_size 10
python main.py --task qpth --nRuns 1 --cuda_device 0 --seed 0 --chunk_size 10
python main.py --task cvxpylayer --nRuns 1 --cuda_device 0 --seed 0 --chunk_size 10
python main.py --task cvxpylayer_lpgd --nRuns 1 --cuda_device 0 --seed 0 --chunk_size 10
