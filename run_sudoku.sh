#!/bin/bash

batchSize=8 #8
epochs=1
seed=2
n=2

python sudoku/main_sudoku.py --method="ffocp_eq" --epochs=$epochs --seed=$seed --batch_size=$batchSize --n=$n
# python sudoku/main_sudoku.py --method="cvxpylayer" --epochs=$epochs --seed=$seed --batch_size=$batchSize --n=$n
# python sudoku/main_sudoku.py --method="lpgd" --epochs=$epochs --seed=$seed --batch_size=$batchSize --n=$n
# python sudoku/main_sudoku.py --method="ffoqp_eq" --epochs=$epochs --seed=$seed --batch_size=$batchSize --n=$n
# python sudoku/main_sudoku.py --method="qpth" --epochs=$epochs --seed=$seed --batch_size=$batchSize --n=$n
