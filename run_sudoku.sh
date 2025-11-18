#!/bin/bash

batchSize=1
epochs=10
seed=1
n=2

python sudoku/main_sudoku.py --method="ffocp_eq" --epochs=$epochs --seed=$seed --batch_size=$batchSize --n=$n