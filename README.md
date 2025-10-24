This is the implementation of fully first-order bilevel gradient to replace the non-fully first-order methods in differentiable optimization.
The main comparison is ``qpth`` and ``cvxpylayer``. Our method provides a fully first-order method to differentiate through optimization layers approximately.
It has the advantage in computation and memory efficiency, but in the tradeoff of the accuracy of the gradient.

Please install the required packages:
- torch
- cvxpy
- qpth
- cvxpylayer
- gurobi (python -m pip install gurobipy)
- cvxtorch (git clone https://github.com/cvxpy/cvxtorch.git, cd cvxtorch, pip install -e .)

---

## 1) `Decision-focused learning (QP case)`
**Key files**
  - `main.py` — entrypoint for all methods
  - `ffoqp.py` - previous ffo version with l2 penalty
  - `ffoqp_eq_cst.py` - new ffo version without l2 penalty (faster convergence in theory)
  - `ffoqp_eq_cst_schur/pdipm/parallelize.py` - try to accelerate ffoqp_eq_cst

To run the code, please use:
```
python main.py --method=ffoqp_eq_cst --eps=0.1 --lr=0.00001 --ydim=200
python main.py --method=ffoqp --eps=0.1 --lr=0.00001 --ydim=200
python main.py --method=cvxpylayer --eps=0.1 --lr=0.00001 --ydim=200
python main.py --method=qpth --eps=0.1 --lr=0.00001 --ydim=200

(optional)
python main.py --method=ffoqp_eq_cst_pdipm --eps=0.1 --lr=0.00001 --ydim=200
python main.py --method=ffoqp_eq_cst_parallelize --eps=0.1 --lr=0.00001 --ydim=200
python main.py --method=ffoqp_eq_cst_schur --eps=0.1 --lr=0.00001 --ydim=200
```


I ran a few simple experiments to compare the performance of the three methods. Each epoch includes 2048 samples (split into training and testing) to run decision-focused leanring (end-to-end learning).

For optimization problems of size 200, the computation results are as follows (please run plot_toy_dfl.ipynb to reproduce it)
<img width="959" height="465" alt="image" src="https://github.com/user-attachments/assets/b2ae816b-0353-4420-b6ee-8758f9567514" />



The computation improvement can only be seen in larger instances of the optimization problem. This is likely due to the dominance of the overhead in the forward process to solve the optimization problem. Only when the size goes larger, the backward process of the optimization problem becomes more expensive than the forward process. In smaller instances, the overhead in the forward process dominates the computation time, and thus the improvement in the backward process is not significant enough to be observed.
qpth also uses a faster algorithm to parallelize and implement the forward pass, which is why qpth is faster than cvxpylayer (in theory they are the same).

---

## 2) `Second-order Cone program`
To run the code, please run:
```
ffoqsocp_test.ipynb
```

We found that ffoqp's gradient approximation is more accurate than LPGD in this case.

---
## 3) `Sudoku (differentiable optimization as a layer)`
**Key files**
  - `sudoku/main_sudoku.py` — entrypoint for all methods
  - `ffocp_eq.py` - new ffo version for **general convex problem** without l2 penalty
  - `cvxpylayers_local/cvxpylayer.py` - incorporate the latest diffcp into cvxpylayer to support LPGD

To run the code, please use:
```
python sudoku/main_sudoku.py --method ffocp --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method lpgd --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method ffoqp --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method cvxpylayer --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method qpth --n 3 --epoch 1 --batch_size 8
```

To plot the results, please use
```
python sudoku/plot_results.py
```

Now we observed that LPGD has a very high test error (~0.9), while ffocp and other methods can achieve low test errors (~0.01-0.1).
