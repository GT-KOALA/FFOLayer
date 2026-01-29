# FFOLayer — A Fully First-Order Layer for Differentiable Optimization

`FFOLayer` is a PyTorch-friendly library for **differentiable optimization layers** that computes hypergradients using **only first-order information**. It is designed as a practical, drop-in alternative to implicit-differentiation-based layers when memory or backward-time is the bottleneck.

---
## Installation
FFOLayer is available on pip:

```bash
pip install ffolayer
```

You may also need to install cvxtorch:

```bash
git clone https://github.com/cvxpy/cvxtorch.git
cd cvxtorch
pip install -e .
```

---

## Usage

FFOLayer follows the same workflow as differentiable layers like [CvxpyLayer](https://github.com/cvxpy/cvxpylayers/):

> define a [CVXPY](https://github.com/cvxpy/cvxpy) problem → wrap it as a layer → call it in PyTorch → backprop

### API

#### `FFOLayer(problem, parameters, variables, eps=..., **kwargs)`

**Arguments**
- `problem`: a `cvxpy.Problem` (must satisfy DPP when using CVXPY parameters).
- `parameters`: CVXPY `Parameter` (the inputs to the layer).
- `variables`: CVXPY `Variable` (the outputs returned by the layer).
- `alpha`: perturbation scale (\delta in Eq.4) used in the finite-difference hypergradient approximation. Bigger alpha leads to a smaller delta: less bias, more numerical noise.
- `eps`: solver tolerence for forward pass
- `backward_eps`: solver tolerence for backward pass
- `max_workers`: maximum number of worker threads/processes used to parallelize solver calls 

#### Calling the layer
```python
(outputs,) = layer(*torch_parameters, solver_args={...})
```

**Arguments**
- `*torch_parameters`: PyTorch tensors matching `parameters` in shape (optionally batched).
- `solver_args` (optional): forwarded to `problem.solve(...)` inside CVXPY.

**Return value**
- A tuple of PyTorch tensors corresponding to `variables`.


### Solver-agnostic differentiation!!!

FFOLayer is **solver-agnostic**: it treats the solver as a **black box** and computes hypergradients by re-solving **perturbed** problems, instead of differentiating through solver internals.  
This means you can use **any CVXPY solver** (e.g., GUROBI, MOSEK, ECOS, SCS) without requiring custom backward implementations.

```python
(solution,) = layer(*torch_parameters, solver_args={"solver": cp.GUROBI, "eps": 1e-5})
```



#### Example

```python
import torch

from ffolayer import FFOLayer

batch, n, m = 8, 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

# layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
layer_ffo = FFOLayer(problem, parameters=[A, b], variables=[x], eps=1e-8)
A_tch = torch.randn(batch, m, n, requires_grad=True)
b_tch = torch.randn(batch, m, requires_grad=True)

# solve the problem
(solution,) = layer_ffo(A_tch, b_tch)

# compute the gradient of the sum of the solution with respect to A, b
solution.sum().backward()
```
---
## Code Structure

- `src/`: core layer implementations (FFOCP / FFOQP variants)
- `synthetic_task/`: synthetic decision-focused learning (QP) benchmark
- `sudoku/`: Sudoku as an optimization layer benchmark
- `baselines/`: reference baselines used in experiments
- `tests/`: basic checks / utilities
- `plot_results_*.ipynb`: notebooks for plotting paper figures

### Variants in this repo

We provide two main variants (same core idea, different specialization):

- **FFOCP:** applies to general convex programs.
- **FFOQP:** specializes to QP layers, exploiting quadratic structure for efficiency.


---
## Reproducing the paper experiments

### 1) Synthetic Qradratic Program (QP)

**Key files**
  - `synthetic_task/main_synthetic.py`: entrypoint for all methods
  - `models.py`: all models' definitions and settings

To run the code, if your cluster supports SLURM, please use:
```bash
sh scripts/loop_synthetic_per_seed.sh
```
If not, please use:
```bash
python synthetic_task/main_synthetic.py --method ffocp_eq --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic.py --method ffoqp_eq --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic.py --method lpgd --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic.py --method cvxpylayer --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic.py --method qpth --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic.py --method bpqp --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic.py --method dqp --ydim 800 --epoch 1 --batch_size 8
```

To plot the results, please run `plot_results_synthetic.ipynb`.


---

### 2) Sudoku task
**Key files**
  - `sudoku/main_sudoku.py`: entrypoint for all methods
  - `models_sudoku.py`: all models' definitions and settings

To run the code, if your cluster supports SLURM, please use:
```bash
sh scripts/loop_sudoku_per_seed.sh
```
If not, please use:
```bash
python sudoku/main_sudoku.py --method ffocp_eq --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method ffoqp_eq --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method lpgd --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method cvxpylayer --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method qpth --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method bpqp --n 3 --epoch 1 --batch_size 8
python sudoku/main_sudoku.py --method dqp --n 3 --epoch 1 --batch_size 8
```

To plot the results, please use
```bash
python sudoku/plot_results.py
```

---

### 3) `Synthetic Second-order Cone Progrem (SOCP))`
**Key files**
- `synthetic_task/main_synthetic_general.py`: entrypoint for all methods
- `models.py`: all models' definitions and settings

To run the code, if your cluster supports SLURM, please use:
```bash
sh scripts/loop_synthetic_general_per_seed.sh
```
If not, please use:
```bash
python synthetic_task/main_synthetic_general.py --method ffocp_eq --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic_general.py --method lpgd --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic_general.py --method cvxpylayer --ydim 800 --epoch 1 --batch_size 8
python synthetic_task/main_synthetic_general.py --method bpqp --ydim 800 --epoch 1 --batch_size 8
```

To plot the results, please run `plot_results_synthetic.ipynb`.

---

