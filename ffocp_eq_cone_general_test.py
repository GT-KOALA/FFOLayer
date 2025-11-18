import torch
import cvxpy as cp
import numpy as np
import time
import os

from cvxpylayers.torch import CvxpyLayer
from ffocp_eq_cone_general import BLOLayer


def test_soc_blolayer_vs_cvxpy():
    torch.manual_seed(0)

    n = 500
    m = 10          # number of SOC constraints
    k = 10           # each ||A_i x + b_i||_2 has k components (A_i in R^{k x n})
    # p = 3

    Q = torch.eye(n)
    q = torch.rand(n, requires_grad=True)

    A_list = [torch.randn(k, n) for _ in range(m)]
    b_list = [torch.randn(k)     for _ in range(m)]
    c_list = [torch.randn(n)     for _ in range(m)]
    d_list = [torch.randn(1)     for _ in range(m)]

    # F = torch.randn(p, n)
    # g = torch.randn(p)

    optimizer = torch.optim.SGD([q], lr=0.1)

    x_cp = cp.Variable(n)

    Q_cp = cp.Parameter((n, n), PSD=True)
    q_cp = cp.Parameter(n)

    # F_cp = cp.Parameter((p, n))
    # g_cp = cp.Parameter(p)

    A_cp = [cp.Parameter((k, n)) for _ in range(m)]
    b_cp = [cp.Parameter(k)     for _ in range(m)]
    c_cp = [cp.Parameter(n)     for _ in range(m)]
    d_cp = [cp.Parameter()      for _ in range(m)]   # scalar

    objective_fn = 0.5 * cp.sum_squares(Q_cp @ x_cp) + q_cp.T @ x_cp

    # constraints: ||A_i x + b_i||_2 <= c_i^T x + d_i,  i = 1..m
    constraints = []
    for i in range(m):
        constraints.append(
            cp.SOC(c_cp[i] @ x_cp + d_cp[i],
                   A_cp[i] @ x_cp + b_cp[i])
        )

    # constraints.append(F_cp @ x_cp == g_cp)

    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    assert problem.is_dpp()

    params_cp = [Q_cp, q_cp] + A_cp + b_cp + c_cp + d_cp
    cvx_layer = CvxpyLayer(problem, parameters=params_cp, variables=[x_cp])
    blolayer = BLOLayer(problem, parameters=params_cp, variables=[x_cp],
                        compute_cos_sim=False)

    params_torch = [Q, q] + A_list + b_list + c_list + d_list

    start_time = time.time()
    sol_cvx, = cvx_layer(*params_torch)
    end_time = time.time()
    print(f"CvxpyLayer forward time: {end_time - start_time}")

    start_time = time.time()
    loss_cvx = sol_cvx.sum()
    end_time = time.time()
    print(f"CvxpyLayer loss backward time: {end_time - start_time}")

    loss_cvx.backward()
    grad_cvx = q.grad.detach().clone()
    optimizer.zero_grad()

    print("CvxpyLayer gradient:", grad_cvx)

    cpu_threads = os.cpu_count()

    start_time = time.time()
    sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.GUROBI, "Threads": cpu_threads, "BarConvTol": 1e-4, "FeasibilityTol": 1e-6, "OptimalityTol": 1e-6})
    # sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.MOSEK, "mosek_params": {'MSK_DPAR_OPTIMIZER_MAX_TIME':  1, }})
    end_time = time.time()
    print(f"BLOLayer forward time: {end_time - start_time}")

    start_time = time.time()
    loss_blo = sol_blo.sum()
    loss_blo.backward()
    end_time = time.time()
    print(f"BLOLayer loss backward time: {end_time - start_time}")

    grad_blo = q.grad.detach().clone()
    optimizer.zero_grad()

    print("BLOLayer gradient:", grad_blo)

    est = grad_blo.reshape(-1)
    gt  = grad_cvx.reshape(-1)

    eps = 1e-12
    denom = (est.norm() * gt.norm()).clamp_min(eps)
    cos_sim = torch.dot(est, gt) / denom
    l2_diff = (est - gt).norm()

    print(f"cosine similarity: {cos_sim.item():.6f}")
    print(f"L2 difference:     {l2_diff.item():.6e}")


if __name__ == "__main__":
    test_soc_blolayer_vs_cvxpy()
