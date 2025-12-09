import torch
import cvxpy as cp
import numpy as np
import time
import os

from cvxpylayers.torch import CvxpyLayer
from ffocp_eq_cone_general_not_dpp import BLOLayer

torch.set_default_dtype(torch.double) 

def random_problem(
    n, 
    m, 
    k, 
    p_eq=0, 
    p_ineq=0, 
    margin=1, 
    scale=0.1,
    local_fraction=0.1,
):
    Q = torch.eye(n)
    q = (scale * torch.randn(n)).requires_grad_()

    x_star = torch.randn(n)

    A_list, b_list, c_list, d_list = [], [], [], []

    m_local = int(m * local_fraction)    
    m_global = m - m_local
    for i in range(m_local):
        A = torch.zeros(k, n)
        c = torch.zeros(n)

        start = (i * k) % max(1, (n - k + 1))
        end = start + k
        idx = slice(start, end)

        A_block = scale * torch.randn(k, k)
        c_block = scale * torch.randn(k)

        A[:, idx] = A_block
        c[idx] = c_block

        b = scale * torch.randn(k)

        left = torch.linalg.norm(A @ x_star + b)
        right_base = torch.dot(c, x_star)
        d_val = (left - right_base + margin).item()

        A_list.append(A)
        b_list.append(b)
        c_list.append(c)
        d_list.append(torch.tensor(d_val))

    for i in range(m_global):
        A = scale * torch.randn(k, n)
        b = scale * torch.randn(k)
        c = scale * torch.randn(n)

        left = torch.linalg.norm(A @ x_star + b)
        right_base = torch.dot(c, x_star)
        d_val = (left - right_base + margin).item()

        A_list.append(A)
        b_list.append(b)
        c_list.append(c)
        d_list.append(torch.tensor(d_val))

    if p_eq > 0:
        F = scale * torch.randn(p_eq, n)
        g = F @ x_star
    else:
        F, g = None, None

    if p_ineq > 0:
        H = torch.zeros(p_ineq, n)

        band_width = min(3, n)
        for row in range(p_ineq):
            col_start = row % n
            cols = [(col_start + j) % n for j in range(band_width)]
            H[row, cols] = scale * torch.randn(band_width)

        base = H @ x_star
        h = base + (margin + scale * torch.rand(p_ineq)).abs()
    else:
        H, h = None, None

    return Q, q, A_list, b_list, c_list, d_list, F, g, H, h



def test_soc_blolayer_vs_cvxpy(seed=0):
    torch.manual_seed(seed)

    n = 50
    m = 300
    k = 3
    p_eq = 10
    p_ineq = 10
    
    x_cp = cp.Variable(n)

    Q = torch.eye(n)
    q_cp = cp.Parameter(n)

    # F_cp = cp.Parameter((p, n))
    # g_cp = cp.Parameter(p)

    A_cp = [cp.Parameter((k, n)) for _ in range(m)]
    b_cp = [cp.Parameter(k)     for _ in range(m)]
    c_cp = [cp.Parameter(n)     for _ in range(m)]
    d_cp = [cp.Parameter()      for _ in range(m)]   # scalar

    if p_eq > 0:
        F_cp = cp.Parameter((p_eq, n))
        g_eq_cp = cp.Parameter(p_eq)
    else:
        F_cp, g_eq_cp = None, None

    if p_ineq > 0:
        H_cp = cp.Parameter((p_ineq, n))
        h_cp = cp.Parameter(p_ineq)
    else:
        H_cp, h_cp = None, None

    objective_fn = 0.5 * cp.sum_squares(Q @ x_cp) + q_cp.T @ x_cp

    constraints = []
    for i in range(m):
        constraints.append(
            cp.SOC(c_cp[i] @ x_cp + d_cp[i],
                   A_cp[i] @ x_cp + b_cp[i])
        )

    if p_eq > 0:
        constraints.append(F_cp @ x_cp == g_eq_cp)

    if p_ineq > 0:
        constraints.append(H_cp @ x_cp <= h_cp)

    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    assert problem.is_dpp()

    params_cp = [q_cp] + A_cp + b_cp + c_cp + d_cp + [F_cp, g_eq_cp, H_cp, h_cp]
    cvx_layer = CvxpyLayer(problem, parameters=params_cp, variables=[x_cp])
    blolayer = BLOLayer(problem, parameters=params_cp, variables=[x_cp])

    cpu_threads = os.cpu_count()
    repeat_times = 2
    
    # with torch.no_grad():
    #     start_time = time.time()
    #     # sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.GUROBI, "Threads": cpu_threads})
    #     # sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.SCS, "max_iters": 2000, "eps": 1e-3, "warm_start": True})
    #     # sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.MOSEK})

    #     sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.SCS, "max_iters": 1000, "eps": 1e-4})
    #     print(f"BLOLayer forward time with no grad: {time.time() - start_time}")

    blo_total_fw_time = []
    cvx_total_fw_time = []
    blo_total_bw_time = []
    cvx_total_bw_time = []
    cos_sim_list = []
    l2_diff_list = []
    for _ in range(repeat_times):
        _, q, A_list, b_list, c_list, d_list, F, g, H, h = random_problem(n, m, k, p_eq, p_ineq)
        params_torch = [q] + A_list + b_list + c_list + d_list + [F, g, H, h]
        optimizer = torch.optim.SGD([q], lr=0.1)

        blo_forward_start_time = time.time()
        # sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.GUROBI, "Threads": cpu_threads, "BarConvTol": 1e-4, "FeasibilityTol": 1e-6, "OptimalityTol": 1e-6})
        # sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.GUROBI, "Threads": cpu_threads})
        # sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.MOSEK, "mosek_params": {'MSK_DPAR_OPTIMIZER_MAX_TIME':  1, }})
        # sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.MOSEK})
        sol_blo, = blolayer(*params_torch, solver_args={"solver": cp.SCS, "max_iters": 2000, "eps": 1e-8, "warm_start": True})
        blo_forward_end_time = time.time()
        print(f"BLOLayer forward time: {blo_forward_end_time - blo_forward_start_time}")

        loss_blo = sol_blo.sum()
        blo_loss_backward_start_time = time.time()
        loss_blo.backward()
        blo_loss_backward_end_time = time.time()
        print(f"BLOLayer loss backward time: {blo_loss_backward_end_time - blo_loss_backward_start_time}")

        grad_blo = q.grad.detach().clone()
        optimizer.zero_grad()

        blo_total_fw_time.append(blo_forward_end_time - blo_forward_start_time)
        blo_total_bw_time.append(blo_loss_backward_end_time - blo_loss_backward_start_time)
        
        cvx_forward_start_time = time.time()
        sol_cvx, = cvx_layer(*params_torch, solver_args={"eps": 1e-10})
        cvx_forward_end_time = time.time()
        print(f"CvxpyLayer forward time: {cvx_forward_end_time - cvx_forward_start_time}")

        loss_cvx = sol_cvx.sum()
        cvx_loss_backward_start_time = time.time()
        loss_cvx.backward()
        cvx_loss_backward_end_time = time.time()
        print(f"CvxpyLayer loss backward time: {cvx_loss_backward_end_time - cvx_loss_backward_start_time}")

        grad_cvx = q.grad.detach().clone()
        optimizer.zero_grad()

        # print("CvxpyLayer gradient:", grad_cvx)
        cvx_total_fw_time.append(cvx_forward_end_time - cvx_forward_start_time)
        cvx_total_bw_time.append(cvx_loss_backward_end_time - cvx_loss_backward_start_time)


        est = grad_blo.reshape(-1)
        gt  = grad_cvx.reshape(-1)

        eps = 1e-12
        denom = (est.norm() * gt.norm()).clamp_min(eps)
        cos_sim = torch.dot(est, gt) / denom
        l2_diff = (est - gt).norm()

        cos_sim_list.append(cos_sim)
        l2_diff_list.append(l2_diff)

    print(f"CvxpyLayer total forward time: {cvx_total_fw_time}")
    print(f"CvxpyLayer total backward time: {cvx_total_bw_time}")
    print(f"CvxpyLayer mean forward time: {np.mean(cvx_total_fw_time[1:])}")
    print(f"CvxpyLayer mean backward time: {np.mean(cvx_total_bw_time[1:])}")
    print(f"--------------------------------")
    print(f"BLOLayer total forward time: {blo_total_fw_time}")
    print(f"BLOLayer total backward time: {blo_total_bw_time}")
    print(f"BLOLayer mean forward time: {np.mean(blo_total_fw_time[1:])}")
    print(f"BLOLayer mean backward time: {np.mean(blo_total_bw_time[1:])}")

    print(f"cosine similarity: {cos_sim_list}")

if __name__ == "__main__":
    for seed in range(1):
        test_soc_blolayer_vs_cvxpy(seed)