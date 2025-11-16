import torch
import cvxpy as cp
import numpy as np

from cvxpylayers.torch import CvxpyLayer

from ffocp_eq_cone import BLOLayer

def test_soc_blolayer_vs_cvxpy():
    torch.manual_seed(0)

    n = 100
    n_eq_constraints   = 0
    n_ineq_constraints = 50
    Q = torch.eye(n)
    q = torch.rand(n)
    q.requires_grad_(True)
    A = torch.empty(n_eq_constraints, n)
    b = torch.empty(n_eq_constraints)
    G = torch.randn(n_ineq_constraints, n)
    h = torch.randn(n_ineq_constraints)
    
    Q_cp = cp.Parameter((n, n), PSD=True)
    q_cp = cp.Parameter(n)
    # A_cp = cp.Parameter((n_eq_constraints, n))
    # b_cp = cp.Parameter(n_eq_constraints)
    G_cp = cp.Parameter((n_ineq_constraints, n))
    h_cp = cp.Parameter(n_ineq_constraints)
    z_cp = cp.Variable(n)
    
    optimizer = torch.optim.SGD([q], lr=0.1)

    objective_fn = 0.5 * cp.sum_squares(Q_cp @ z_cp) + q_cp.T @ z_cp
    constraints = [G_cp @ z_cp <= h_cp, cp.SOC(1.0, z_cp)]

    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    assert problem.is_dpp()

    cvx_layer = CvxpyLayer(problem, parameters=[Q_cp, q_cp, G_cp, h_cp], variables=[z_cp])
    blolayer = BLOLayer(problem, parameters=[Q_cp, q_cp, G_cp, h_cp], variables=[z_cp], compute_cos_sim=False)

    sol_cvx, = cvx_layer(Q, q, G, h)
    loss_cvx = sol_cvx.sum()
    loss_cvx.backward()
    grad_cvx = q.grad.detach().clone()
    optimizer.zero_grad()

    print("CvxpyLayer gradient:", grad_cvx)

    sol_blo, = blolayer(Q, q, G, h)
    loss_blo = sol_blo.sum()
    loss_blo.backward()
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
