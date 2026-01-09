import torch
import cvxpy as cp
import numpy as np
import ffolayer.ffocp_eq as ffolayer

def test_example():
    # Create a strongly convex quadratic problem
    # minimize 0.5 * x^T Q x + q^T x
    # subject to A @ x == b
    n = 5
    m = 2

    Q_cp = cp.Parameter((n, n), PSD=True)
    q_cp = cp.Parameter(n)
    A_cp = cp.Parameter((m, n))
    b_cp = cp.Parameter(m)
    
    x = cp.Variable(n)
    
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q_cp) + q_cp.T @ x)
    
    constraints = [A_cp @ x == b_cp]
    
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), "Problem must be DPP-compliant"
    
    torch.manual_seed(1)
    np.random.seed(1)
    
    M = torch.randn(n, n)
    eps = 0.1
    Q_tch = M.T @ M + eps * torch.eye(n)
    Q_tch = Q_tch.double().requires_grad_(True)
    
    q_tch = torch.randn(n, dtype=torch.float64).requires_grad_(True)
    A_tch = torch.randn(m, n, dtype=torch.float64).requires_grad_(True)
    
    # Make sure b is feasible (choose b = A @ x0 for some x0)
    x0 = torch.randn(n, dtype=torch.float64)
    b_tch = (A_tch @ x0).requires_grad_(True)
    
    # Solve using CVXPY directly to verify
    Q_cp.value = Q_tch.detach().numpy()
    q_cp.value = q_tch.detach().numpy()
    A_cp.value = A_tch.detach().numpy()
    b_cp.value = b_tch.detach().numpy()
    
    problem.solve()
    print(f"CVXPY solve status: {problem.status}")
    print(f"Optimal value: {problem.value:.6f}")
    print(f"Optimal x: {x.value}")
    
    x_tch = torch.tensor(x.value, dtype=torch.float64, requires_grad=True)
    loss = (0.5 * (x_tch.T @ Q_tch @ x_tch) + q_tch.T @ x_tch).sum()
    loss.backward()
    
    print("Test completed successfully!")


if __name__ == "__main__":
    test_example()