import ffoqp
import ffoqp_eq_cst
import torch
import numpy as np
import unittest
# import qpth
# from qpthlocal.qp import QPSolvers
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import matplotlib.pyplot as plt

# SEED = 25

def test_ffoqp_equality(eps_grid):
    # torch.manual_seed(SEED)
    # Create a sample optimization problem
    n = 100
    n_eq_constraints   = 5
    n_ineq_constraints = 50
    Q = torch.eye(n)
    q = torch.rand(n)
    q.requires_grad_(True)
    A = torch.randn(n_eq_constraints, n)
    b = torch.randn(n_eq_constraints)
    G = torch.randn(n_ineq_constraints, n)
    h = torch.randn(n_ineq_constraints)

    optimizer = torch.optim.SGD([q], lr=0.1)

    # Create an instance of the FFOQP class
    # solver = QPSolvers.PDIPM_BATCHED
    # solver = QPSolvers.CVXPY
    solver = 'OSQP'
    solver_opts = dict(verbose=False, max_iter=10000, adaptive_rho_interval=25,
                       warm_starting=False, polishing=False)

    # Create an instance of the CVXPY layer
    Q_cp = cp.Parameter((n, n), PSD=True)
    q_cp = cp.Parameter(n)
    A_cp = cp.Parameter((n_eq_constraints, n))
    b_cp = cp.Parameter(n_eq_constraints)
    G_cp = cp.Parameter((n_ineq_constraints, n))
    h_cp = cp.Parameter(n_ineq_constraints)
    z_cp = cp.Variable(n)

    # Define the objective and constraints for CVXPY
    objective_fn = 0.5 * cp.sum_squares(Q_cp @ z_cp) + q_cp.T @ z_cp
    constraints = [G_cp @ z_cp <= h_cp, A_cp @ z_cp == b_cp]

    # Create the CVXPY problem
    # print('calling cvxpylaer...')
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    # print("problem is dpp:", problem.is_dpp())
    # assert(problem.is_dpp())

    # Cvxpylayer
    layer = CvxpyLayer(problem, parameters=[Q_cp, q_cp, G_cp, h_cp, A_cp, b_cp], variables=[z_cp])
    sol = layer(Q, q, G, h, A, b)
    z_sol = sol[0]

    loss = torch.sum(z_sol)
    loss.backward()
    
    cvxpylayer_grad = q.grad.clone().detach()
    optimizer.zero_grad()

    def _grad(ffoqp_instance):
      # Forward pass through the FFOQP instance
      z = ffoqp_instance(Q, q, G, h, A, b)
      # print('solution', z)

      loss = torch.sum(z)
      loss.backward(retain_graph=True)
      ffoqp_grad = q.grad.clone().detach()
      optimizer.zero_grad()
      return ffoqp_grad

    def _diff(ffoqp_grad):
      return torch.norm(ffoqp_grad - cvxpylayer_grad, p=1).item()

    def _corr(ffoqp_grad):
      return torch.nn.functional.cosine_similarity(ffoqp_grad, cvxpylayer_grad, dim=0).item()

    cand_cls = {
        'ffoqp': lambda solver, solver_opts:
          ffoqp.ffoqp(lamb=100, solver=solver, solver_opts=solver_opts),
        'ffoqp_eq_cst': lambda solver, solver_opts:
          ffoqp_eq_cst.ffoqp(alpha=100, solver=solver, solver_opts=solver_opts),
    }
    grad_diffs = {k: [] for k in cand_cls.keys()}
    grad_corrs = {k: [] for k in cand_cls.keys()}
    for eps in eps_grid:
      solver_opts["eps_abs"] = eps
      solver_opts["eps_rel"] = eps
      # solver_opts["verbose"] = True
      for k, cls in cand_cls.items():
        ffoqp_instance = cls(solver, solver_opts)
        ffoqp_grad = _grad(ffoqp_instance)
        grad_diffs[k].append(_diff(ffoqp_grad))
        grad_corrs[k].append(_corr(ffoqp_grad))

    grad_diffs = {k: np.array(v) for k, v in grad_diffs.items()}
    grad_corrs = {k: np.array(v) for k, v in grad_corrs.items()}
    return grad_diffs, grad_corrs


if __name__ == "__main__":
    eps_grid = np.logspace(1, 5, base=0.1)
    grad_diffs = {}
    grad_corrs = {}
    for i in range(10):
        grad_diffs_i, grad_corrs_i = test_ffoqp_equality(eps_grid)
        if i == 0:
            grad_diffs, grad_corrs = grad_diffs_i, grad_corrs_i
        else:
            for k in grad_diffs.keys():
                grad_diffs[k] += (grad_diffs_i[k] - grad_diffs[k]) / (i + 1)
                grad_corrs[k] += (grad_corrs_i[k] - grad_corrs[k]) / (i + 1)


    # plotit
    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.set_title('Gradient difference')
    for k, v in grad_diffs.items():
        ax.plot(eps_grid, v, label=k)
    ax.set_xscale('log')
    ax.legend()

    ax2.set_title('Cosine similarity')
    for k, v in grad_corrs.items():
        ax2.plot(eps_grid, v, label=k)
    ax2.set_xscale('log')
    ax2.legend()
    plt.tight_layout()
    plt.show()
