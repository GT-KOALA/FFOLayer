import torch
import numpy as np
import pyepo

from qpthlocal.qp import QPFunction, QPSolvers

def df_loss_fn(y_pred, y):
    # y: (batch_size, n)
    batch_size, n = y.shape

    # QP formulation
    Q = torch.eye(n)
    p = y_pred
    G = torch.cat([torch.eye(n), -torch.eye(n), torch.ones(1,n)], dim=0)
    h = torch.cat([torch.zeros(n), torch.ones(n), torch.Tensor([3])], dim=0)

    # Solve QP
    y = y.view(-1, n)
    Q = Q.unsqueeze(0).expand(batch_size, -1, -1)
    G = G.unsqueeze(0).expand(batch_size, -1, -1)
    h = h.unsqueeze(0).expand(batch_size, -1)
    A = torch.Tensor()
    b = torch.Tensor()
    
    # qpth solver
    sol = QPFunction(solver=QPSolvers.CVXPY, verbose=False)(Q, p, G, h, A, b)

    # Compute loss
    loss = torch.mean(torch.bmm(sol.unsqueeze(1), y.unsqueeze(2)).squeeze(2))
    return loss

def food_loss_fn(y_pred, y, alpha=0.01):
    # y: (batch_size, n)
    # y_pred: (batch_size, n)
    # alpha: accuracy of the approximation
    batch_size, n = y.shape

    # QP formulation
    Q = torch.eye(n)
    p = y_pred.detach() # size (batch_size, n)
    G = torch.cat([torch.eye(n), -torch.eye(n), torch.ones(1,n)], dim=0)
    h = torch.cat([torch.zeros(n), torch.ones(n), torch.Tensor([3])], dim=0)
    n_constraints = G.shape[0]
    
    Q = Q.unsqueeze(0).expand(batch_size, -1, -1) # size (batch_size, n, n)
    G = G.unsqueeze(0).expand(batch_size, -1, -1) # size (batch_size, n_constraints, n)
    h = h.unsqueeze(0).expand(batch_size, -1)     # size (batch_size, n_constraints)

    obj = lambda z: 0.5 * torch.bmm(z.unsqueeze(1), torch.bmm(Q, z.unsqueeze(2))).squeeze(2) + torch.bmm(p.unsqueeze(1), z.unsqueeze(2)).squeeze(2)
    obj_grad = lambda z: 0.5 * torch.bmm(z.unsqueeze(1), torch.bmm(Q, z.unsqueeze(2))).squeeze(2) + torch.bmm(y_pred.unsqueeze(1), z.unsqueeze(2)).squeeze(2)
    constraints = lambda z: torch.bmm(G, z.unsqueeze(2)).squeeze(2) - h

    # primal dual gradient descent
    z = torch.rand(batch_size, n, requires_grad=True) # Random primal initialization
    multiplier = torch.rand(batch_size, n_constraints, requires_grad=True) # Random dual initialization

    optimizer_primal = torch.optim.Adam([z], lr=0.01)
    optimizer_dual = torch.optim.Adam([multiplier], lr=0.01)
    
    num_iterations = 100
    for iteration in range(num_iterations):
        objective = obj(z)
        constraint_violation = constraints(z)
        # print(iteration, torch.mean(objective), torch.mean(torch.clamp(constraint_violation, min=0)))
        lagrangian = objective + torch.bmm(multiplier.unsqueeze(1), constraint_violation.unsqueeze(2)).squeeze(2)

        # print(obj(z), constraints(z))
        # primal update
        torch.sum(lagrangian).backward()
        optimizer_primal.step()

        multiplier.grad = - multiplier.grad
        optimizer_dual.step()
        
        optimizer_primal.zero_grad()
        optimizer_dual.zero_grad()

        multiplier.data = torch.clamp(multiplier.data, min=0) # Ensure dual feasibility

    z_opt = z.detach()
    multiplier_opt = multiplier.detach()
    # gradient descent for unconstrained lagrangian minimization
    eps = 1e-4
    z_lambda = torch.rand(batch_size, n, requires_grad=True) # Random primal initialization
    optimizer = torch.optim.Adam([z_lambda], lr=0.01)
    for iteration in range(num_iterations):
        external_objective = torch.bmm(y.unsqueeze(1), z_lambda.unsqueeze(2)).squeeze(2) # f(y,z) = y @ z_lambda
        lagrangian_unconstrained = alpha * (obj(z_lambda) + torch.bmm(multiplier_opt.unsqueeze(1), constraints(z_lambda).unsqueeze(2)).squeeze(2) - obj(z_opt)) 
        constraint_correction = 0.5 * alpha ** 2 * torch.norm(constraints(z_lambda) * (multiplier_opt > eps), dim=-1)
        total_objective = external_objective + lagrangian_unconstrained + constraint_correction

        torch.sum(total_objective).backward()
        optimizer.step()
        optimizer.zero_grad()

    z_lambda_opt = z_lambda.detach()

    # Compute loss
    losses = torch.bmm(y.unsqueeze(1), z_lambda_opt.unsqueeze(2)).squeeze(2) \
            + alpha * (obj_grad(z_lambda_opt) + torch.bmm(multiplier_opt.unsqueeze(1), constraints(z_lambda_opt).unsqueeze(2)).squeeze(2) - obj_grad(z_opt)) \
            + 0.5 * alpha ** 2 * torch.norm(constraints(z_lambda_opt) * (multiplier_opt > eps), dim=-1)

    loss = torch.mean(losses)

    return loss
