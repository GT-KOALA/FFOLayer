# import ffodo
import ffoqp
import torch
import numpy as np
import unittest
# import qpth
# from qpthlocal.qp import QPSolvers
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from qpth.qp import QPFunction
import qpth
import time

# SEED = 25

class TestFFOQP(unittest.TestCase):
    # def test_ffodo(self):
    #     # Create a sample optimization problem
    #     # torch.manual_seed(SEED)
    #     n = 100
    #     n_ineq_constraints = 50
    #     Q = 0.01 * torch.eye(n) # .numpy()
    #     q = torch.rand(n) # .numpy()
    #     G = torch.rand(n_ineq_constraints, n) #.numpy()
    #     h = torch.rand(n_ineq_constraints) # .numpy()
    #     def func(params, z):
    #         obj = 0.5 * z.T @ Q @ z + z.T @ params[0]  # Objective function: 0.5 * z^T Q z + params^T z
    #         return obj
    #     # func = lambda params, z: z.t() @ Q @ z + z.t() @ params
    #     ineq_constraints = [
    #         lambda params, z: h[i] - G[i] @ z for i in range(n_ineq_constraints)
    #     ]

    #     # Create an instance of the FFODO class
    #     ffodo_instance = ffodo.ffodo(func=func, ineq_constraints=ineq_constraints, n=n)

    #     # Forward pass through the FFODO instance
    #     y = torch.tensor(q)
    #     z = ffodo_instance(y)

    #     print(z)

    def test_ffoqp(self):
        # torch.manual_seed(SEED)
        # Create a sample optimization problem
        
        n = 500
        n_eq_constraints   = 0
        n_ineq_constraints = 50
        Q = torch.eye(n)
        q = torch.rand(n)
        q.requires_grad_(True)
        A = torch.empty(n_eq_constraints, n)
        b = torch.empty(n_eq_constraints)
        G = torch.randn(n_ineq_constraints, n)
        h = torch.randn(n_ineq_constraints)

        optimizer = torch.optim.SGD([q], lr=0.1)

        # Create an instance of the FFOQP class
        # solver = QPSolvers.PDIPM_BATCHED
        # solver = QPSolvers.CVXPY
        ffoqp_instance = ffoqp.ffoqp(lamb=100)

        # Forward pass through the FFOQP instance
        start_time = time.time()
        z = ffoqp_instance(Q, q, G, h, A, b)
        # print('solution', z)
        end_time = time.time()
        print(f'ffoqp forward time: {end_time - start_time}')

        print('backpropagating qp...')
        start_time = time.time()
        loss = torch.sum(z)
        loss.backward(retain_graph=True)
        end_time = time.time()
        print(f'ffoqp backward time: {end_time - start_time}')
        ffoqp_grad = q.grad.clone().detach()
        optimizer.zero_grad()

        # Create an instance of the CVXPY layer
        Q_cp = cp.Parameter((n, n), PSD=True)
        q_cp = cp.Parameter(n)
        # A_cp = cp.Parameter((n_eq_constraints, n))
        # b_cp = cp.Parameter(n_eq_constraints)
        G_cp = cp.Parameter((n_ineq_constraints, n))
        h_cp = cp.Parameter(n_ineq_constraints)
        z_cp = cp.Variable(n)

        # Define the objective and constraints for CVXPY
        objective_fn = 0.5 * cp.sum_squares(Q_cp @ z_cp) + q_cp.T @ z_cp
        constraints = [G_cp @ z_cp <= h_cp]
        # if n_eq_constraints > 0:
        #     constraints.append(A_cp @ z_cp == b_cp)

        # Create the CVXPY problem
        print('calling cvxpylaer...')
        problem = cp.Problem(cp.Minimize(objective_fn), constraints)
        # print("problem is dpp:", problem.is_dpp())
        # assert(problem.is_dpp())

        # Cvxpylayer
        layer = CvxpyLayer(problem, parameters=[Q_cp, q_cp, G_cp, h_cp], variables=[z_cp])
        start_time = time.time()
        sol = layer(Q, q, G, h)
        z_sol = sol[0]
        end_time = time.time()
        print(f'cvxpylayer forward time: {end_time - start_time}')

        loss = torch.sum(z_sol)
        start_time = time.time()
        loss.backward()
        end_time = time.time()
        print(f'cvxpylayer backward time: {end_time - start_time}')
        
        cvxpylayer_grad = q.grad.clone().detach()
        optimizer.zero_grad()
        # print('ffoqp_grad', ffoqp_grad)
        # print('cvxpylayer_grad', cvxpylayer_grad)
        print('gradient difference', torch.norm(ffoqp_grad - cvxpylayer_grad, p=1))
        print('cosine similarity', torch.nn.functional.cosine_similarity(ffoqp_grad, cvxpylayer_grad, dim=0))

        start_time = time.time()
        out = QPFunction(verbose=False, solver=qpth.qp.QPSolvers.PDIPM_BATCHED)(Q, q, G, h, A, b)
        end_time = time.time()
        print(f'qpth forward time: {end_time - start_time}')
        start_time = time.time()
        loss = torch.sum(out)
        loss.backward()
        end_time = time.time()
        print(f'qpth backward time: {end_time - start_time}')

    def test_ffoqp_equality(self):
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
        ffoqp_instance = ffoqp.ffoqp(lamb=100)

        # Forward pass through the FFOQP instance
        print('solving qp...')
        z = ffoqp_instance(Q, q, G, h, A, b)
        # print('solution', z)

        print('backpropagating qp...')
        loss = torch.sum(z)
        loss.backward(retain_graph=True)
        ffoqp_grad = q.grad.clone().detach()
        optimizer.zero_grad()

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
        print('calling cvxpylaer...')
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
        print('ffoqp_grad', ffoqp_grad)
        print('cvxpylayer_grad', cvxpylayer_grad)
        print('gradient difference', torch.norm(ffoqp_grad - cvxpylayer_grad, p=1))
        print('cosine similarity', torch.nn.functional.cosine_similarity(ffoqp_grad, cvxpylayer_grad, dim=0))

if __name__ == '__main__':
    # Run the tests
    unittest.main()