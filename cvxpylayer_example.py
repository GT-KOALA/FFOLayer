import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

n, m = 2, 3
x = cp.Variable(n)
y = cp.Variable(n)
A = cp.Parameter((m, n))
C = cp.Parameter((m, n))
b = cp.Parameter(m)
batch_size = 10
constraints = [x >= 0, y >=0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x + C @ y - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[A, C, b], variables=[x, y])
A_tch = torch.randn(batch_size, m, n, requires_grad=True)
C_tch = torch.randn(batch_size, m, n, requires_grad=True)
b_tch = torch.randn(batch_size, m, requires_grad=True)

# solve the problem
solution, = cvxpylayer(A_tch, C_tch, b_tch)

# compute the gradient of the sum of the solution with respect to A, b
solution.sum().backward()
