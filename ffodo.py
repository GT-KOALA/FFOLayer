import torch
import torch.nn as nn
import numpy as np
import scipy
from torch.autograd import Function
from cvxpylayers.torch import CvxpyLayer


class ffodo(torch.nn.Module):
    def __init__(self, func, ineq_constraints, n, z0=None, gamma0=None, method=None, bounds=None, maxiter=1000, lamb=100):
        super(ffodo, self).__init__()
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.n = n
        self.z0 = z0
        self.gamma0 = gamma0
        self.method = method
        self.bounds = bounds
        self.maxiter = maxiter
        self.lamb = lamb
        # primal dual gradient descent

        pass

    def forward(self, *params, solver_args={}):
        print('z0', self.z0)
        f = _ffodoFn(
            func=self.func, 
            ineq_constraints=self.ineq_constraints, 
            n=self.n,
            z0=self.z0,
            gamma0=self.gamma0,
            maxiter=self.maxiter
            )
        sol = f(*params)
        return sol

def _ffodnCvxpyFn(
        func,
        ineq_constraints, # list of inequality constraints (each should be a callable that takes y and z as inputs)
        n,                # dimensionality of the primal decision variable
        z0,               # initialization of the primal variable (z)
        gamma0,           # initialization of the dual variable (multipliers)
        maxiter           # maximum number of iterations for the optimization
        ):
    class _ffodnCvxpyFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            pass

        @staticmethod
        def backward(ctx, grad_output):
            # This is where you would compute the gradients for backpropagation
            # For now, we will just return None since we are not using gradients in this example
            pass

    return _ffodnCvxpyFnFn.apply

def _ffodoFn(
        func,             # objective function to minimize (should be a callable that takes y and z as inputs)
        ineq_constraints, # list of inequality constraints (each should be a callable that takes y and z as inputs)
        n,                # dimensionality of the primal decision variable
        z0,               # initialization of the primal variable (z)
        gamma0,           # initialization of the dual variable (multipliers)
        maxiter           # maximum number of iterations for the optimization
        ):
    class _ffodoFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            n_constraints = len(ineq_constraints)
            # params_numpy = [to_numpy(p) for p in params]
            # primal dual gradient descent
            # if z0 is not None:
            #     z0 = torch.tensor(z0, requires_grad=True) # Primal initialization
            # else:
            # z0 = torch.rand(n, requires_grad=True).detach().numpy() # Random primal initialization
            # constraints = ({'type': 'ineq', 'fun': lambda z: ineq_constraint(params_numpy, z)} for ineq_constraint in ineq_constraints)
            # sol = scipy.optimize.minimize(fun=lambda z: func(params_numpy, z),  # Objective function to minimize
            #                               x0=z0,  # Initial guess for the primal variable
            #                               constraints=constraints,  # Inequality constraints
            #                               method='COBYLA',
            #                               options={'maxiter': maxiter})

            # print('optimal solution:', sol.x)
            # print('optimal value:', func(params_numpy, sol.x))

            # The iterative gradient descent method. This is not very stable
            with torch.enable_grad():
                n_constraints = len(ineq_constraints)
                # primal dual gradient descent
                if z0 is not None:
                    z = torch.tensor(z0).requires_grad_()  # Primal initialization
                else:
                    z = torch.rand((n,1), requires_grad=True) # Random primal initialization

                if gamma0 is not None:
                    multipliers = torch.tensor(gamma0, requires_grad=True) # Dual initialization
                else:
                    multipliers = torch.rand(n_constraints, requires_grad=True) # Random dual initialization

                optimizer_primal = torch.optim.Adam([z], lr=0.01)
                optimizer_dual = torch.optim.Adam([multipliers], lr=0.001)

                for i in range(maxiter):
                    obj = func(params, z)  # Objective function
                    constraint_violations = [ineq_constraint(params, z) for ineq_constraint in ineq_constraints]  # Evaluate constraints
                    lagrangian = obj + sum([constraint_violations[i] * multipliers[i] for i in range(len(constraint_violations))]) - torch.norm(multipliers)**2  # Lagrangian

                    lagrangian.backward()  # Backpropagation to compute gradients
                    optimizer_primal.step()

                    multipliers.grad = - multipliers.grad # Update multipliers with negative gradient
                    optimizer_dual.step()

                    multipliers.data.clamp_(min=0)  # Ensure multipliers are non-negative (for inequality constraints)
                    print('iteration', i, obj)

            return z.detach(), multipliers.detach()  # Return the optimized primal variable

        @staticmethod
        def backward(ctx, grad_output):
            pass

    return _ffodoFnFn.apply

def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()
