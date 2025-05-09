import time
import cvxpy as cp
import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("Unable to import torch. Please install at "
                      "https://pytorch.org.")

torch_major_version = int(torch.__version__.split('.')[0])
if torch_major_version < 1:
    raise ImportError("cvxpylayers requires PyTorch >= 1.0; please "
                      "upgrade your installation of PyTorch, which is "
                      "version %s." % torch.__version__)


class BLOLayer(torch.nn.Module):
    """A differentiable convex optimization layer

    A BLOLayer solves a parametrized convex optimization problem given by a
    CVXPY problem. It solves the problem in its forward pass, and it computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass. The CVPXY problem must be a disciplined parametrized
    program.

    Example usage:
        ```
        import cvxpy as cp
        import torch
        from blolayers.torch import BLOLayer

        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        equality_constraints = [x = 0]
        inequality_constriants = [x >= 0
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, equality_constraints + inequality_constraints)
        assert problem.is_dpp()

        blolayer = BLOLayer(problem, parameters=[A, b], variables=[x])
        A_tch = torch.randn(m, n, requires_grad=True)
        b_tch = torch.randn(m, requires_grad=True)

        # solve the problem
        solution, = blolayer(A_tch, b_tch)

        # compute the gradient of the sum of the solution with respect to A, b
        solution.sum().backward()
        ```
    """

    def __init__(self, objective, equality_functions, inequality_functions, parameters, variables, lamb):
        """Construct a BLOLayer

        Args:
          objective: a CVXPY Objective object defining the objective of the
                     problem.
          equality_functions: a list of CVXPY Constraint objects defining the problem.
          inequality_functions: a list of CVXPY Constraint objects defining the problem.
          parameters: A list of CVXPY Parameters in the problem; the order
                      of the Parameters determines the order in which parameter
                      values must be supplied in the forward pass. Must include
                      every parameter involved in problem.
          variables: A list of CVXPY Variables in the problem; the order of the
                     Variables determines the order of the optimal variable
                     values returned from the forward pass.
        """
        super(BLOLayer, self).__init__()
        
        self.objective = objective
        self.equality_functions = equality_functions
        self.inequality_functions = inequality_functions
        self.param_order = parameters
        self.variables = variables
        self.lamb = lamb

    def forward(self, *params, solver_args={}):
        """Solve problem (or a batch of problems) corresponding to `params`

        Args:
          params: a sequence of torch Tensors; the n-th Tensor specifies
                  the value for the n-th CVXPY Parameter. These Tensors
                  can be batched: if a Tensor has 3 dimensions, then its
                  first dimension is interpreted as the batch size. These
                  Tensors must all have the same dtype and device.
          solver_args: a dict of optional arguments, to send to `diffcp`. Keys
                       should be the names of keyword arguments.

        Returns:
          a list of optimal variable values, one for each CVXPY Variable
          supplied to the constructor.
        """
        info = {}
        f = _BLOLayerFn(
            objective=self.objective,
            equality_functions=self.equality_functions,
            inequality_functions=self.inequality_functions,
            param_order=self.param_order,
            variables=self.variables,
            lamb=self.lamb,
            info=info,
        )
        sol = f(*params)
        self.info = info
        return sol


def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()


def to_torch(x, dtype, device):
    # convert numpy array to torch tensor
    return torch.from_numpy(x).type(dtype).to(device)


def _BLOLayerFn(
        objective,
        equality_functions,
        inequality_functions,
        param_order,
        variables,
        lamb,
        info):
    class _BLOLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            # infer dtype, device, and whether or not params are batched
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device

            ctx.batch_sizes = []
            for i, (p, q) in enumerate(zip(params, param_order)):
                # check dtype, device of params
                if p.dtype != ctx.dtype:
                    raise ValueError(
                        "Two or more parameters have different dtypes. "
                        "Expected parameter %d to have dtype %s but "
                        "got dtype %s." %
                        (i, str(ctx.dtype), str(p.dtype))
                    )
                if p.device != ctx.device:
                    raise ValueError(
                        "Two or more parameters are on different devices. "
                        "Expected parameter %d to be on device %s "
                        "but got device %s." %
                        (i, str(ctx.device), str(p.device))
                    )

                # check and extract the batch size for the parameter
                # 0 means there is no batch dimension for this parameter
                # and we assume the batch dimension is non-zero
                if p.ndimension() == q.ndim:
                    batch_size = 0
                elif p.ndimension() == q.ndim + 1:
                    batch_size = p.size(0)
                    if batch_size == 0:
                        raise ValueError(
                            "The batch dimension for parameter {} is zero "
                            "but should be non-zero.".format(i))
                else:
                    raise ValueError(
                        "Invalid parameter size passed in. Expected "
                        "parameter {} to have have {} or {} dimensions "
                        "but got {} dimensions".format(
                            i, q.ndim, q.ndim + 1, p.ndimension()))

                ctx.batch_sizes.append(batch_size)

                # validate the parameter shape
                p_shape = p.shape if batch_size == 0 else p.shape[1:]
                if not np.all(p_shape == param_order[i].shape):
                    raise ValueError(
                        "Inconsistent parameter shapes passed in. "
                        "Expected parameter {} to have non-batched shape of "
                        "{} but got {}.".format(
                                i,
                                q.shape,
                                p.shape))

            ctx.batch_sizes = np.array(ctx.batch_sizes)
            ctx.batch = np.any(ctx.batch_sizes > 0)

            if ctx.batch:
                nonzero_batch_sizes = ctx.batch_sizes[ctx.batch_sizes > 0]
                ctx.batch_size = nonzero_batch_sizes[0]
                if np.any(nonzero_batch_sizes != ctx.batch_size):
                    raise ValueError(
                        "Inconsistent batch sizes passed in. Expected "
                        "parameters to have no batch size or all the same "
                        "batch size but got sizes: {}.".format(
                            ctx.batch_sizes))
            else:
                ctx.batch_size = 1

            # convert to numpy arrays
            params_numpy = [to_numpy(p) for p in params]

            # loop through the batch
            sol             = [[] for v in variables]
            sol_numpy       = [[] for v in variables]
            equality_dual   = [[] for c in equality_functions]
            inequality_dual = [[] for c in inequality_functions]

            equality_constraints = [equality_function == 0 for equality_function in equality_functions]
            inequality_constraints = [inequality_function <= 0 for inequality_function in inequality_functions]
            print('equality_constraints', equality_constraints)
            print('inequality_constraints', inequality_constraints)

            for i in range(ctx.batch_size):
                if ctx.batch:
                    # select the i-th batch element for each parameter
                    params_numpy_i = [
                        p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)
                    ]
                else:
                    params_numpy_i = params_numpy

                for p, q in zip(params_numpy_i, param_order):
                    q.value = p

                problem = cp.Problem(
                    cp.Minimize(objective),
                    constraints=equality_constraints + inequality_constraints
                )

                problem.solve()
                sol_i = [v.value for v in variables]
                equality_dual_i = [
                    c.dual_value for c in equality_constraints
                ]
                inequality_dual_i = [
                    c.dual_value for c in inequality_constraints
                ]

                # convert to torch tensors and incorporate info_forward
                for v_id,v in enumerate(variables):
                    sol_numpy[v_id].append(v.value[np.newaxis,:])
                    sol[v_id].append(to_torch(v.value, ctx.dtype, ctx.device).unsqueeze(0))

                for c_id,c in enumerate(equality_constraints):
                    equality_dual[c_id].append(c.dual_value[np.newaxis,:])

                for c_id,c in enumerate(inequality_constraints):
                    inequality_dual[c_id].append(c.dual_value[np.newaxis,:])

            for v_id in range(len(variables)):
                sol[v_id] = torch.cat(sol[v_id])
                sol_numpy[v_id] = np.concatenate(sol_numpy[v_id])
            for c_id in range(equality_constraints):
                equality_dual[c_id] = np.concatenate(equality_dual[c_id])
            for c_id in range(inequality_constraints):
                inequality_dual[c_id] = np.concatenate(inequality_dual[c_id])

            ctx.sol = sol
            ctx.sol_numpy = sol_numpy
            ctx.equality_dual = equality_dual
            ctx.inequality_dual = inequality_dual
            ctx.params = params_numpy
            # sol = torch.cat(sol, dim=0)

            return tuple([sol[x] for x in sol])

        @staticmethod
        def backward(ctx, *dvars):
            
            # convert to numpy arrays
            dvars_numpy = [to_numpy(dvar) for dvar in dvars]
            
            sol = ctx.sol_numpy
            equality_dual = ctx.equality_dual
            inequality_dual = ctx.inequality_dual
            params = ctx.params
            batch = ctx.batch
            batch_size = ctx.batch_size
            print('sol', sol)
            print('equality_dual', equality_dual)
            print('inequality_dual', inequality_dual)

            inequality_dual_params = [
                cp.Parameter(inequality_dual[c_id][0].shape) for c_id in range(len(inequality_functions))
            ]

            for i in range(batch_size):
                new_objective = objective
                vars_dvars_product = cp.sum(
                        [dvar[i] * v for dvar, v in zip(dvars, variables)]
                        )

                for c_id, ineq in zip(inequality_dual, inequality_functions):
                    print('c_id', c_id)
                    print('inequality_dual[c_id]', inequality_dual[c_id][i])
                    print('ineq', ineq)
                    print('variable', variables)

                ineq_constraints_dual_product = cp.sum(
                        [inequality_dual[c_id][i] * ineq for c_id, ineq in zip(inequality_dual, inequality_functions)]
                        )

                dual_penalty = cp.sum(
                        [np.tanh(ineq_dual[i]) * ineq**2 for ineq_dual, ineq in zip(inequality_dual, inequality_functions)]
                        )

                # objective = 1/lamb * f(x,y) + (g(x,y) + \gamma^\top h(x,y)) + lamb * h(x,y)^2 * tanh(gamma)
                new_objective = 1 / lamb * cp.sum(vars_dvars_product) + \
                        objective + \
                        cp.sum(inequality_dual_params[i] * ineq_constraints_dual_product) + \
                        lamb * cp.sum(dual_penalty)

                problem = cp.Problem(
                    cp.Minimize(new_objective),
                    constraints=equality_constraints
                )

                problem.solve()

            # convert to torch tensors and incorporate info_backward
            grad = [to_torch(g, ctx.dtype, ctx.device) for g in grad_numpy]
            info.update(info_backward)

            return tuple(grad)

    return _BLOLayerFnFn.apply
