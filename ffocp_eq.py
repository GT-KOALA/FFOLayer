import time
import cvxpy as cp
import numpy as np

import torch
from cvxtorch import TorchExpression

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

    def __init__(self, objective, equality_functions, inequality_functions, parameters, variables, alpha):
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
        self.alpha = alpha

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
            alpha=self.alpha,
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
        alpha,
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
            # print('equality_constraints', equality_constraints)
            # print('inequality_constraints', inequality_constraints)

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

                problem.solve(solver=cp.GUROBI)
                # sol_i = [v.value for v in variables]
                # equality_dual_i = [
                #     c.dual_value for c in equality_constraints
                # ]
                # inequality_dual_i = [
                #     c.dual_value for c in inequality_constraints
                # ]

                # convert to torch tensors and incorporate info_forward
                for v_id, v in enumerate(variables):
                    sol_numpy[v_id].append(v.value[np.newaxis,:])
                    sol[v_id].append(to_torch(v.value, ctx.dtype, ctx.device).unsqueeze(0))

                for c_id,c in enumerate(equality_constraints):
                    equality_dual[c_id].append(c.dual_value[np.newaxis,:])

                for c_id,c in enumerate(inequality_constraints):
                    inequality_dual[c_id].append(c.dual_value[np.newaxis,:])

            for v_id in range(len(variables)):
                sol[v_id] = torch.cat(sol[v_id])
                sol_numpy[v_id] = np.concatenate(sol_numpy[v_id])
            for c_id in range(len(equality_constraints)):
                equality_dual[c_id] = np.concatenate(equality_dual[c_id])
            for c_id in range(len(inequality_constraints)):
                inequality_dual[c_id] = np.concatenate(inequality_dual[c_id])

            ctx.sol = sol
            ctx.sol_numpy = sol_numpy
            ctx.equality_dual = equality_dual
            ctx.inequality_dual = inequality_dual
            ctx.params_numpy = params_numpy
            ctx.params = params
            # sol = torch.cat(sol, dim=0)

            return tuple(sol)

        @staticmethod
        def backward(ctx, *dvars):
            # convert to numpy arrays
            dvars_numpy = [to_numpy(dvar) for dvar in dvars]
            
            temperature = 10

            sol = ctx.sol_numpy
            equality_dual = ctx.equality_dual
            inequality_dual = ctx.inequality_dual
            inequality_dual_tanh = [np.tanh(dual * temperature) for dual in inequality_dual]
            params_numpy = ctx.params_numpy
            params = ctx.params
            batch = ctx.batch
            batch_size = ctx.batch_size

            equality_constraints = [equality_function == 0 for equality_function in equality_functions]
            print('sol', sol)
            print('equality_dual', equality_dual)
            print('inequality_dual', inequality_dual)

            sol_lagrangian = [[] for v in variables]
            grad_numpy = [[] for _ in param_order]

            dvar_params = [cp.Parameter(shape=v.shape) for v in variables]
            inequality_dual_params = [cp.Parameter(shape=v.shape) for v in inequality_functions]
            equality_dual_params = [cp.Parameter(shape=v.shape) for v in equality_functions]

            active_mask_params = [cp.Parameter(shape=f.shape) for f in inequality_functions]

            vars_dvars_product = cp.sum([cp.sum(cp.multiply(dvar, v)) for dvar, v in zip(dvar_params, variables)])
            ineq_constraints_dual_product = cp.sum([cp.sum(cp.multiply(dual, ineq)) for dual, ineq in zip(inequality_dual_params, inequality_functions)])
            equality_dual_product = cp.sum([cp.sum(cp.multiply(dual, eq)) for dual, eq in zip(equality_dual_params, equality_functions)])

            # should only contain the equality constraints
            new_objective = 1 / alpha * cp.sum(vars_dvars_product) + objective

            # “Activate” only those inequality entries where mask==1:
            # mask * ineq == 0  -> forces ineq==0 when mask=1; becomes 0==0 when mask=0.
            active_ineq_constraints = [
                cp.multiply(active_mask_params[j], inequality_functions[j]) == 0
                for j in range(len(inequality_functions))
            ]

            problem = cp.Problem(cp.Minimize(new_objective), constraints=equality_constraints + active_ineq_constraints)

            for i in range(batch_size):
                for j, _ in enumerate(param_order):
                    param_order[j].value = params_numpy[j][i]

                for j, _ in enumerate(variables):
                    dvar_params[j].value = dvars_numpy[j][i]

                for j, _ in enumerate(inequality_functions):
                    # key for bilevel algorithm: identify the active constraints and add them to the equality constraints
                    lam = inequality_dual[j][i]
                    active_mask_params[j].value = (lam > 1e-3).astype(np.float64)

                for j, _ in enumerate(equality_functions):
                    equality_dual_params[j].value = equality_dual[j][i]

                problem.solve(solver=cp.GUROBI)
                sol_i_lagrangian = np.array([v.value for v in variables])
                sol_i = np.array([sol[j][i] for j in range(len(variables))])
                print('batch index', i)
                print(problem)
                print('forward solve solution', sol_i)
                print('backward solve solution', sol_i_lagrangian)
                print('solution distance', np.linalg.norm(sol_i_lagrangian - sol_i))
                for j, v in enumerate(variables):
                    sol_lagrangian[j].append(v.value[np.newaxis,:])

            new_sol = [to_torch(np.concatenate(v), ctx.dtype, ctx.device) for v in sol_lagrangian]
            new_inequality_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in inequality_dual]
            new_equality_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in equality_dual]

            finite_difference_obj = objective + equality_dual_product + ineq_constraints_dual_product

            _torch_exp = TorchExpression(
                finite_difference_obj,
                provided_vars_list=[*variables, *param_order, *equality_dual_params, *inequality_dual_params]
            ).torch_expression

            params_req = [p.detach().clone().requires_grad_(True) for p in params if p.requires_grad]

            def slice_params_for_batch(params_req, batch_sizes, i):
                """Pick p[i] if that parameter was batched; else p."""
                out = []
                for p, bs in zip(params_req, ctx.batch_sizes):
                    out.append(p[i] if bs > 0 else p)
                return out

            loss = 0.0
            with torch.enable_grad():
                for i in range(ctx.batch_size):
                    vars_new_i = [v[i] for v in new_sol]
                    vars_old_i = [to_torch(v[i], ctx.dtype, ctx.device) for v in sol]
                    
                    params_i = slice_params_for_batch(params_req, ctx.batch_sizes, i)

                    new_eq_dual_i   = [d[i] for d in new_equality_dual_torch]
                    new_ineq_dual_i = [d[i] for d in new_inequality_dual_torch]
                    old_eq_dual_i   = [to_torch(d[i], ctx.dtype, ctx.device) for d in equality_dual]
                    old_ineq_dual_i = [to_torch(d[i], ctx.dtype, ctx.device) for d in inequality_dual]

                    new_val_i = _torch_exp(*vars_new_i, *params_i, *new_eq_dual_i, *new_ineq_dual_i)
                    old_val_i = _torch_exp(*vars_old_i, *params_i, *old_eq_dual_i, *old_ineq_dual_i)
                    
                    loss += new_val_i - old_val_i

                loss = alpha * loss

            # grads_req = torch.autograd.grad(
            #     loss, params_req, allow_unused=True, retain_graph=False, create_graph=False
            # )

            loss.backward()

            grads = [p.grad for p in params_req]

            # convert to torch tensors and incorporate info_backward
            # grad = [to_torch(g, ctx.dtype, ctx.device) for g in grad_numpy]
            # info.update(info_backward)
            return tuple(grads)

    return _BLOLayerFnFn.apply
