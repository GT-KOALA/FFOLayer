import time
import cvxpy as cp
import numpy as np
import os

import torch
from cvxtorch import TorchExpression
from cvxpylayers.torch import CvxpyLayer
import wandb
from utils import to_numpy, to_torch, _dump_cvxpy, n_threads, slice_params_for_batch

@torch.no_grad()
def _compare_grads(params_req, grads, ground_truth_grads):
    est_chunks, gt_chunks = [], []
    for p, ge, gg in zip(params_req, grads, ground_truth_grads):
        ge = torch.zeros_like(p) if ge is None else ge.detach()
        gg = torch.zeros_like(p) if gg is None else gg.detach()
        est_chunks.append(ge.reshape(-1))
        gt_chunks.append(gg.reshape(-1))

    est = torch.cat(est_chunks)
    gt  = torch.cat(gt_chunks)

    eps = 1e-12

    denom = (est.norm() * gt.norm()).clamp_min(eps)
    cos_sim = torch.dot(est, gt) / denom

    diff = (est - gt)
    l2_diff = diff.norm()
    # rel_l2_diff = l2_diff / gt.norm().clamp_min(eps)
    return cos_sim, l2_diff

# wrapper function for BLOLayer
def BLOLayer(
    problem: cp.Problem,
    parameters,
    variables,
    alpha: float = 100.0,
    dual_cutoff: float = 1e-3,
    slack_tol: float = 1e-8,
    eps: float = 1e-7,
    compute_cos_sim: bool = False,
):
    """
    Create an optimization layer that can be called like a CvxpyLayer:
        layer = BLOLayer(...);  y = layer(*param_tensors)

    Args:
        problem:   cvxpy.Problem with objective + constraints
        parameters: list[cp.Parameter]
        variables:  list[cp.Variable]
        alpha, dual_cutoff, slack_tol, compute_cos_sim: hyperparameters for BLOLayer

    Returns:
        A module with forward(*params, solver_args={}) -> tuple[tensor,...]
    """

    # Extract objective expression (ensure minimization)
    obj = problem.objective
    if isinstance(obj, cp.Minimize):
        objective_expr = obj.expr
    elif isinstance(obj, cp.Maximize):
        objective_expr = -obj.expr  # convert to minimize
    else:
        objective_expr = getattr(obj, "expr", None)
        if objective_expr is None:
            raise ValueError("Unsupported objective type; expected Minimize/Maximize.")

    eq_funcs, ineq_funcs = [], []
    for c in problem.constraints:
        # Equality: g(x,θ) == 0  -> store g(x,θ)
        if isinstance(c, cp.constraints.zero.Equality):
            eq_funcs.append(c.expr)

        # Inequality: g(x,θ) <= 0 -> store g(x,θ)
        elif isinstance(c, cp.constraints.nonpos.Inequality):
            ineq_funcs.append(c.expr)

        else:
            # save for PSD or SOC constraints
            raise NotImplementedError(
                f"Constraint type {type(c)} not supported in BLOLayer wrapper."
            )

    return _BLOLayer(
        objective=objective_expr,
        eq_functions=eq_funcs,
        ineq_functions=ineq_funcs,
        parameters=parameters,
        variables=variables,
        alpha=alpha,
        dual_cutoff=dual_cutoff,
        slack_tol=slack_tol,
        eps=eps,
        _compute_cos_sim=compute_cos_sim,
    )

class _BLOLayer(torch.nn.Module):
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
        eq_constraints = [x = 0]
        ineq_constriants = [x >= 0
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, eq_constraints + ineq_constraints)
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

    def __init__(self, objective, eq_functions, ineq_functions, parameters, variables, alpha, dual_cutoff, slack_tol, eps, _compute_cos_sim=False):
        """Construct a BLOLayer

        Args:
          objective: a CVXPY Objective object defining the objective of the
                     problem.
          eq_functions: a list of CVXPY Constraint objects defining the problem.
          ineq_functions: a list of CVXPY Constraint objects defining the problem.
          parameters: A list of CVXPY Parameters in the problem; the order
                      of the Parameters determines the order in which parameter
                      values must be supplied in the forward pass. Must include
                      every parameter involved in problem.
          variables: A list of CVXPY Variables in the problem; the order of the
                     Variables determines the order of the optimal variable
                     values returned from the forward pass.
        """
        super(_BLOLayer, self).__init__()
        
        self.objective = objective
        self.eq_functions = eq_functions
        self.ineq_functions = ineq_functions
        self.param_order = parameters
        self.variables = variables
        self.alpha = alpha
        self.dual_cutoff = dual_cutoff
        self.slack_tol = float(slack_tol) 
        self._compute_cos_sim = _compute_cos_sim
        self.eps = eps

        self.eq_constraints = [f == 0 for f in eq_functions]
        self.ineq_constraints = [g <= 0 for g in ineq_functions]
        self.problem = cp.Problem(cp.Minimize(objective), self.eq_constraints + self.ineq_constraints)

        self.dvar_params   = [cp.Parameter(shape=v.shape) for v in variables]
        self.eq_dual_params   = [cp.Parameter(shape=f.shape) for f in eq_functions]
        self.ineq_dual_params = [cp.Parameter(shape=f.shape) for f in ineq_functions]
        self.active_mask_params = [cp.Parameter(shape=f.shape) for f in ineq_functions]

        vars_dvars_product = cp.sum([cp.sum(cp.multiply(dv, v))
                                    for dv, v in zip(self.dvar_params, variables)])
        ineq_dual_product = cp.sum([cp.sum(cp.multiply(lm, g))
                                    for lm, g in zip(self.ineq_dual_params, ineq_functions)])

        self.new_objective = (1/self.alpha) * vars_dvars_product + objective + ineq_dual_product
        self.active_eq_constraints = [
            cp.multiply(self.active_mask_params[j], ineq_functions[j]) == 0
            for j in range(len(ineq_functions))
        ]
        self.perturbed_problem = cp.Problem(cp.Minimize(self.new_objective),
                                        self.eq_constraints + self.active_eq_constraints)

        phi_expr = objective \
            + cp.sum([cp.sum(cp.multiply(du, f)) for du, f in zip(self.eq_dual_params, eq_functions)]) \
            + cp.sum([cp.sum(cp.multiply(du, f)) for du, f in zip(self.ineq_dual_params, ineq_functions)])
        self.phi_torch = TorchExpression(
            phi_expr,
            provided_vars_list=[*variables, *self.param_order, *self.eq_dual_params, *self.ineq_dual_params]
        ).torch_expression

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
        if solver_args is None:
            solver_args = {}
        solver_args = {"warm_start": True, "solver": cp.GUROBI, "Threads": n_threads, "OutputFlag": 0, **solver_args}
        
        info = {}
        f = _BLOLayerFn(
            blolayer=self,
            _compute_cos_sim=self._compute_cos_sim,
            info=info
        )
        sol = f(*params)
        self.info = info
        return sol

def _BLOLayerFn(
        blolayer,
        _compute_cos_sim,
        info):
    class _BLOLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            # infer dtype, device, and whether or not params are batched
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device

            ctx.batch_sizes = []
            for i, (p, q) in enumerate(zip(params, blolayer.param_order)):
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
                if not np.all(p_shape == blolayer.param_order[i].shape):
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
            
            B = ctx.batch_size

            # convert to numpy arrays
            params_numpy = [to_numpy(p) for p in params]

            sol_numpy = [np.empty((B,) + v.shape, dtype=float) for v in blolayer.variables]
            eq_dual = [np.empty((B,) + f.shape, dtype=float) for f in blolayer.eq_functions]
            ineq_dual = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.ineq_functions]
            ineq_slack_residual = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.ineq_functions]

            for i in range(B):
                if ctx.batch:
                    # select the i-th batch element for each parameter
                    params_numpy_i = [
                        p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)
                    ]
                else:
                    params_numpy_i = params_numpy

                for p, q in zip(params_numpy_i, blolayer.param_order):
                    q.value = p

                try:
                    # blolayer.problem.solve(solver=cp.GUROBI, ignore_dpp=True, warm_start=True, **{"Threads": n_threads, "OutputFlag": 0})
                    blolayer.problem.solve(solver=cp.SCS, warm_start=False, ignore_dpp=True, max_iters=2500, eps=blolayer.eps)
                except:
                    print("Forward pass GUROBI failed, using OSQP")
                    blolayer.problem.solve(solver=cp.OSQP, warm_start=False, verbose=False)
                
                # convert to torch tensors and incorporate info_forward
                for v_id, v in enumerate(blolayer.variables):
                    sol_numpy[v_id][i, ...] = v.value

                for c_id, c in enumerate(blolayer.eq_constraints):
                    eq_dual[c_id][i, ...] = c.dual_value

                for c_id, c in enumerate(blolayer.ineq_constraints):
                    ineq_dual[c_id][i, ...] = c.dual_value

                for c_id, expr in enumerate(blolayer.ineq_functions):
                    g_val = expr.value
                    s_val = -g_val
                    s_val = np.maximum(s_val, 0.0)
                    ineq_slack_residual[c_id][i, ...] = s_val

            ctx.sol_numpy = sol_numpy
            ctx.eq_dual = eq_dual
            ctx.ineq_dual = ineq_dual
            ctx.params_numpy = params_numpy
            ctx.params = params
            ctx.slack = ineq_slack_residual
            ctx.blolayer = blolayer

            ctx._warm_vars = [v.value.copy() for v in blolayer.variables]
            ctx._warm_eq_duals = [c.dual_value.copy() for c in blolayer.eq_constraints]
            ctx._warm_ineq_duals = [c.dual_value.copy() for c in blolayer.ineq_constraints]
            ctx._warm_ineq_slack_residuals = [c.value.copy() for c in blolayer.ineq_functions]

            sol_torch = [to_torch(arr, ctx.dtype, ctx.device) for arr in sol_numpy]

            return tuple(sol_torch)

        @staticmethod
        def backward(ctx, *dvars):
            # convert to numpy arrays
            dvars_numpy = [to_numpy(dvar) for dvar in dvars]
            
            # temperature = 10
            # ineq_dual_tanh = [np.tanh(dual * temperature) for dual in ctx.ineq_dual]

            blolayer = ctx.blolayer
            sol_numpy = ctx.sol_numpy
            eq_dual = ctx.eq_dual
            ineq_dual = ctx.ineq_dual
            slack = ctx.slack
            y_dim = dvars_numpy[0].shape[1]
            if len(eq_dual) == 0:
                num_eq = 0
            else:
                num_eq = eq_dual[0].shape[1]
            B = ctx.batch_size

            params_numpy = ctx.params_numpy

            params_req, req_grad_mask = [], []
            for p in ctx.params:
                q = p.detach().clone()
                req_grad = bool(p.requires_grad)
                q.requires_grad_(req_grad)
                params_req.append(q)
                req_grad_mask.append(req_grad)

            if _compute_cos_sim:
                # compute ground truth gradient using cvxpylayer
                cvxpylayer_problem = cp.Problem(cp.Minimize(blolayer.objective),
                            constraints=blolayer.eq_constraints + blolayer.ineq_constraints)

                _cvx_layer = CvxpyLayer(cvxpylayer_problem, parameters=blolayer.param_order, variables=blolayer.variables)
                with torch.enable_grad():
                    sol_tensors = _cvx_layer(*params_req)

                if not isinstance(sol_tensors, (tuple, list)):
                    sol_tensors = (sol_tensors,)

                grad_outputs = [torch.zeros_like(out) if gv is None else gv for out, gv in zip(sol_tensors, dvars)]
                inputs_for_grad = tuple(q for q, need in zip(params_req, req_grad_mask) if need)

                ground_truth_grads = torch.autograd.grad(
                    outputs=tuple(sol_tensors),
                    inputs=inputs_for_grad,
                    grad_outputs=tuple(grad_outputs),
                    allow_unused=True,
                    retain_graph=False
                )

            new_sol_lagrangian = [np.empty_like(sol_numpy[k]) for k in range(len(blolayer.variables))]
            new_eq_dual = [np.empty_like(eq_dual[k]) for k in range(len(blolayer.eq_constraints))]

            new_active_dual = [np.empty((B,) + c.shape, dtype=float) for c in blolayer.active_eq_constraints]
            
            sol_diffs = []

            for i in range(B):
                for j, _ in enumerate(blolayer.param_order):
                    blolayer.param_order[j].value = params_numpy[j][i]

                for j, v in enumerate(blolayer.variables):
                    blolayer.dvar_params[j].value = dvars_numpy[j][i]
                    # ZIHAO ADDED
                    v.value = ctx._warm_vars[j]

                for j, c in enumerate(blolayer.eq_constraints):
                    if c.dual_value is None and ctx._warm_eq_duals[j] is not None:
                        c.dual_value = ctx._warm_eq_duals[j]
                for j, c in enumerate(blolayer.ineq_constraints):
                    if c.dual_value is None and ctx._warm_ineq_duals[j] is not None:
                        c.dual_value = ctx._warm_ineq_duals[j]

                for j, _ in enumerate(blolayer.ineq_functions):
                    # key for bilevel algorithm: identify the active constraints and add them to the equality constraints
                    lam = ineq_dual[j][i]
                    blolayer.ineq_dual_params[j].value = lam
                    
                    # active_mask_params[j].value = ((lam > dual_cutoff)).astype(np.float64)
                    _requires_active = (slack[j][i] <= blolayer.slack_tol).astype(np.float64)
                    blolayer.active_mask_params[j].value = _requires_active

                    # print(f"num active constraints: {active_mask_params[j].value.sum()}")
                    if _requires_active.sum() > y_dim - num_eq:
                        print(f"num active constraints: {_requires_active.sum()}")

                        k = int(y_dim - num_eq)
                        idx = np.argpartition(lam, -k)[-k:]
                        mask = np.zeros_like(lam, dtype=np.float64)
                        mask[idx] = 1.0
                        blolayer.active_mask_params[j].value = mask

                for j, _ in enumerate(blolayer.eq_functions):
                    blolayer.eq_dual_params[j].value = eq_dual[j][i]

                # blolayer.perturbed_problem.solve(solver=cp.GUROBI, ignore_dpp=True, warm_start=True, **{"Threads": n_threads, "OutputFlag": 0})
                blolayer.perturbed_problem.solve(solver=cp.SCS, warm_start=True, ignore_dpp=True, max_iters=2500, eps=blolayer.eps)

                st = blolayer.perturbed_problem.status
                try:
                    if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        raise RuntimeError(f"New bilevel problem status = {st}")
                    for j, v in enumerate(blolayer.variables):
                        new_sol_lagrangian[j][i, ...] = v.value
                        sol_diff = np.linalg.norm(sol_numpy[j][i] - v.value)
                        sol_diffs.append(sol_diff)
                except:
                    print("Backward pass GUROBI failed, using OSQP")
                    blolayer.perturbed_problem.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, warm_start=True, verbose=False)
                    for j, v in enumerate(blolayer.variables):
                        new_sol_lagrangian[j][i, ...] = v.value
                        sol_diff = np.linalg.norm(sol_numpy[j][i] - v.value)
                        sol_diffs.append(sol_diff)
                
                for c_id, c in enumerate(blolayer.eq_constraints):
                    if c.dual_value.any() == None:
                        print(f"equality constraint {c_id} dual value is None")
                    new_eq_dual[c_id][i, ...] = c.dual_value
                for c_id, c in enumerate(blolayer.active_eq_constraints):
                    if c.dual_value.any() == None:
                        print(f"active inequality constraint {c_id} dual value is None")
                    active_mask = np.array([a.value for a in blolayer.active_mask_params])
                    new_active_dual[c_id][i, ...] = c.dual_value
            
            # print('--- sol_diff mean: ', np.mean(np.array(sol_diffs)), 'max: ', np.max(np.array(sol_diffs)), 'min: ', np.min(np.array(sol_diffs)))

            new_sol = [to_torch(v, ctx.dtype, ctx.device) for v in new_sol_lagrangian]
            new_ineq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_active_dual]
            new_eq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_eq_dual]

            # sol_dis = torch.linalg.norm(to_torch(new_sol_lagrangian[0], ctx.dtype, ctx.device) - to_torch(sol_numpy[0], ctx.dtype, ctx.device))
            # print("solution distance: {:.6f}".format(sol_dis))
            # if sol_dis > 0.01:
            #     _dump_cvxpy(
            #         save_dir='./cvxpy_logs',
            #         file_name=f'ffocp_eq_{i}',
            #         batch_i=i,
            #         ctx=ctx,
            #         param_order=blolayer.param_order,
            #         variables=blolayer.variables,
            #         alpha=blolayer.alpha,
            #         dual_cutoff=blolayer.dual_cutoff,
            #         solver_used='gurobi',
            #         trigger='',
            #         params_numpy=params_numpy,
            #         sol_numpy=new_sol_lagrangian,
            #         eq_dual=new_eq_dual,
            #         ineq_dual=new_active_dual,
            #         slack=slack,
            #         new_sol_lagrangian=new_sol_lagrangian,
            #         new_eq_dual=new_eq_dual,
            #         new_active_dual=new_active_dual,
            #         active_mask_params=blolayer.active_mask_params,
            #         dvars_numpy=dvars_numpy,
            #     )
            #     import pdb; pdb.set_trace()

            params_req = []
            for p, need in zip(ctx.params, req_grad_mask):
                q = p.detach().clone()
                if need:
                    q.requires_grad_(True)
                    params_req.append(q)
                else:
                    params_req.append(q)
            if ctx.device != torch.device('cpu'):
                torch.set_default_device(torch.device(ctx.device))
            loss = 0.0
            with torch.enable_grad():
                for i in range(B):
                    vars_new_i = [v[i] for v in new_sol]
                    vars_old_i = [to_torch(sol_numpy[j][i], ctx.dtype, ctx.device) for j in range(len(blolayer.variables))]
                    
                    params_i = slice_params_for_batch(params_req, ctx.batch_sizes, i)

                    new_eq_dual_i = [d[i] for d in new_eq_dual_torch]
                    new_ineq_dual_i = [d[i] for d in new_ineq_dual_torch]
                    old_eq_dual_i = [to_torch(eq_dual[j][i], ctx.dtype, ctx.device) for j in range(len(blolayer.eq_constraints))]
                    old_ineq_dual_i = [to_torch(ineq_dual[j][i], ctx.dtype, ctx.device) for j in range(len(blolayer.ineq_functions))]

                    phi_new_i = blolayer.phi_torch(*vars_new_i, *params_i, *new_eq_dual_i, *new_ineq_dual_i)
                    phi_old_i = blolayer.phi_torch(*vars_old_i, *params_i, *old_eq_dual_i, *old_ineq_dual_i)
                    loss +=  phi_new_i - phi_old_i

                loss = blolayer.alpha * loss

            loss.backward()
            grads = [p.grad for p in params_req]

            if _compute_cos_sim:
                with torch.no_grad():
                    total_l2 = torch.sqrt(sum(
                        (g.detach().float() ** 2).sum()
                        for g in grads if g is not None
                    ))
                    total_inf = max(
                        (g.detach().float().abs().max() for g in grads if g is not None)
                    )

                cos_sim, l2_norm = _compare_grads(params_req, [p.grad for p in params_req if p.requires_grad], ground_truth_grads)
                print(f"cos_sim = {cos_sim:.6f}")
                wandb.log({
                    # "solution_distance": sol_dis,
                    "grad_l2": total_l2,
                    "grad_inf": total_inf,
                    "cos_sim": cos_sim,
                    "l2_norm": l2_norm,
                })

            return tuple(grads)

    return _BLOLayerFnFn.apply
