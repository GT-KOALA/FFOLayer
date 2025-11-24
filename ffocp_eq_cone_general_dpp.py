import time
import cvxpy as cp
import numpy as np
import os
from copy import copy

import torch
from cvxtorch import TorchExpression
from cvxpylayers.torch import CvxpyLayer
from cvxpy.constraints.second_order import SOC
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

    eq_funcs = []
    scalar_ineq_funcs = []
    cone_ineq_funcs = []
    cone_exprs = []
    for c in problem.constraints:
        # Equality: g(x,θ) == 0 
        if isinstance(c, cp.constraints.zero.Equality):
            eq_funcs.append(c.expr)

        # Inequality: g(x,θ) <= 0
        elif isinstance(c, cp.constraints.nonpos.Inequality):
            scalar_ineq_funcs.append(c.expr)

        else:
            # SOCcone constraints: t, X = c.args
            # ExpCone: args = (x, y, z)
            # PSD: args = (X,)
            cone_ineq_funcs.append(c)
            flat_blocks = []
            for arg in c.args:
                if arg.ndim == 1:
                    flat_blocks.append(arg)
                else:
                    flat_blocks.append(cp.vec(arg))  # flatten to 1D
            g_expr = cp.hstack(flat_blocks)
            cone_exprs.append(g_expr)

    return _BLOLayer(
        objective=objective_expr,
        eq_functions=eq_funcs,
        scalar_ineq_functions=scalar_ineq_funcs,
        cone_constraints=cone_ineq_funcs,
        cone_exprs=cone_exprs,
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

    def __init__(self, objective, eq_functions, scalar_ineq_functions, cone_constraints, cone_exprs, parameters, variables, alpha, dual_cutoff, slack_tol, eps, _compute_cos_sim=False):
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
        self.scalar_ineq_functions = scalar_ineq_functions
        self.cone_constraints = cone_constraints
        self.cone_exprs = cone_exprs

        self.param_order = parameters
        self.variables = variables
        self.alpha = alpha
        self.dual_cutoff = dual_cutoff
        self.slack_tol = float(slack_tol) 
        self._compute_cos_sim = _compute_cos_sim
        self.eps = eps

        self.eq_constraints = [f == 0 for f in self.eq_functions]
        self.scalar_ineq_constraints = [g <= 0 for g in self.scalar_ineq_functions]
        self.problem = cp.Problem(cp.Minimize(objective), self.eq_constraints + self.scalar_ineq_constraints + self.cone_constraints)

        self.dvar_params = [cp.Parameter(shape=v.shape) for v in self.variables]
        self.eq_dual_params = [
            cp.Parameter(shape=f.shape) for f in self.eq_functions
        ]
        self.scalar_ineq_dual_params = [
            cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions
        ]

        self.scalar_active_mask_params = [
            cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions
        ]

        # cone's related parameters
        self.cone_dual_params = [
            cp.Parameter(shape=g.shape) for g in self.cone_exprs
        ]
        self.cone_primal_star = [
            cp.Parameter(shape=g.shape) for g in self.cone_exprs
        ]
        self.cone_active_mask = [
            cp.Parameter(nonneg=True) for _ in self.cone_exprs
        ]

        self.scalar_ineq_vals = [
            cp.Variable(shape=g.shape, name=f"sineq_val_{k}")
            for k, g in enumerate(self.scalar_ineq_functions)
        ]
        self.cone_vals = [
            cp.Variable(shape=g.shape, name=f"cone_val_{k}")
            for k, g in enumerate(self.cone_exprs)
        ]

        self.link_constraints = []
        for z, g in zip(self.scalar_ineq_vals, self.scalar_ineq_functions):
            self.link_constraints.append(z == g)
        for z, g in zip(self.cone_vals, self.cone_exprs):
            self.link_constraints.append(z == g)

        vars_dvars_product = cp.sum([
            cp.sum(cp.multiply(dv, v))
            for dv, v in zip(self.dvar_params, self.variables)
        ])
        scalar_ineq_dual_product = cp.sum([
            cp.sum(cp.multiply(lm, z))
            for lm, z in zip(self.scalar_ineq_dual_params,
                             self.scalar_ineq_vals)
        ])
        cone_dual_product = cp.sum([
            cp.sum(cp.multiply(y, z))
            for y, z in zip(self.cone_dual_params, self.cone_vals)
        ])

        self.cone_dual_dot_gstar = [
            cp.Parameter(shape=(), name=f"cone_dual_dot_gstar_{j}")
            for j in range(len(self.cone_exprs))
        ]

        self.new_objective = (1.0 / self.alpha) * vars_dvars_product \
                             + self.objective + scalar_ineq_dual_product + cone_dual_product

        self.active_eq_constraints = []

        print("new_objective DPP?", self.new_objective.is_dcp(dpp=True))

        print("vars_dvars_product DPP?", vars_dvars_product.is_dcp(dpp=True))
        # print("scalar_ineq_dual_product DPP?", scalar_ineq_dual_product.is_dcp(dpp=True))
        print("cone_dual_product DPP?", cone_dual_product.is_dcp(dpp=True))

        M_scalar = 1e4
        for j, z in enumerate(self.scalar_ineq_vals):
            self.active_eq_constraints.append(
                cp.norm(z, 1) <= M_scalar * (1 - self.scalar_active_mask_params[j])
            )

        M_cone = 1e4

        for j, z in enumerate(self.cone_vals):
            dual_param = self.cone_dual_params[j]
            inner_y_g = cp.sum(cp.multiply(dual_param, z))
            lin = inner_y_g - self.cone_dual_dot_gstar[j]
            self.active_eq_constraints.append(
                cp.norm(lin, 1) <= M_cone * (1 - self.cone_active_mask[j])
            )

        self.perturbed_problem = cp.Problem(
            cp.Minimize(self.new_objective),
            self.eq_constraints + self.active_eq_constraints + self.link_constraints
        )
       
        print("perturbed_problem is_dcp:", self.perturbed_problem.is_dcp())
        print("perturbed_problem is_dpp:", self.perturbed_problem.is_dpp())

        phi_expr = self.objective \
            + cp.sum([
                cp.sum(cp.multiply(du, f))
                for du, f in zip(self.eq_dual_params, self.eq_functions)
            ]) \
            + cp.sum([
                cp.sum(cp.multiply(du, g))
                for du, g in zip(self.scalar_ineq_dual_params,
                                 self.scalar_ineq_functions)
            ]) \
            + cp.sum([
                cp.sum(cp.multiply(du, g))
                for du, g in zip(self.cone_dual_params, self.cone_exprs)
            ])

        self.phi_torch = TorchExpression(
            phi_expr,
            provided_vars_list=[
                *self.variables,
                *self.param_order,
                *self.eq_dual_params,
                *self.scalar_ineq_dual_params,
                *self.cone_dual_params,
            ],
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
        elif solver_args.get("solver") == cp.SCS:
            default_solver_args = dict(
                solver=cp.SCS,
                warm_start=False,
                ignore_dpp=False,
                max_iters=2500,
                eps=self.eps,
            )
        else:
            default_solver_args = {"ignore_dpp": False}
        solver_args = {**default_solver_args, **solver_args}
        
        info = {}
        f = _BLOLayerFn(
            blolayer=self,
            solver_args=solver_args,
            _compute_cos_sim=self._compute_cos_sim,
            info=info
        )
        sol = f(*params)
        self.info = info
        return sol

def _BLOLayerFn(
        blolayer,
        solver_args,
        _compute_cos_sim,
        info):
    class _BLOLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            # infer dtype, device, and whether or not params are batched
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device
            ctx.solver_args = solver_args

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

            scalar_ineq_dual = [
                np.empty((B,) + g.shape, dtype=float)
                for g in blolayer.scalar_ineq_functions
            ]
            scalar_ineq_slack = [
                np.empty((B,) + g.shape, dtype=float)
                for g in blolayer.scalar_ineq_functions
            ]
            cone_primal_vals = [
                np.empty((B,) + g.shape, dtype=float)
                for g in blolayer.cone_exprs
            ]
            cone_dual_vals = [
                np.empty((B,) + g.shape, dtype=float)
                for g in blolayer.cone_exprs
            ]

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
                    # blolayer.problem.solve(solver=cp.SCS, warm_start=False, ignore_dpp=True, max_iters=2500, eps=blolayer.eps)
                    blolayer.problem.solve(**ctx.solver_args)
                except:
                    print("Forward pass GUROBI failed, using OSQP")
                    blolayer.problem.solve(solver=cp.OSQP, warm_start=False, verbose=False)

                print(f"Forward compilation time: {blolayer.problem.compilation_time}")
                print(f"Forward setup time: {blolayer.problem.solver_stats.setup_time}")
                print(f"Forward solve time: {blolayer.problem.solver_stats.solve_time}")
                print(f"Forward num iters: {blolayer.problem.solver_stats.num_iters}")
                
                assert blolayer.problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                # primal
                for v_id, v in enumerate(blolayer.variables):
                    sol_numpy[v_id][i, ...] = v.value

                # eq dual
                for c_id, c in enumerate(blolayer.eq_constraints):
                    eq_dual[c_id][i, ...] = c.dual_value

                # ineq dual & slack
                for j, g_expr in enumerate(blolayer.scalar_ineq_functions):
                    g_val = g_expr.value
                    scalar_ineq_dual[j][i, ...] = blolayer.scalar_ineq_constraints[j].dual_value
                    s_val = -g_val
                    s_val = np.maximum(s_val, 0.0)
                    scalar_ineq_slack[j][i, ...] = s_val

                # cone primal & dual
                for j, g_expr in enumerate(blolayer.cone_exprs):
                    cone_primal_vals[j][i, ...] = np.array(g_expr.value, dtype=float)

                for j, c in enumerate(blolayer.cone_constraints):
                    dv_raw = c.dual_value

                    if isinstance(dv_raw, (list, tuple)):
                        flat_chunks = []
                        for part in dv_raw:
                            part_arr = np.asarray(part, dtype=float)
                            flat_chunks.append(part_arr.reshape(-1))
                        dv_flat = np.concatenate(flat_chunks, axis=0)
                    else:
                        dv_flat = np.asarray(dv_raw, dtype=float).reshape(-1)

                    g_shape = blolayer.cone_exprs[j].shape
                    n_expected = int(np.prod(g_shape))
                    if dv_flat.size != n_expected:
                        print(
                            f"[WARN] cone dual size mismatch at constraint {j}: "
                            f"got {dv_flat.size}, expected {n_expected}"
                        )
                        exit()
                        # if dv_flat.size > n_expected:
                        #     dv_flat = dv_flat[:n_expected]
                        # else:
                        #     dv_flat = np.pad(dv_flat, (0, n_expected - dv_flat.size))

                    dv_arr = dv_flat.reshape(g_shape)
                    cone_dual_vals[j][i, ...] = dv_arr


            ctx.sol_numpy = sol_numpy
            ctx.eq_dual = eq_dual
            ctx.scalar_ineq_dual = scalar_ineq_dual
            ctx.scalar_ineq_slack = scalar_ineq_slack
            ctx.cone_primal_vals = cone_primal_vals
            ctx.cone_dual_vals = cone_dual_vals

            ctx.params_numpy = params_numpy
            ctx.params = params
            ctx.blolayer = blolayer

            ctx._warm_vars = [copy(v.value) for v in blolayer.variables]
            if len(blolayer.eq_constraints) > 0:
                ctx._warm_eq_duals = [copy(c.dual_value) for c in blolayer.eq_constraints]
            ctx._warm_ineq_duals = [copy(c.dual_value) for c in blolayer.scalar_ineq_constraints]
            ctx._warm_ineq_slack_residuals = [copy(c.value) for c in blolayer.scalar_ineq_functions]

            sol_torch = [to_torch(arr, ctx.dtype, ctx.device) for arr in sol_numpy]

            return tuple(sol_torch)

        @staticmethod
        def backward(ctx, *dvars):
            backward_start_time = time.time()

            # convert to numpy arrays
            dvars_numpy = [to_numpy(dvar) for dvar in dvars]
            
            # temperature = 10
            # ineq_dual_tanh = [np.tanh(dual * temperature) for dual in ctx.ineq_dual]

            blolayer = ctx.blolayer
            B = ctx.batch_size
            sol_numpy = ctx.sol_numpy
            eq_dual = ctx.eq_dual
            scalar_ineq_dual = ctx.scalar_ineq_dual
            scalar_ineq_slack = ctx.scalar_ineq_slack
            cone_primal_vals = ctx.cone_primal_vals
            cone_dual_vals = ctx.cone_dual_vals
            
            y_dim = dvars_numpy[0].shape[1]
            if len(eq_dual) == 0:
                num_eq = 0
            else:
                num_eq = eq_dual[0].shape[1]

            params_numpy = ctx.params_numpy

            backward_start_time2 = time.time()
            params_req, req_grad_mask = [], []
            for p in ctx.params:
                q = p.detach().clone()
                req_grad = bool(p.requires_grad)
                q.requires_grad_(req_grad)
                params_req.append(q)
                req_grad_mask.append(req_grad)
            

            if _compute_cos_sim:
                # compute ground truth gradient using cvxpylayer
                cvxpylayer_problem = cp.Problem(
                    cp.Minimize(blolayer.objective),
                    constraints=(
                        blolayer.eq_constraints
                        + blolayer.scalar_ineq_constraints
                        + blolayer.cone_constraints
                    ),
                )
                _cvx_layer = CvxpyLayer(
                    cvxpylayer_problem,
                    parameters=blolayer.param_order,
                    variables=blolayer.variables,
                )
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
                    retain_graph=False,
                )

            new_sol_lagrangian = [np.empty_like(sol_numpy[k]) for k in range(len(blolayer.variables))]
            new_eq_dual = [np.empty_like(eq_dual[k]) for k in range(len(blolayer.eq_constraints))]
            num_scalar_ineq = len(blolayer.scalar_ineq_functions)
            num_cones = len(blolayer.cone_exprs)

            new_scalar_ineq_dual = [
                np.empty_like(scalar_ineq_dual[j]) 
                for j in range(num_scalar_ineq)
            ]

            new_cone_dual_vals = [
                np.empty_like(cone_dual_vals[j]) 
                for j in range(num_cones)
            ]

            new_active_dual = [np.empty((B,) + c.shape, dtype=float) for c in blolayer.active_eq_constraints]
            backward_time2 = time.time() - backward_start_time2
            print(f"Backward cloning time: {backward_time2}")
            
            sol_diffs = []
            print("Batch size: ", B)
            for i in range(B):
                backward_part1 = time.time()
                if ctx.batch:
                    params_numpy_i = [
                        p[i] if bs > 0 else p
                        for p, bs in zip(params_numpy, ctx.batch_sizes)
                    ]
                else:
                    params_numpy_i = params_numpy
                
                for j, _ in enumerate(blolayer.param_order):
                    blolayer.param_order[j].value = params_numpy_i[j]

                for j, v in enumerate(blolayer.variables):
                    blolayer.dvar_params[j].value = dvars_numpy[j][i]
                    # Warm start
                    v.value = sol_numpy[j][i, ...]

                for j, _ in enumerate(blolayer.scalar_ineq_functions):
                    lam = scalar_ineq_dual[j][i]
                    sl = scalar_ineq_slack[j][i]
                    blolayer.scalar_ineq_dual_params[j].value = lam
                    
                    mask = (sl <= blolayer.slack_tol).astype(np.float64)
                    if mask.sum() > max(1, y_dim - num_eq):
                        k = int(max(1, y_dim - num_eq))
                        lam_flat = lam.reshape(-1)
                        idx = np.argpartition(lam_flat, -k)[-k:]
                        mask_flat = np.zeros_like(lam_flat, dtype=np.float64)
                        mask_flat[idx] = 1.0
                        mask = mask_flat.reshape(lam.shape)
                    blolayer.scalar_active_mask_params[j].value = mask

                _num_active_cones = 0
                for j, _ in enumerate(blolayer.cone_exprs):
                    g_star = ctx.cone_primal_vals[j][i, ...]   # g(x*)
                    y_star = ctx.cone_dual_vals[j][i, ...]     # dual

                    blolayer.cone_primal_star[j].value = g_star
                    blolayer.cone_dual_params[j].value = y_star

                    blolayer.cone_dual_dot_gstar[j].value = float(np.sum(y_star * g_star))

                    y_norm = float(np.linalg.norm(y_star.reshape(-1), ord=np.inf))
                    if y_norm > blolayer.dual_cutoff:
                        # Cone constraint is active
                        blolayer.cone_active_mask[j].value = 1.0
                        _num_active_cones += 1
                    else:
                        blolayer.cone_active_mask[j].value = 0.0
                    
                print(f"Number of active cones: {_num_active_cones}")

                for j, _ in enumerate(blolayer.eq_functions):
                    blolayer.eq_dual_params[j].value = eq_dual[j][i]
                
                # blolayer.perturbed_problem.solve(solver=cp.GUROBI, ignore_dpp=True, warm_start=True, **{"Threads": n_threads, "OutputFlag": 0})
                # blolayer.perturbed_problem.solve(solver=cp.SCS, warm_start=True, ignore_dpp=True, max_iters=2500, eps=blolayer.eps)

                print(f"Backward part1 time: {time.time() - backward_part1}")

                backward_part2 = time.time()

                backward_solver_args = dict(ctx.solver_args)
                if ctx.solver_args.get("solver") != cp.MOSEK:
                    backward_solver_args["warm_start"] = True
                
                blolayer.perturbed_problem.solve(**backward_solver_args)
                
                print(f"Backward actual solving time: {time.time() - backward_part2}")

                print(f"Backward compilation time: {blolayer.perturbed_problem.compilation_time}")
                print(f"Backward setup time: {blolayer.perturbed_problem.solver_stats.setup_time}")
                print(f"Backward solve time: {blolayer.perturbed_problem.solver_stats.solve_time}")
                print(f"Backward num iters: {blolayer.perturbed_problem.solver_stats.num_iters}")

                # blolayer.perturbed_problem.solve(**ctx.solver_args)

                backward_part3 = time.time()
                st = blolayer.perturbed_problem.status
                try:
                    if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        raise RuntimeError(f"New bilevel problem status = {st}")
                    for j, v in enumerate(blolayer.variables):
                        new_sol_lagrangian[j][i, ...] = v.value
                        # sol_diff = np.linalg.norm(sol_numpy[j][i] - v.value)
                        # sol_diffs.append(sol_diff)
                except:
                    print("Backward pass GUROBI failed, using OSQP")
                    blolayer.perturbed_problem.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, warm_start=True, verbose=False)
                    for j, v in enumerate(blolayer.variables):
                        new_sol_lagrangian[j][i, ...] = v.value
                        sol_diff = np.linalg.norm(sol_numpy[j][i] - v.value)
                        sol_diffs.append(sol_diff)
                
                for c_id, c in enumerate(blolayer.eq_constraints):
                    # if c.dual_value.any() == None:
                    #     print(f"equality constraint {c_id} dual value is None")
                    new_eq_dual[c_id][i, ...] = c.dual_value
                for c_id, c in enumerate(blolayer.active_eq_constraints):
                    # if c.dual_value.any() == None:
                    #     print(f"active inequality constraint {c_id} dual value is None")
                    # active_mask = np.array([a.value for a in blolayer.active_mask_params])
                    new_active_dual[c_id][i, ...] = c.dual_value
                for j in range(num_scalar_ineq):
                    lam_new = new_active_dual[j][i, ...]
                    new_scalar_ineq_dual[j][i, ...] = lam_new

                for j in range(num_cones):
                    idx = num_scalar_ineq + j
                    lam_cone = float(new_active_dual[idx][i, ...])
                    y_old    = cone_dual_vals[j][i, ...]
                    new_cone_dual_vals[j][i, ...] = (1.0 + lam_cone) * y_old

                print(f"Backward part3 time: {time.time() - backward_part3}")

            backward_time = time.time() - backward_start_time
            print(f"Backward solving time: {backward_time}")
            
            # print('--- sol_diff mean: ', np.mean(np.array(sol_diffs)), 'max: ', np.max(np.array(sol_diffs)), 'min: ', np.min(np.array(sol_diffs)))

            new_sol = [to_torch(v, ctx.dtype, ctx.device) for v in new_sol_lagrangian]
            new_eq_dual_torch  = [to_torch(v, ctx.dtype, ctx.device) for v in new_eq_dual]
            old_eq_dual_torch  = [to_torch(v, ctx.dtype, ctx.device) for v in eq_dual]

            old_scalar_ineq_dual_torch = [
                to_torch(v, ctx.dtype, ctx.device) for v in scalar_ineq_dual
            ]
            new_scalar_ineq_dual_torch = [
                to_torch(v, ctx.dtype, ctx.device) for v in new_scalar_ineq_dual
            ]

            old_cone_dual_torch = [
                to_torch(v, ctx.dtype, ctx.device)
                for v in cone_dual_vals if v is not None
            ]
            new_cone_dual_torch = [
                to_torch(v, ctx.dtype, ctx.device)
                for v in new_cone_dual_vals
            ]

            start_time = time.time()
            params_req = []
            for p, need in zip(ctx.params, req_grad_mask):
                q = p.detach().clone()
                if need:
                    q.requires_grad_(True)
                params_req.append(q)
            if ctx.device != torch.device('cpu'):
                torch.set_default_device(torch.device(ctx.device))
            loss = 0.0
            with torch.enable_grad():
                for i in range(B):
                    vars_new_i = [v[i] for v in new_sol]
                    vars_old_i = [to_torch(sol_numpy[j][i], ctx.dtype, ctx.device) for j in range(len(blolayer.variables))]
                    
                    params_i = slice_params_for_batch(params_req, ctx.batch_sizes, i)

                    # eq dual
                    new_eq_dual_i  = [d[i] for d in new_eq_dual_torch]
                    old_eq_dual_i  = [d[i] for d in old_eq_dual_torch]

                    # scalar ineq dual
                    new_scalar_ineq_dual_i = [d[i] for d in new_scalar_ineq_dual_torch]
                    old_scalar_ineq_dual_i = [d[i] for d in old_scalar_ineq_dual_torch]

                    # cone dual
                    new_cone_dual_i = [d[i] for d in new_cone_dual_torch]
                    old_cone_dual_i = [d[i] for d in old_cone_dual_torch]

                    phi_new_i = blolayer.phi_torch(*vars_new_i, *params_i, *new_eq_dual_i, *new_scalar_ineq_dual_i, *new_cone_dual_i)
                    phi_old_i = blolayer.phi_torch(*vars_old_i, *params_i, *old_eq_dual_i, *old_scalar_ineq_dual_i, *old_cone_dual_i)
                    loss +=  phi_new_i - phi_old_i

                loss = blolayer.alpha * loss

            # loss.backward()
            # grads = [p.grad for p in params_req]

            grads_req = torch.autograd.grad(
                outputs=loss,
                inputs=[q for q, need in zip(params_req, req_grad_mask) if need],
                allow_unused=True,
                retain_graph=False,
            )

            grads = []
            it = iter(grads_req)
            for need in req_grad_mask:
                grads.append(next(it) if need else None)
            time_autograd = time.time() - start_time
            print(f"BLOLayer autograd time: {time_autograd}")

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
