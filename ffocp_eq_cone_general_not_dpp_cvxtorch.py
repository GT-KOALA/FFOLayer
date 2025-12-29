import time
import cvxpy as cp
import numpy as np
import os
from copy import copy

import torch
from cvxtorch import TorchExpression
from cvxpylayers.torch import CvxpyLayer
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD
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
    cone_exprs_soc = []
    cone_exprs_exp = []
    cone_exprs_psd = []
    cone_exprs_other = []
    all_cone_exprs = {}
    all_cone_exprs["soc"] = cone_exprs_soc
    all_cone_exprs["exp"] = cone_exprs_exp
    all_cone_exprs["psd"] = cone_exprs_psd
    all_cone_exprs["other"] = cone_exprs_other
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
            if isinstance(c, SOC):
                # t, X = c.args
                # axis = c.axis
                # g_ineq = cp.norm(X, 2, axis=axis) - t
                # cone_exprs_soc.append(g_ineq)
                cone_exprs_soc.append(c)
            elif isinstance(c, ExpCone):
                # Keep the ExpCone constraint itself so we can access dual_variables later.
                cone_exprs_exp.append(c)
            elif isinstance(c, PSD):
                # Keep the PSD constraint itself so we can access dual_variables later.
                cone_exprs_psd.append(c)
            else:
                flat_blocks = []
                for arg in c.args:
                    if arg.ndim == 1:
                        flat_blocks.append(arg)
                    else:
                        flat_blocks.append(cp.vec(arg))  # flatten to 1D
                g_expr = cp.hstack(flat_blocks)
                cone_exprs_other.append(g_expr)
            
    return _BLOLayer(
        objective=objective_expr,
        eq_functions=eq_funcs,
        scalar_ineq_functions=scalar_ineq_funcs,
        cone_constraints=cone_ineq_funcs,
        cone_exprs=all_cone_exprs,
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
        # --- Cone-related bookkeeping (SOC / ExpCone / PSD) ---
        # Keep everything defined even if a cone type is absent, so the rest of the code can be uniform.
        soc_dual_product = 0
        exp_dual_product = 0
        psd_dual_product = 0

        # SOC
        self.soc_constraints = []
        self.soc_dual_params_0 = []
        self.soc_dual_params_1 = []
        self.soc_lam_params = []
        self.soc_lin_constraints = []
        if len(cone_exprs["soc"]) > 0:
            self.soc_constraints = cone_exprs["soc"]
            self.soc_dual_params_0 = [
                cp.Parameter(shape=f.dual_variables[0].shape, nonneg=True) for f in self.soc_constraints
            ]
            self.soc_dual_params_1 = [
                cp.Parameter(shape=f.dual_variables[1].shape) for f in self.soc_constraints
            ]
            # Lagrange multipliers for the SOC linear (complementarity) constraints in the ghost program.
            self.soc_lam_params = [cp.Parameter(shape=()) for _ in self.soc_constraints]

            soc_dual_product = cp.sum([
                cp.multiply(cp.pnorm(h.args[1].expr, p=2) - h.args[0].expr, du)
                for du, h in zip(self.soc_dual_params_0, self.soc_constraints)
            ])

            self.soc_lin_constraints = [
                (
                    self.soc_dual_params_1[j].T @ self.soc_constraints[j].args[1].expr
                    + cp.multiply(self.soc_constraints[j].args[0].expr, self.soc_dual_params_0[j])
                ) == 0
                for j in range(len(self.soc_constraints))
            ]

        # ExpCone
        # We use a "pre-transform" scalar weight times an h_C proxy in the perturbed objective, plus the
        # standard complementarity hyperplane <z*, s(y)> = 0.
        self.exp_constraints = []
        self.exp_dual_params = []          # list of [u, v, w] Parameters (from forward duals)
        self.exp_lambda_star_params = []   # scalar Parameters (computed from forward duals)
        self.exp_lam_params = []           # multipliers for exp_lin_constraints
        self.exp_lin_constraints = []
        self.exp_domain_constraints = []
        self._exp_z_eps = 1e-8
        if len(cone_exprs["exp"]) > 0:
            self.exp_constraints = cone_exprs["exp"]
            for con in self.exp_constraints:
                duals = con.dual_variables
                if not isinstance(duals, (list, tuple)) or len(duals) != 3:
                    raise ValueError("Unexpected ExpCone dual_variables structure")
                u_p = cp.Parameter(shape=duals[0].shape)
                v_p = cp.Parameter(shape=duals[1].shape)
                w_p = cp.Parameter(shape=duals[2].shape)
                self.exp_dual_params.append([u_p, v_p, w_p])

                # scalar tilde-lambda for the h_C proxy term
                self.exp_lambda_star_params.append(cp.Parameter(shape=(), nonneg=True))
                # multipliers for the linear constraint
                self.exp_lam_params.append(cp.Parameter(shape=()))

                x_expr, y_expr, z_expr = con.args
                lin_expr = (
                    cp.sum(cp.multiply(u_p, x_expr))
                    + cp.sum(cp.multiply(v_p, y_expr))
                    + cp.sum(cp.multiply(w_p, z_expr))
                )
                self.exp_lin_constraints.append(lin_expr == 0)

                # Domain constraints so rel_entr(y,z) is well-defined even though we drop ExpCone in perturbed problem.
                self.exp_domain_constraints.append(y_expr >= 0)
                self.exp_domain_constraints.append(z_expr >= self._exp_z_eps)

            exp_dual_product = cp.sum([
                cp.multiply(
                    lam_p,
                    cp.max(cp.vec(con.args[0] + cp.rel_entr(con.args[1], con.args[2])))
                )
                for con, lam_p in zip(self.exp_constraints, self.exp_lambda_star_params)
            ])

        # PSD
        self.psd_constraints = []
        self.psd_dual_params = []          # list of Z Parameters
        self.psd_lambda_star_params = []   # scalar Parameters
        self.psd_lam_params = []           # multipliers for psd_lin_constraints
        self.psd_lin_constraints = []
        if len(cone_exprs["psd"]) > 0:
            self.psd_constraints = cone_exprs["psd"]
            for con in self.psd_constraints:
                duals = con.dual_variables
                if not isinstance(duals, (list, tuple)) or len(duals) != 1:
                    raise ValueError("Unexpected PSD dual_variables structure")
                Z_p = cp.Parameter(shape=duals[0].shape)
                self.psd_dual_params.append(Z_p)

                self.psd_lambda_star_params.append(cp.Parameter(shape=(), nonneg=True))
                self.psd_lam_params.append(cp.Parameter(shape=()))

                (X_expr,) = con.args
                lin_expr = cp.sum(cp.multiply(Z_p, X_expr))
                self.psd_lin_constraints.append(lin_expr == 0)

            psd_dual_product = cp.sum([
                cp.multiply(lam_p, -cp.lambda_min(con.args[0]))
                for con, lam_p in zip(self.psd_constraints, self.psd_lambda_star_params)
            ])

        # Any other cones remain unsupported for now.
        if len(cone_exprs.get("other", [])) > 0:
            raise ValueError("Other cone types are not implemented")

        vars_dvars_product = cp.sum([
            cp.sum(cp.multiply(dv, v))
            for dv, v in zip(self.dvar_params, self.variables)
        ])
        scalar_ineq_dual_product = cp.sum([
            cp.sum(cp.multiply(lm, g))
            for lm, g in zip(self.scalar_ineq_dual_params,
                             self.scalar_ineq_functions)
        ])
        if 'soc_dual_product' not in locals():
            soc_dual_product = 0
        if 'exp_dual_product' not in locals():
            exp_dual_product = 0
        if 'psd_dual_product' not in locals():
            psd_dual_product = 0

        self.new_objective = (1.0 / self.alpha) * vars_dvars_product \
                             + self.objective + scalar_ineq_dual_product + soc_dual_product + exp_dual_product + psd_dual_product

        self.active_eq_constraints = []
        # print("scalar_ineq_dual_product DPP?", scalar_ineq_dual_product.is_dcp(dpp=True))
        # print("cone_dual_product DPP?", cone_dual_product.is_dcp(dpp=True))

        # 1) scalar：mask * g(x) == 0
        for j, g in enumerate(self.scalar_ineq_functions):
            self.active_eq_constraints.append(
                cp.multiply(self.scalar_active_mask_params[j], g) == 0
            )
            print("active_eq_constraints[j] DPP?", self.active_eq_constraints[j].is_dcp(dpp=True))

        self.perturbed_problem = cp.Problem(
            cp.Minimize(self.new_objective),
            self.eq_constraints + self.active_eq_constraints + self.soc_lin_constraints + self.exp_lin_constraints + self.psd_lin_constraints + self.exp_domain_constraints
        )
       
        print("perturbed_problem is_dcp:", self.perturbed_problem.is_dcp())
        print("perturbed_problem is_dpp:", self.perturbed_problem.is_dpp())

        phi_expr = (self.objective
            + cp.sum([
                cp.sum(cp.multiply(du, f))
                for du, f in zip(self.eq_dual_params, self.eq_functions)
            ])
            + cp.sum([
                cp.sum(cp.multiply(du, g))
                for du, g in zip(self.scalar_ineq_dual_params,
                                 self.scalar_ineq_functions)
            ])
            + soc_dual_product + exp_dual_product + psd_dual_product  # for new ghost objective \tilde{g}
            + cp.sum([
                cp.sum(cp.multiply(du, f.expr))
                for du, f in zip(self.soc_lam_params, self.soc_lin_constraints)
            ])
            + cp.sum([
                cp.sum(cp.multiply(du, f.expr))
                for du, f in zip(self.exp_lam_params, self.exp_lin_constraints)
            ])
            + cp.sum([
                cp.sum(cp.multiply(du, f.expr))
                for du, f in zip(self.psd_lam_params, self.psd_lin_constraints)
            ])  # for active cone dual products
        )

        self.phi_torch = TorchExpression(
            phi_expr,
            provided_vars_list=[
                *self.variables,
                *self.param_order,
                *self.eq_dual_params,
                *self.scalar_ineq_dual_params,
                *self.soc_dual_params_0,
                *self.soc_dual_params_1,
                *[p for tri in self.exp_dual_params for p in tri],
                *self.psd_dual_params,
                *self.exp_lambda_star_params,
                *self.psd_lambda_star_params,
                *self.soc_lam_params,
                *self.exp_lam_params,
                *self.psd_lam_params,
            ],
        ).torch_expression

        self.forward_setup_time = 0
        self.backward_setup_time = 0
        self.forward_solve_time = 0
        self.backward_solve_time = 0

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
            soc_dual_0 = [np.empty((B,) + h.dual_variables[0].shape, dtype=float) for h in blolayer.soc_constraints]
            soc_dual_1 = [np.empty((B,) + h.dual_variables[1].shape, dtype=float) for h in blolayer.soc_constraints]
            soc_lam = [np.empty((B,) + h.shape, dtype=float) for h in blolayer.soc_lin_constraints]
            exp_dual_0 = [np.empty((B,) + c.dual_variables[0].shape, dtype=float) for c in blolayer.exp_constraints]
            exp_dual_1 = [np.empty((B,) + c.dual_variables[1].shape, dtype=float) for c in blolayer.exp_constraints]
            exp_dual_2 = [np.empty((B,) + c.dual_variables[2].shape, dtype=float) for c in blolayer.exp_constraints]
            exp_lam = [np.empty((B,) + h.shape, dtype=float) for h in blolayer.exp_lin_constraints]
            psd_dual = [np.empty((B,) + c.dual_variables[0].shape, dtype=float) for c in blolayer.psd_constraints]
            psd_lam = [np.empty((B,) + h.shape, dtype=float) for h in blolayer.psd_lin_constraints]


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
                    # blolayer.problem.solve(**ctx.solver_args)
                except:
                    print("Forward pass GUROBI failed, using OSQP")
                    blolayer.problem.solve(solver=cp.OSQP, warm_start=False, verbose=False)

                # print(f"Forward compilation time: {blolayer.problem.compilation_time}")
                # print(f"Forward setup time: {blolayer.problem.solver_stats.setup_time}")
                # print(f"Forward solve time: {blolayer.problem.solver_stats.solve_time}")
                # print(f"Forward num iters: {blolayer.problem.solver_stats.num_iters}")
                
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

                for c_id, c in enumerate(blolayer.soc_constraints):
                    soc_dual_0[c_id][i, ...] = c.dual_value[0]
                    if c.dual_value[1].shape[1] == 1:
                        soc_dual_1[c_id][i, ...] = c.dual_value[1].reshape(-1)
                    else:
                        soc_dual_1[c_id][i, ...] = c.dual_value[1]

                for c_id, c in enumerate(blolayer.soc_lin_constraints):
                    soc_lam[c_id][i, ...] = c.dual_value

                # ExpCone duals
                for c_id, c in enumerate(blolayer.exp_constraints):
                    dv = c.dual_value
                    if dv is None:
                        exp_dual_0[c_id][i, ...] = np.zeros(exp_dual_0[c_id][i, ...].shape, dtype=float)
                        exp_dual_1[c_id][i, ...] = np.zeros(exp_dual_1[c_id][i, ...].shape, dtype=float)
                        exp_dual_2[c_id][i, ...] = np.zeros(exp_dual_2[c_id][i, ...].shape, dtype=float)
                    else:
                        exp_dual_0[c_id][i, ...] = np.array(dv[0], dtype=float)
                        exp_dual_1[c_id][i, ...] = np.array(dv[1], dtype=float)
                        exp_dual_2[c_id][i, ...] = np.array(dv[2], dtype=float)

                for c_id, c in enumerate(blolayer.exp_lin_constraints):
                    exp_lam[c_id][i, ...] = 0.0 if c.dual_value is None else c.dual_value

                # PSD duals
                for c_id, c in enumerate(blolayer.psd_constraints):
                    dv = c.dual_value
                    if dv is None:
                        psd_dual[c_id][i, ...] = np.zeros(psd_dual[c_id][i, ...].shape, dtype=float)
                    else:
                        psd_dual[c_id][i, ...] = np.array(dv, dtype=float)

                for c_id, c in enumerate(blolayer.psd_lin_constraints):
                    psd_lam[c_id][i, ...] = 0.0 if c.dual_value is None else c.dual_value



                # cone primal & dual
                # for j, g_expr in enumerate(blolayer.cone_exprs):
                #     cone_primal_vals[j][i, ...] = np.array(g_expr.value, dtype=float)

                # for j, c in enumerate(blolayer.cone_constraints):
                #     dv_raw = c.dual_value

                #     if isinstance(dv_raw, (list, tuple)):
                #         flat_chunks = []
                #         for part in dv_raw:
                #             part_arr = np.asarray(part, dtype=float)
                #             flat_chunks.append(part_arr.reshape(-1))
                #         dv_flat = np.concatenate(flat_chunks, axis=0)
                #     else:
                #         dv_flat = np.asarray(dv_raw, dtype=float).reshape(-1)
            ctx.exp_dual_0 = exp_dual_0
            ctx.exp_dual_1 = exp_dual_1
            ctx.exp_dual_2 = exp_dual_2
            ctx.exp_lam = exp_lam
            ctx.psd_dual = psd_dual
            ctx.psd_lam = psd_lam


                #     g_shape = blolayer.cone_exprs[j].shape
                #     n_expected = int(np.prod(g_shape))
                #     if dv_flat.size != n_expected:
                #         print(
                #             f"[WARN] cone dual size mismatch at constraint {j}: "
                #             f"got {dv_flat.size}, expected {n_expected}"
                #         )
                #         exit()
                #         # if dv_flat.size > n_expected:
                #         #     dv_flat = dv_flat[:n_expected]
                #         # else:
                #         #     dv_flat = np.pad(dv_flat, (0, n_expected - dv_flat.size))

                #     dv_arr = dv_flat.reshape(g_shape)
                #     cone_dual_vals[j][i, ...] = dv_arr


            ctx.sol_numpy = sol_numpy
            ctx.eq_dual = eq_dual
            ctx.scalar_ineq_dual = scalar_ineq_dual
            ctx.scalar_ineq_slack = scalar_ineq_slack
            # ctx.cone_primal_vals = cone_primal_vals
            # ctx.cone_dual_vals = cone_dual_vals
            ctx.soc_dual_0 = soc_dual_0
            ctx.soc_dual_1 = soc_dual_1
            ctx.soc_lam = soc_lam

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
            soc_dual_0 = ctx.soc_dual_0
            soc_dual_1 = ctx.soc_dual_1
            soc_lam = ctx.soc_lam
            exp_dual_0 = ctx.exp_dual_0
            exp_dual_1 = ctx.exp_dual_1
            exp_dual_2 = ctx.exp_dual_2
            exp_lam = ctx.exp_lam
            psd_dual = ctx.psd_dual
            psd_lam = ctx.psd_lam


            num_scalar_ineq = len(blolayer.scalar_ineq_functions)
            num_soc_cones = len(blolayer.soc_constraints)
            num_exp_cones = len(blolayer.exp_constraints)
            num_psd_cones = len(blolayer.psd_constraints)

            
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

            new_sol_lagrangian = [np.empty_like(sol_numpy[k]) for k in range(len(blolayer.variables))]
            new_eq_dual = [np.empty_like(eq_dual[k]) for k in range(len(blolayer.eq_constraints))]
            
            new_soc_lam = [np.empty((B,) + con.expr.shape, dtype=float) for con in blolayer.soc_lin_constraints]
            new_exp_lam = [np.empty((B,) + con.expr.shape, dtype=float) for con in blolayer.exp_lin_constraints]
            new_psd_lam = [np.empty((B,) + con.expr.shape, dtype=float) for con in blolayer.psd_lin_constraints]

            exp_lambda_star = [np.empty((B,), dtype=float) for _ in blolayer.exp_constraints]
            psd_lambda_star = [np.empty((B,), dtype=float) for _ in blolayer.psd_constraints]


            new_scalar_ineq_dual = [
                np.empty_like(scalar_ineq_dual[j]) 
                for j in range(num_scalar_ineq)
            ]

            new_active_dual = [np.empty((B,) + c.shape, dtype=float) for c in blolayer.active_eq_constraints]
            backward_time2 = time.time() - backward_start_time2
            # print(f"Backward cloning time: {backward_time2}")
            
            sol_diffs = []
            # print("Batch size: ", B)
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
                    v.value = ctx._warm_vars[j]

                for j, c in enumerate(blolayer.eq_constraints):
                    if c.dual_value is None and ctx._warm_eq_duals[j] is not None:
                        c.dual_value = ctx._warm_eq_duals[j]
                for j, c in enumerate(blolayer.scalar_ineq_constraints):
                    if c.dual_value is None and ctx._warm_ineq_duals[j] is not None:
                        c.dual_value = ctx._warm_ineq_duals[j]

                for j, _ in enumerate(blolayer.scalar_ineq_functions):
                    lam = scalar_ineq_dual[j][i]
                    lam = np.where(lam < -1e-6, lam, np.maximum(lam, 0.0))
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
                    print(f"scalar_active_mask_params[j].value: {blolayer.scalar_active_mask_params[j].value}")

                # print(f"Number of active ineq constraints: {blolayer.scalar_active_mask_params[0].value.sum()}")    
                # print(f"Number of active cones: {_num_active_cones}")

                for j, _ in enumerate(blolayer.eq_functions):
                    blolayer.eq_dual_params[j].value = eq_dual[j][i]

                for j in range(num_soc_cones):
                    u = soc_dual_0[j][i]
                    v = soc_dual_1[j][i]

                    u = np.maximum(u, 0.0)

                    blolayer.soc_dual_params_0[j].value = u
                    blolayer.soc_dual_params_1[j].value = v

                # ExpCone dual params and scalar weights
                for j in range(num_exp_cones):
                    u = exp_dual_0[j][i]
                    v = exp_dual_1[j][i]
                    w = exp_dual_2[j][i]
                    blolayer.exp_dual_params[j][0].value = u
                    blolayer.exp_dual_params[j][1].value = v
                    blolayer.exp_dual_params[j][2].value = w
                    lam = np.linalg.norm(np.concatenate([u.ravel(), v.ravel(), w.ravel()]))
                    lam = float(min(max(lam, 0.0), blolayer.dual_cutoff))
                    exp_lambda_star[j][i] = lam
                    blolayer.exp_lambda_star_params[j].value = lam

                # PSD dual params and scalar weights
                for j in range(num_psd_cones):
                    Z = psd_dual[j][i]
                    blolayer.psd_dual_params[j].value = Z
                    lam = np.linalg.norm(Z.ravel())
                    lam = float(min(max(lam, 0.0), blolayer.dual_cutoff))
                    psd_lambda_star[j][i] = lam
                    blolayer.psd_lambda_star_params[j].value = lam

                # print(f"Backward part1 time: {time.time() - backward_part1}")

                backward_part2 = time.time()

                backward_solver_args = dict(ctx.solver_args)
                if ctx.solver_args.get("solver") != cp.MOSEK:
                    backward_solver_args["warm_start"] = True
                
                # blolayer.perturbed_problem.solve(**backward_solver_args)
                blolayer.perturbed_problem.solve(solver=cp.SCS, warm_start=False, ignore_dpp=True, max_iters=2500, eps=1e-12)
                
                # print(f"Backward actual solving time: {time.time() - backward_part2}")

                # print(f"Backward compilation time: {blolayer.perturbed_problem.compilation_time}")
                # print(f"Backward setup time: {blolayer.perturbed_problem.solver_stats.setup_time}")
                # print(f"Backward solve time: {blolayer.perturbed_problem.solver_stats.solve_time}")
                # print(f"Backward num iters: {blolayer.perturbed_problem.solver_stats.num_iters}")

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
                for c_id, c in enumerate(blolayer.soc_lin_constraints):
                    new_soc_lam[c_id][i, ...] = c.dual_value
                for c_id, c in enumerate(blolayer.exp_lin_constraints):
                    new_exp_lam[c_id][i, ...] = 0.0 if c.dual_value is None else c.dual_value
                for c_id, c in enumerate(blolayer.psd_lin_constraints):
                    new_psd_lam[c_id][i, ...] = 0.0 if c.dual_value is None else c.dual_value
                for j in range(num_scalar_ineq):
                    lam_new = new_active_dual[j][i, ...]
                    new_scalar_ineq_dual[j][i, ...] = lam_new

                # print(f"Backward part3 time: {time.time() - backward_part3}")

            backward_time = time.time() - backward_start_time
            # print(f"Backward solving time: {backward_time}")
            
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

            old_soc_dual_0_torch = [
                to_torch(v, ctx.dtype, ctx.device)
                for v in soc_dual_0 if v is not None
            ]
            old_soc_dual_1_torch = [
                to_torch(v, ctx.dtype, ctx.device)
                for v in soc_dual_1 if v is not None
            ]

            # ExpCone / PSD torch duals and weights (from forward duals)
            old_exp_dual_flat_torch = []
            for u_arr, v_arr, w_arr in zip(exp_dual_0, exp_dual_1, exp_dual_2):
                old_exp_dual_flat_torch.append(to_torch(u_arr, ctx.dtype, ctx.device))
                old_exp_dual_flat_torch.append(to_torch(v_arr, ctx.dtype, ctx.device))
                old_exp_dual_flat_torch.append(to_torch(w_arr, ctx.dtype, ctx.device))

            old_psd_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in psd_dual]
            old_exp_lambda_star_torch = [to_torch(v, ctx.dtype, ctx.device) for v in exp_lambda_star]
            old_psd_lambda_star_torch = [to_torch(v, ctx.dtype, ctx.device) for v in psd_lambda_star]

            old_exp_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in exp_lam]
            new_exp_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_exp_lam]
            old_psd_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in psd_lam]
            new_psd_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_psd_lam]

            old_soc_lam_torch = [
                to_torch(v, ctx.dtype, ctx.device)
                for v in soc_lam if v is not None
            ]

            new_soc_lam_torch = [
                to_torch(v, ctx.dtype, ctx.device)
                for v in new_soc_lam
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
                    new_soc_lam_i = [d[i] for d in new_soc_lam_torch]
                    old_soc_lam_i = [d[i] for d in old_soc_lam_torch]
                    # import pdb; pdb.set_trace()

                    old_soc_dual_0_i = [d[i] for d in old_soc_dual_0_torch]
                    old_soc_dual_1_i = [d[i] for d in old_soc_dual_1_torch]


                    # ExpCone / PSD duals and cone-hyperplane multipliers
                    old_exp_dual_flat_i = [d[i] for d in old_exp_dual_flat_torch]
                    old_psd_dual_i = [d[i] for d in old_psd_dual_torch]
                    old_exp_lambda_star_i = [d[i] for d in old_exp_lambda_star_torch]
                    old_psd_lambda_star_i = [d[i] for d in old_psd_lambda_star_torch]

                    new_exp_lam_i = [d[i] for d in new_exp_lam_torch]
                    old_exp_lam_i = [d[i] for d in old_exp_lam_torch]
                    new_psd_lam_i = [d[i] for d in new_psd_lam_torch]
                    old_psd_lam_i = [d[i] for d in old_psd_lam_torch]

                    phi_new_i = blolayer.phi_torch(*vars_new_i, *params_i, *new_eq_dual_i, *new_scalar_ineq_dual_i, *old_soc_dual_0_i, *old_soc_dual_1_i, *old_exp_dual_flat_i, *old_psd_dual_i, *old_exp_lambda_star_i, *old_psd_lambda_star_i, *new_soc_lam_i, *new_exp_lam_i, *new_psd_lam_i)
                    phi_old_i = blolayer.phi_torch(*vars_old_i, *params_i, *old_eq_dual_i, *old_scalar_ineq_dual_i, *old_soc_dual_0_i, *old_soc_dual_1_i, *old_exp_dual_flat_i, *old_psd_dual_i, *old_exp_lambda_star_i, *old_psd_lambda_star_i, *old_soc_lam_i, *old_exp_lam_i, *old_psd_lam_i)
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
            # print(f"BLOLayer autograd time: {time_autograd}")

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