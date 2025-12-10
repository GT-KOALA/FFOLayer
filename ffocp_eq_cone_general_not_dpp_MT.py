import time
import cvxpy as cp
import numpy as np
from copy import copy

import torch
from cvxtorch import TorchExpression
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD
import wandb
from utils import to_numpy, to_torch, _dump_cvxpy, n_threads, slice_params_for_batch

try:
    from multiprocessing.pool import ThreadPool
except Exception:
    ThreadPool = None
try:
    from threadpoolctl import threadpool_limits
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(*args, **kwargs):
        yield

@torch.no_grad()
def _compare_grads(params_req, grads, ground_truth_grads):
    est_chunks, gt_chunks = [], []
    for p, ge, gg in zip(params_req, grads, ground_truth_grads):
        ge = torch.zeros_like(p) if ge is None else ge.detach()
        gg = torch.zeros_like(p) if gg is None else gg.detach()
        est_chunks.append(ge.reshape(-1))
        gt_chunks.append(gg.reshape(-1))
    est = torch.cat(est_chunks)
    gt = torch.cat(gt_chunks)
    eps = 1e-12
    denom = (est.norm() * gt.norm()).clamp_min(eps)
    cos_sim = torch.dot(est, gt) / denom
    l2_diff = (est - gt).norm()
    return cos_sim, l2_diff


# def BLOLayer(problem, parameters, variables, alpha=100.0, dual_cutoff=1e-3, slack_tol=1e-8, eps=1e-7, compute_cos_sim=False):
#     if isinstance(problem, (list, tuple)):
#         problem_list = list(problem)
#         parameters_list = list(parameters)
#         variables_list = list(variables)
#     else:
#         try:
#             problem_list, parameters_list, variables_list = _dump_cvxpy(problem, parameters, variables, n_threads)
#         except Exception:
#             problem_list = [problem]
#             parameters_list = [parameters]
#             variables_list = [variables]
#     return _BLOLayer(problem_list, parameters_list, variables_list, alpha, dual_cutoff, slack_tol, eps, compute_cos_sim)

class BLOLayer(torch.nn.Module):
    def __init__(self, problem_list, parameters_list, variables_list, alpha, dual_cutoff, slack_tol, eps, _compute_cos_sim=False):
        super().__init__()
        self.problem_list_in = problem_list
        self.param_order_list = parameters_list
        self.variables_list = variables_list
        self.alpha = float(alpha)
        self.dual_cutoff = float(dual_cutoff)
        self.slack_tol = float(slack_tol)
        self.eps = float(eps)
        self._compute_cos_sim = bool(_compute_cos_sim)
        self.num_copies = len(self.problem_list_in)

        self.objective_list = []
        self.eq_functions_list = []
        self.scalar_ineq_functions_list = []
        self.soc_constraints_list = []

        for prob in self.problem_list_in:
            obj = prob.objective
            if isinstance(obj, cp.Minimize):
                objective_expr = obj.expr
            elif isinstance(obj, cp.Maximize):
                objective_expr = -obj.expr
            else:
                objective_expr = getattr(obj, "expr", None)
                if objective_expr is None:
                    raise ValueError("Unsupported objective type; expected Minimize/Maximize.")
            eq_funcs, scalar_ineq_funcs, soc_cons = [], [], []
            for c in prob.constraints:
                if isinstance(c, cp.constraints.zero.Equality):
                    eq_funcs.append(c.expr)
                elif isinstance(c, cp.constraints.nonpos.Inequality):
                    scalar_ineq_funcs.append(c.expr)
                else:
                    if isinstance(c, SOC):
                        soc_cons.append(c)
                    elif isinstance(c, (ExpCone, PSD)):
                        raise ValueError("Not implemented")
                    else:
                        raise ValueError("Not implemented")
            self.objective_list.append(objective_expr)
            self.eq_functions_list.append(eq_funcs)
            self.scalar_ineq_functions_list.append(scalar_ineq_funcs)
            self.soc_constraints_list.append(soc_cons)

        self.eq_constraints_list = []
        self.scalar_ineq_constraints_list = []
        self.problem_list = []

        self.dvar_params_list = []
        self.eq_dual_params_list = []
        self.scalar_ineq_dual_params_list = []
        self.scalar_active_mask_params_list = []

        self.soc_dual_params_0_list = []
        self.soc_dual_params_1_list = []
        self.soc_lam_params_list = []
        self.soc_lin_constraints_list = []
        self.soc_dual_product_list = []

        self.active_eq_constraints_list = []
        self.perturbed_problem_list = []

        for slot in range(self.num_copies):
            eq_constraints = [f == 0 for f in self.eq_functions_list[slot]]
            scalar_ineq_constraints = [g <= 0 for g in self.scalar_ineq_functions_list[slot]]
            self.eq_constraints_list.append(eq_constraints)
            self.scalar_ineq_constraints_list.append(scalar_ineq_constraints)

            base_prob = cp.Problem(
                cp.Minimize(self.objective_list[slot]),
                eq_constraints + scalar_ineq_constraints + self.soc_constraints_list[slot],
            )
            self.problem_list.append(base_prob)

            dvar_params = [cp.Parameter(shape=v.shape) for v in self.variables_list[slot]]
            eq_dual_params = [cp.Parameter(shape=f.shape) for f in self.eq_functions_list[slot]]
            scalar_ineq_dual_params = [cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions_list[slot]]
            scalar_active_mask_params = [cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions_list[slot]]

            self.dvar_params_list.append(dvar_params)
            self.eq_dual_params_list.append(eq_dual_params)
            self.scalar_ineq_dual_params_list.append(scalar_ineq_dual_params)
            self.scalar_active_mask_params_list.append(scalar_active_mask_params)

            soc_cons = self.soc_constraints_list[slot]
            if len(soc_cons) > 0:
                soc_dual_params_0 = [cp.Parameter(shape=h.dual_variables[0].shape, nonneg=True) for h in soc_cons]
                soc_dual_params_1 = [cp.Parameter(shape=h.dual_variables[1].shape) for h in soc_cons]
                soc_lam_params = [cp.Parameter(shape=h.shape) for h in soc_cons]
                soc_dual_product = cp.sum([
                    cp.multiply(cp.pnorm(h.args[1].expr, p=2) - h.args[0].expr, du)
                    for du, h in zip(soc_dual_params_0, soc_cons)
                ])
                soc_lin_constraints = [
                    (soc_dual_params_1[j].T @ soc_cons[j].args[1].expr + cp.multiply(soc_cons[j].args[0].expr, soc_dual_params_0[j])) == 0
                    for j in range(len(soc_cons))
                ]
            else:
                soc_dual_params_0, soc_dual_params_1, soc_lam_params, soc_lin_constraints = [], [], [], []
                soc_dual_product = 0

            self.soc_dual_params_0_list.append(soc_dual_params_0)
            self.soc_dual_params_1_list.append(soc_dual_params_1)
            self.soc_lam_params_list.append(soc_lam_params)
            self.soc_lin_constraints_list.append(soc_lin_constraints)
            self.soc_dual_product_list.append(soc_dual_product)

            vars_dvars_product = cp.sum([cp.sum(cp.multiply(dv, v)) for dv, v in zip(dvar_params, self.variables_list[slot])])
            scalar_ineq_dual_product = cp.sum([cp.sum(cp.multiply(lm, g)) for lm, g in zip(scalar_ineq_dual_params, self.scalar_ineq_functions_list[slot])])

            new_objective = (1.0 / self.alpha) * vars_dvars_product + self.objective_list[slot] + scalar_ineq_dual_product + soc_dual_product

            active_eq_constraints = [cp.multiply(scalar_active_mask_params[j], g) == 0 for j, g in enumerate(self.scalar_ineq_functions_list[slot])]
            self.active_eq_constraints_list.append(active_eq_constraints)

            perturbed_prob = cp.Problem(
                cp.Minimize(new_objective),
                eq_constraints + active_eq_constraints + soc_lin_constraints,
            )
            self.perturbed_problem_list.append(perturbed_prob)

        slot0 = 0
        phi_expr = (
            self.objective_list[slot0]
            + cp.sum([cp.sum(cp.multiply(du, f)) for du, f in zip(self.eq_dual_params_list[slot0], self.eq_functions_list[slot0])])
            + cp.sum([cp.sum(cp.multiply(du, g)) for du, g in zip(self.scalar_ineq_dual_params_list[slot0], self.scalar_ineq_functions_list[slot0])])
            + self.soc_dual_product_list[slot0]
            + cp.sum([cp.sum(cp.multiply(du, f.expr)) for du, f in zip(self.soc_lam_params_list[slot0], self.soc_lin_constraints_list[slot0])])
        )

        provided = [
            *self.variables_list[slot0],
            *self.param_order_list[slot0],
            *self.eq_dual_params_list[slot0],
            *self.scalar_ineq_dual_params_list[slot0],
            *self.soc_dual_params_0_list[slot0],
            *self.soc_dual_params_1_list[slot0],
            *self.soc_lam_params_list[slot0],
        ]
        self.phi_torch = TorchExpression(phi_expr, provided_vars_list=provided).torch_expression

        self.use_iterative_backward = True
        if self.use_iterative_backward:
            base_provided = [*self.variables_list[0], *self.param_order_list[0]]
            self._eq_expr_torch = [
                TorchExpression(f, provided_vars_list=base_provided).torch_expression
                for f in self.eq_functions_list[0]
            ]
            self._ineq_expr_torch = [
                TorchExpression(g, provided_vars_list=base_provided).torch_expression
                for g in self.scalar_ineq_functions_list[0]
            ]

            # soc_lin constraints depend on soc_dual params
            if len(self.soc_lin_constraints_list[0]) > 0:
                soc_provided = [
                    *self.variables_list[0],
                    *self.param_order_list[0],
                    *self.soc_dual_params_0_list[0],
                    *self.soc_dual_params_1_list[0],
                ]
                self._soclin_expr_torch = [
                    TorchExpression(c.expr, provided_vars_list=soc_provided).torch_expression
                    for c in self.soc_lin_constraints_list[0]
                ]
            else:
                self._soclin_expr_torch = []

        self.forward_setup_time = 0
        self.average_forward_solve_time = 0
        self.forward_solve_time = 0
        self.backward_setup_time = 0
        self.backward_solve_time = 0
        self.average_backward_solve_time = 0

    def forward(self, *params, solver_args={}):
        if solver_args is None:
            solver_args = {}
        default_solver_args = {"ignore_dpp": False}
        solver_args = {**default_solver_args, **solver_args}
        info = {}
        f = _BLOLayerFn(self, solver_args, self._compute_cos_sim, info)
        sol = f(*params)
        self.info = info
        return sol

def _BLOLayerFn(blolayer, solver_args, _compute_cos_sim, info):
    class _BLOLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device
            ctx.solver_args = solver_args

            ctx.batch_sizes = []
            for i, (p, q) in enumerate(zip(params, blolayer.param_order_list[0])):
                if p.dtype != ctx.dtype:
                    raise ValueError(f"Parameter {i} dtype mismatch.")
                if p.device != ctx.device:
                    raise ValueError(f"Parameter {i} device mismatch.")
                if p.ndimension() == q.ndim:
                    bs = 0
                elif p.ndimension() == q.ndim + 1:
                    bs = p.size(0)
                    if bs == 0:
                        raise ValueError(f"Batch dim for parameter {i} is zero.")
                else:
                    raise ValueError(f"Invalid parameter dims for {i}.")
                ctx.batch_sizes.append(bs)
                p_shape = p.shape if bs == 0 else p.shape[1:]
                if not np.all(p_shape == q.shape):
                    raise ValueError(f"Inconsistent parameter shapes for {i}.")
            ctx.batch_sizes = np.array(ctx.batch_sizes)
            ctx.batch = np.any(ctx.batch_sizes > 0)
            if ctx.batch:
                nz = ctx.batch_sizes[ctx.batch_sizes > 0]
                ctx.batch_size = int(nz[0])
                if np.any(nz != ctx.batch_size):
                    raise ValueError("Inconsistent batch sizes.")
            else:
                ctx.batch_size = 1
            B = ctx.batch_size

            params_numpy = [to_numpy(p) for p in params]

            sol_numpy = [np.empty((B,) + v.shape, dtype=float) for v in blolayer.variables_list[0]]
            eq_dual = [np.empty((B,) + f.shape, dtype=float) for f in blolayer.eq_functions_list[0]]
            scalar_ineq_dual = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.scalar_ineq_functions_list[0]]
            scalar_ineq_slack = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.scalar_ineq_functions_list[0]]

            soc_dual_0 = [np.empty((B,) + h.dual_variables[0].shape, dtype=float) for h in blolayer.soc_constraints_list[0]]
            soc_dual_1 = [np.empty((B,) + h.dual_variables[1].shape, dtype=float) for h in blolayer.soc_constraints_list[0]]
            soc_lam = [np.empty((B,) + h.shape, dtype=float) for h in blolayer.soc_lin_constraints_list[0]]

            def _solve_one_forward(i):
                slot = i if blolayer.num_copies >= B else 0
                if ctx.batch:
                    params_i = [p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)]
                else:
                    params_i = params_numpy
                for p_val, param_obj in zip(params_i, blolayer.param_order_list[slot]):
                    param_obj.value = p_val
                try:
                    blolayer.problem_list[slot].solve(solver=cp.SCS, warm_start=False, ignore_dpp=True, max_iters=2500, eps=blolayer.eps)
                except Exception:
                    print("forward pass SCS failed, using OSQP")
                    blolayer.problem_list[slot].solve(solver=cp.OSQP, warm_start=False, verbose=False)

                setup_time = blolayer.problem_list[slot].compilation_time
                solve_time = blolayer.problem_list[slot].solver_stats.solve_time

                st = blolayer.problem_list[slot].status
                if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"Forward status = {st}")
                sol_vals = [v.value for v in blolayer.variables_list[slot]]
                eq_vals = [c.dual_value for c in blolayer.eq_constraints_list[slot]]
                ineq_vals = [c.dual_value for c in blolayer.scalar_ineq_constraints_list[slot]]
                slack_vals = [np.maximum(-expr.value, 0.0) for expr in blolayer.scalar_ineq_functions_list[slot]]
                soc0_vals, soc1_vals = [], []
                for c in blolayer.soc_constraints_list[slot]:
                    soc0_vals.append(c.dual_value[0])
                    dv1 = c.dual_value[1]
                    if hasattr(dv1, "shape") and len(dv1.shape) == 2 and dv1.shape[1] == 1:
                        soc1_vals.append(dv1.reshape(-1))
                    else:
                        soc1_vals.append(dv1)
                lam_vals = [c.dual_value for c in blolayer.soc_lin_constraints_list[slot]]
                return sol_vals, eq_vals, ineq_vals, slack_vals, soc0_vals, soc1_vals, lam_vals, setup_time, solve_time

            use_mt = (ThreadPool is not None) and (B > 1) and (blolayer.num_copies >= B)
            forward_solve_time_start = time.time()
            if use_mt:
                with threadpool_limits(limits=1):
                    pool = ThreadPool(processes=min(B, n_threads))
                    try:
                        results = pool.map(_solve_one_forward, range(B))
                    finally:
                        pool.close()
            else:
                raise ValueError("Not implemented")
                # results = [_solve_one_forward(i) for i in range(B)]
            blolayer.forward_solve_time = time.time() - forward_solve_time_start

            avg_setup_time = 0.0
            avg_solve_time = 0.0
            for i, (sol_vals, eq_vals, ineq_vals, slack_vals, soc0_vals, soc1_vals, lam_vals, setup_time, solve_time) in enumerate(results):
                for v_id in range(len(sol_numpy)):
                    sol_numpy[v_id][i, ...] = sol_vals[v_id]
                for c_id in range(len(eq_dual)):
                    eq_dual[c_id][i, ...] = eq_vals[c_id]
                for j in range(len(scalar_ineq_dual)):
                    scalar_ineq_dual[j][i, ...] = ineq_vals[j]
                    scalar_ineq_slack[j][i, ...] = slack_vals[j]
                for j in range(len(soc_dual_0)):
                    soc_dual_0[j][i, ...] = soc0_vals[j]
                    soc_dual_1[j][i, ...] = soc1_vals[j]
                for j in range(len(soc_lam)):
                    soc_lam[j][i, ...] = lam_vals[j]
                
                avg_setup_time += setup_time
                avg_solve_time += solve_time

            blolayer.forward_setup_time = avg_setup_time / B
            blolayer.average_forward_solve_time = avg_solve_time / B

            ctx.sol_numpy = sol_numpy
            ctx.eq_dual = eq_dual
            ctx.scalar_ineq_dual = scalar_ineq_dual
            ctx.scalar_ineq_slack = scalar_ineq_slack
            ctx.soc_dual_0 = soc_dual_0
            ctx.soc_dual_1 = soc_dual_1
            ctx.soc_lam = soc_lam
            ctx.params_numpy = params_numpy
            ctx.params = params
            ctx.blolayer = blolayer

            ctx._warm_vars_list = [[copy(sol_numpy[k][i, ...]) for k in range(len(sol_numpy))] for i in range(B)]

            sol_torch = [to_torch(arr, ctx.dtype, ctx.device) for arr in sol_numpy]
            return tuple(sol_torch)

        @staticmethod
        def backward(ctx, *dvars):
            blolayer = ctx.blolayer
            B = ctx.batch_size

            dvars_numpy = [to_numpy(dvar) for dvar in dvars]
            sol_numpy = ctx.sol_numpy
            eq_dual = ctx.eq_dual
            scalar_ineq_dual = ctx.scalar_ineq_dual
            scalar_ineq_slack = ctx.scalar_ineq_slack
            soc_dual_0 = ctx.soc_dual_0
            soc_dual_1 = ctx.soc_dual_1
            soc_lam = ctx.soc_lam

            num_scalar_ineq = len(blolayer.scalar_ineq_functions_list[0])
            num_soc_cones = len(blolayer.soc_constraints_list[0])

            y_dim = dvars_numpy[0].shape[1]
            num_eq = 0 if len(eq_dual) == 0 else eq_dual[0].shape[1]

            params_req, req_grad_mask = [], []
            for p in ctx.params:
                q = p.detach().clone()
                need = bool(p.requires_grad)
                q.requires_grad_(need)
                params_req.append(q)
                req_grad_mask.append(need)

            new_sol_lagrangian = [np.empty_like(sol_numpy[k]) for k in range(len(blolayer.variables_list[0]))]
            new_eq_dual = [np.empty_like(eq_dual[k]) for k in range(len(blolayer.eq_constraints_list[0]))]
            new_soc_lam = [np.empty((B,) + con.shape, dtype=float) for con in blolayer.soc_lin_constraints_list[0]]
            new_active_dual = [np.empty((B,) + c.shape, dtype=float) for c in blolayer.active_eq_constraints_list[0]]
            new_scalar_ineq_dual = [np.empty_like(scalar_ineq_dual[j]) for j in range(num_scalar_ineq)]

            
            def _iterative_solve_perturbed(i):
                ITERS = 50
                LR = 1e-2
                RHO = 10.0  # penalty strength factor for augmented lagrangian
                t0 = time.time()

                # warm start primal
                vars_t = [
                    to_torch(ctx._warm_vars_list[i][j], ctx.dtype, ctx.device).detach().requires_grad_(True)
                    for j in range(len(blolayer.variables_list[0]))
                ]

                # params (detached)
                params_i = slice_params_for_batch([p.detach() for p in ctx.params], ctx.batch_sizes, i)

                # dvars term (detached)
                dvars_i = [to_torch(dvars_numpy[j][i], ctx.dtype, ctx.device).detach() for j in range(len(vars_t))]

                # inequality duals (use same cleaning as cvxpy-branch)
                lam_list = []
                mask_list = []
                for j in range(num_scalar_ineq):
                    lam_np = scalar_ineq_dual[j][i]
                    lam_np = np.where(lam_np < -1e-6, lam_np, np.maximum(lam_np, 0.0))
                    sl = scalar_ineq_slack[j][i]

                    mask = (sl <= blolayer.slack_tol).astype(np.float64)
                    if mask.sum() > max(1, y_dim - num_eq):
                        k = int(max(1, y_dim - num_eq))
                        lam_flat = lam_np.reshape(-1)
                        idx = np.argpartition(lam_flat, -k)[-k:]
                        mask_flat = np.zeros_like(lam_flat, dtype=np.float64)
                        mask_flat[idx] = 1.0
                        mask = mask_flat.reshape(lam_np.shape)

                    lam_list.append(to_torch(lam_np, ctx.dtype, ctx.device).detach())
                    mask_list.append(to_torch(mask, ctx.dtype, ctx.device).detach())

                # soc dual params (clamped like cvxpy-branch)
                soc0 = [to_torch(np.maximum(soc_dual_0[j][i], 0.0), ctx.dtype, ctx.device).detach() for j in range(len(soc_dual_0))]
                soc1 = [to_torch(soc_dual_1[j][i], ctx.dtype, ctx.device).detach() for j in range(len(soc_dual_1))]

                # eq_dual and soc_lam are NOT in perturbed objective -> set to zero for objective eval
                # (shapes must match phi_torch signature)
                eq0 = []
                if len(blolayer._eq_expr_torch) > 0:
                    # compute once for shapes
                    with torch.no_grad():
                        tmp = [f(*vars_t, *params_i) for f in blolayer._eq_expr_torch]
                    eq0 = [torch.zeros_like(r) for r in tmp]

                soclam0 = []
                if len(blolayer._soclin_expr_torch) > 0:
                    with torch.no_grad():
                        tmp = [c(*vars_t, *params_i, *soc0, *soc1) for c in blolayer._soclin_expr_torch]
                    soclam0 = [torch.zeros_like(r) for r in tmp]

                # multipliers for constraints (these are what we'll return as "dual-like" values)
                nu = [torch.zeros_like(r) for r in eq0]             # eq constraints
                eta = [torch.zeros_like(m) for m in mask_list]      # active eq constraints (mask*g==0), init shape placeholder
                mu = [torch.zeros_like(r) for r in soclam0]         # soc_lin constraints

                for _ in range(ITERS):
                    # objective(new_objective) = phi_torch with eq0/soclam0 + dv_term
                    base = blolayer.phi_torch(*vars_t, *params_i, *eq0, *lam_list, *soc0, *soc1, *soclam0)
                    dv_term = sum((dv * v).sum() for dv, v in zip(dvars_i, vars_t)) / blolayer.alpha
                    obj = base + dv_term

                    # residuals: eq, active(mask*g), soc_lin
                    eq_res = [f(*vars_t, *params_i) for f in blolayer._eq_expr_torch]
                    g_res  = [g(*vars_t, *params_i) for g in blolayer._ineq_expr_torch]
                    act_res = [m * r for m, r in zip(mask_list, g_res)]
                    soc_res = [c(*vars_t, *params_i, *soc0, *soc1) for c in blolayer._soclin_expr_torch]

                    # make eta shapes match act_res (first iter)
                    if len(eta) == len(act_res) and len(act_res) > 0 and (eta[0].shape != act_res[0].shape):
                        eta = [torch.zeros_like(r) for r in act_res]

                    # augmented lagrangian
                    L = obj
                    for y, r in zip(nu, eq_res):
                        L = L + (y * r).sum() + 0.5 * RHO * (r * r).sum()
                    for y, r in zip(eta, act_res):
                        L = L + (y * r).sum() + 0.5 * RHO * (r * r).sum()
                    for y, r in zip(mu, soc_res):
                        L = L + (y * r).sum() + 0.5 * RHO * (r * r).sum()

                    grads = torch.autograd.grad(L, vars_t, allow_unused=True, retain_graph=False, create_graph=False)
                    vars_t = [
                        (v - LR * (g if g is not None else torch.zeros_like(v))).detach().requires_grad_(True)
                        for v, g in zip(vars_t, grads)
                    ]

                    # multiplier ascent (detach residuals)
                    with torch.no_grad():
                        eq_res = [f(*vars_t, *params_i) for f in blolayer._eq_expr_torch]
                        g_res  = [g(*vars_t, *params_i) for g in blolayer._ineq_expr_torch]
                        act_res = [m * r for m, r in zip(mask_list, g_res)]
                        soc_res = [c(*vars_t, *params_i, *soc0, *soc1) for c in blolayer._soclin_expr_torch]

                        nu  = [(y + RHO * r).detach() for y, r in zip(nu, eq_res)]
                        eta = [(y + RHO * r).detach() for y, r in zip(eta, act_res)]
                        mu  = [(y + RHO * r).detach() for y, r in zip(mu, soc_res)]

                solve_time = time.time() - t0
                setup_time = 0.0

                new_sol_vals = [to_numpy(v.detach()) for v in vars_t]

                # IMPORTANT: return shapes aligned with downstream usage
                new_eq_vals  = [to_numpy(y) for y in nu]                 # len = #eq
                new_act_vals = [to_numpy(y) for y in eta]                # len = #ineq (active eq)
                new_lam_vals = [to_numpy(y) for y in mu]                 # len = #soc_lin

                # pad empty lists correctly if no constraints
                if len(blolayer._eq_expr_torch) == 0:
                    new_eq_vals = []
                if num_scalar_ineq == 0:
                    new_act_vals = []
                if len(blolayer._soclin_expr_torch) == 0:
                    new_lam_vals = []

                return new_sol_vals, new_eq_vals, new_act_vals, new_lam_vals, setup_time, solve_time

            def _solve_one_backward(i):
                slot = i if blolayer.num_copies >= B else 0
                
                if blolayer.use_iterative_backward:
                    return _iterative_solve_perturbed(i)
                else:
                    if ctx.batch:
                        params_i = [p[i] if bs > 0 else p for p, bs in zip(ctx.params_numpy, ctx.batch_sizes)]
                    else:
                        params_i = ctx.params_numpy
                    for p_val, param_obj in zip(params_i, blolayer.param_order_list[slot]):
                        param_obj.value = p_val

                    for j, v in enumerate(blolayer.variables_list[slot]):
                        blolayer.dvar_params_list[slot][j].value = dvars_numpy[j][i]
                        v.value = ctx._warm_vars_list[i][j]

                    for j in range(len(blolayer.eq_functions_list[slot])):
                        blolayer.eq_dual_params_list[slot][j].value = eq_dual[j][i]

                    for j in range(num_scalar_ineq):
                        lam = scalar_ineq_dual[j][i]
                        lam = np.where(lam < -1e-6, lam, np.maximum(lam, 0.0))
                        sl = scalar_ineq_slack[j][i]
                        blolayer.scalar_ineq_dual_params_list[slot][j].value = lam
                        mask = (sl <= blolayer.slack_tol).astype(np.float64)
                        if mask.sum() > max(1, y_dim - num_eq):
                            k = int(max(1, y_dim - num_eq))
                            lam_flat = lam.reshape(-1)
                            idx = np.argpartition(lam_flat, -k)[-k:]
                            mask_flat = np.zeros_like(lam_flat, dtype=np.float64)
                            mask_flat[idx] = 1.0
                            mask = mask_flat.reshape(lam.shape)
                        blolayer.scalar_active_mask_params_list[slot][j].value = mask

                    for j in range(num_soc_cones):
                        u = np.maximum(soc_dual_0[j][i], 0.0)
                        v = soc_dual_1[j][i]
                        blolayer.soc_dual_params_0_list[slot][j].value = u
                        blolayer.soc_dual_params_1_list[slot][j].value = v
                    
                    blolayer.perturbed_problem_list[slot].solve(solver=cp.SCS, warm_start=False, ignore_dpp=True, max_iters=2500, eps=1e-4)

                    st = blolayer.perturbed_problem_list[slot].status
                    if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        raise RuntimeError(f"New bilevel status = {st}")

                    new_sol_vals = [v.value for v in blolayer.variables_list[slot]]
                    new_eq_vals = [c.dual_value for c in blolayer.eq_constraints_list[slot]]
                    new_act_vals = [c.dual_value for c in blolayer.active_eq_constraints_list[slot]]
                    new_lam_vals = [c.dual_value for c in blolayer.soc_lin_constraints_list[slot]]

                    setup_time = blolayer.perturbed_problem_list[slot].compilation_time
                    solve_time = blolayer.perturbed_problem_list[slot].solver_stats.solve_time
                    return new_sol_vals, new_eq_vals, new_act_vals, new_lam_vals, setup_time, solve_time

            backward_solve_time_start = time.time()
            use_mt = (ThreadPool is not None) and (B > 1) and (blolayer.num_copies >= B)
            if use_mt:
                with threadpool_limits(limits=1):
                    pool = ThreadPool(processes=min(B, n_threads))
                    try:
                        results = pool.map(_solve_one_backward, range(B))
                    finally:
                        pool.close()
            else:
                raise ValueError("Not implemented")
                # results = [_solve_one_backward(i) for i in range(B)]
            blolayer.backward_solve_time = time.time() - backward_solve_time_start

            avg_setup_time = 0.0
            avg_solve_time = 0.0
            for i, (new_sol_vals, new_eq_vals, new_act_vals, new_lam_vals, setup_time, solve_time) in enumerate(results):
                for j in range(len(new_sol_lagrangian)):
                    new_sol_lagrangian[j][i, ...] = new_sol_vals[j]
                for j in range(len(new_eq_dual)):
                    new_eq_dual[j][i, ...] = new_eq_vals[j]
                for j in range(len(new_active_dual)):
                    new_active_dual[j][i, ...] = new_act_vals[j]
                    new_scalar_ineq_dual[j][i, ...] = new_act_vals[j]
                for j in range(len(new_soc_lam)):
                    new_soc_lam[j][i, ...] = new_lam_vals[j]

                avg_setup_time += setup_time
                avg_solve_time += solve_time

            blolayer.backward_setup_time = avg_setup_time / B
            blolayer.average_backward_solve_time = avg_solve_time / B

            new_sol = [to_torch(v, ctx.dtype, ctx.device) for v in new_sol_lagrangian]
            new_eq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_eq_dual]
            old_eq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in eq_dual]
            old_scalar_ineq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in scalar_ineq_dual]
            new_scalar_ineq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_scalar_ineq_dual]
            old_soc_dual_0_torch = [to_torch(v, ctx.dtype, ctx.device) for v in soc_dual_0 if v is not None]
            old_soc_dual_1_torch = [to_torch(v, ctx.dtype, ctx.device) for v in soc_dual_1 if v is not None]
            old_soc_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in soc_lam if v is not None]
            new_soc_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_soc_lam]

            params_req2 = []
            for p, need in zip(ctx.params, req_grad_mask):
                q = p.detach().clone()
                if need:
                    q.requires_grad_(True)
                params_req2.append(q)

            if ctx.device != torch.device("cpu"):
                torch.set_default_device(torch.device(ctx.device))

            loss = 0.0
            with torch.enable_grad():
                for i in range(B):
                    vars_new_i = [v[i] for v in new_sol]
                    vars_old_i = [to_torch(sol_numpy[j][i], ctx.dtype, ctx.device) for j in range(len(blolayer.variables_list[0]))]
                    params_i = slice_params_for_batch(params_req2, ctx.batch_sizes, i)
                    new_eq_dual_i = [d[i] for d in new_eq_dual_torch]
                    old_eq_dual_i = [d[i] for d in old_eq_dual_torch]
                    new_scalar_ineq_dual_i = [d[i] for d in new_scalar_ineq_dual_torch]
                    old_scalar_ineq_dual_i = [d[i] for d in old_scalar_ineq_dual_torch]
                    new_soc_lam_i = [d[i] for d in new_soc_lam_torch]
                    old_soc_lam_i = [d[i] for d in old_soc_lam_torch]
                    old_soc_dual_0_i = [d[i] for d in old_soc_dual_0_torch]
                    old_soc_dual_1_i = [d[i] for d in old_soc_dual_1_torch]
                    phi_new_i = blolayer.phi_torch(*vars_new_i, *params_i, *new_eq_dual_i, *new_scalar_ineq_dual_i, *old_soc_dual_0_i, *old_soc_dual_1_i, *new_soc_lam_i)
                    phi_old_i = blolayer.phi_torch(*vars_old_i, *params_i, *old_eq_dual_i, *old_scalar_ineq_dual_i, *old_soc_dual_0_i, *old_soc_dual_1_i, *old_soc_lam_i)
                    loss = loss + (phi_new_i - phi_old_i)
                loss = blolayer.alpha * loss

            grads_req = torch.autograd.grad(
                outputs=loss,
                inputs=[q for q, need in zip(params_req2, req_grad_mask) if need],
                allow_unused=True,
                retain_graph=False,
            )

            grads = []
            it = iter(grads_req)
            for need in req_grad_mask:
                grads.append(next(it) if need else None)

            if _compute_cos_sim:
                with torch.no_grad():
                    total_l2 = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads if g is not None))
                    total_inf = max((g.detach().float().abs().max() for g in grads if g is not None), default=torch.tensor(0.0))
                wandb.log({"grad_l2": total_l2, "grad_inf": total_inf})

            return tuple(grads)

    return _BLOLayerFnFn.apply