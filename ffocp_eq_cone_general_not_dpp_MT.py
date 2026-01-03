import os
from copy import copy
from concurrent.futures import ThreadPoolExecutor

import cvxpy as cp
import numpy as np
import torch
from cvxtorch import TorchExpression
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC

from utils import to_numpy, to_torch, slice_params_for_batch


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


def _cvx_sum_or_zero(terms):
    return cp.sum(terms) if len(terms) > 0 else cp.Constant(0.0)


def _has_pnorm_atom(expr) -> bool:
    """Recursively detect whether expr contains a pnorm atom."""
    try:
        nm_fn = getattr(expr, "name", None)
        if callable(nm_fn):
            nm = nm_fn()
            if nm in {"pnorm", "norm1", "norm_inf"}:
                return True
    except Exception:
        pass

    try:
        cls = expr.__class__.__name__.lower()
        if cls in {"pnorm", "norm1", "norminf", "norm_inf"}:
            return True
    except Exception:
        pass

    for a in getattr(expr, "args", []) or []:
        if _has_pnorm_atom(a):
            return True
    return False


class _BLOLayerSingle(torch.nn.Module):
    """
    A self-contained single-problem FFO/BLO layer core:
    - Owns its own CVXPY Problem objects and Parameters (thread-safe when used per-thread).
    - Exposes attributes consumed by the multi-thread wrapper.
    """

    def __init__(
        self,
        problem: cp.Problem,
        parameters,
        variables,
        alpha: float = 100.0,
        dual_cutoff: float = 1e-3,
        slack_tol: float = 1e-8,
        eps: float = 1e-13,
        compute_cos_sim: bool = False,
    ):
        super().__init__()

        # ---- objective ----
        obj = problem.objective
        if isinstance(obj, cp.Minimize):
            objective_expr = obj.expr
        elif isinstance(obj, cp.Maximize):
            objective_expr = -obj.expr
        else:
            objective_expr = getattr(obj, "expr", None)
            if objective_expr is None:
                raise ValueError("Unsupported objective type; expected Minimize/Maximize.")

        # ---- split constraints ----
        eq_funcs = []
        scalar_ineq_funcs = []
        soc_constraints = []
        exp_cones = []
        psd_cones = []

        for c in problem.constraints:
            if isinstance(c, cp.constraints.zero.Equality):
                eq_funcs.append(c.expr)
            elif isinstance(c, cp.constraints.nonpos.Inequality):
                scalar_ineq_funcs.append(c.expr)
            elif isinstance(c, SOC):
                soc_constraints.append(c)
            elif isinstance(c, ExpCone):
                exp_cones.append(c)
            elif isinstance(c, PSD):
                psd_cones.append(c)
            else:
                raise ValueError(f"Unsupported constraint type: {type(c)}")

        # ---- store core ----
        self.objective = objective_expr
        self.eq_functions = eq_funcs
        self.scalar_ineq_functions = scalar_ineq_funcs
        self.soc_constraints = soc_constraints
        self.exp_cones = exp_cones
        self.psd_cones = psd_cones

        self.param_order = list(parameters)
        self.variables = list(variables)

        self.alpha = float(alpha)
        self.dual_cutoff = float(dual_cutoff)
        self.slack_tol = float(slack_tol)
        self.eps = float(eps)
        self._compute_cos_sim = bool(compute_cos_sim)

        # ---- original problem (for forward) ----
        self.eq_constraints = [f == 0 for f in self.eq_functions]
        self.scalar_ineq_constraints = [g <= 0 for g in self.scalar_ineq_functions]
        self.problem = cp.Problem(
            cp.Minimize(self.objective),
            self.eq_constraints + self.scalar_ineq_constraints + self.soc_constraints + self.exp_cones + self.psd_cones,
        )

        # ---- dvar params ----
        self.dvar_params = [cp.Parameter(shape=v.shape) for v in self.variables]

        # ---- dual params for eq / scalar ineq ----
        self.eq_dual_params = [cp.Parameter(shape=f.shape) for f in self.eq_functions]
        self.scalar_ineq_dual_params = [cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions]

        # ---- active masks for scalar inequalities ----
        self.scalar_active_mask_params = [cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions]

        # ---- SOC duals ----
        self.soc_dual_params_0 = [cp.Parameter(shape=c.dual_variables[0].shape, nonneg=True) for c in self.soc_constraints]
        self.soc_dual_params_1 = [cp.Parameter(shape=c.dual_variables[1].shape) for c in self.soc_constraints]
        soc_dual_product = _cvx_sum_or_zero([
            cp.multiply(cp.pnorm(c.args[1].expr, p=2) - c.args[0].expr, u)
            for u, c in zip(self.soc_dual_params_0, self.soc_constraints)
        ])
        self.soc_lin_constraints = [
            (self.soc_dual_params_1[j].T @ self.soc_constraints[j].args[1].expr
             + cp.multiply(self.soc_constraints[j].args[0].expr, self.soc_dual_params_0[j])) == 0
            for j in range(len(self.soc_constraints))
        ]

        # ---- ExpCone duals ----
        # ExpCone has 3 primal args; dual is typically a 3-vector.
        self.exp_dual_params = [cp.Parameter(shape=c.dual_variables[0].shape) for c in self.exp_cones]
        exp_dual_product = _cvx_sum_or_zero([
            cp.sum(cp.multiply(u, c.expr)) for u, c in zip(self.exp_dual_params, self.exp_cones)
        ])

        # ---- PSD duals ----
        self.psd_dual_params = [cp.Parameter(shape=c.dual_variables[0].shape) for c in self.psd_cones]
        psd_dual_product = _cvx_sum_or_zero([
            cp.sum(cp.multiply(u, c.expr)) for u, c in zip(self.psd_dual_params, self.psd_cones)
        ])

        # ---- pnorm tangent support for scalar inequalities ----
        self.pnorm_ineq_ids = []
        self.non_pnorm_scalar_ids = []
        self.pnorm_xstar_params = []
        self.pnorm_grad_params = []
        self.pnorm_tangent_constraints = []
        self.pnorm_g_torch = []

        for j, g in enumerate(self.scalar_ineq_functions):
            is_scalar = int(np.prod(g.shape)) == 1
            is_pnorm = is_scalar and _has_pnorm_atom(g)
            if not is_pnorm:
                self.non_pnorm_scalar_ids.append(j)
                continue

            local_id = len(self.pnorm_ineq_ids)
            self.pnorm_ineq_ids.append(j)

            xs = []
            gs = []
            for v in self.variables:
                xs.append(cp.Parameter(shape=v.shape))
                gs.append(cp.Parameter(shape=v.shape))
            self.pnorm_xstar_params.append(xs)
            self.pnorm_grad_params.append(gs)

            lin = cp.Constant(0.0)
            for v_id, v in enumerate(self.variables):
                dv = v - self.pnorm_xstar_params[local_id][v_id]
                lin += cp.sum(cp.multiply(self.pnorm_grad_params[local_id][v_id], dv))

            self.pnorm_tangent_constraints.append(cp.multiply(self.scalar_active_mask_params[j], lin) == 0)

            self.pnorm_g_torch.append(
                TorchExpression(
                    g,
                    provided_vars_list=[*self.variables, *self.param_order],
                ).torch_expression
            )

        # ---- perturbed problem ----
        vars_dvars_product = _cvx_sum_or_zero([cp.sum(cp.multiply(dv, v)) for dv, v in zip(self.dvar_params, self.variables)])
        scalar_ineq_dual_product = _cvx_sum_or_zero([
            cp.sum(cp.multiply(lm, g)) for lm, g in zip(self.scalar_ineq_dual_params, self.scalar_ineq_functions)
        ])

        new_objective = (1.0 / self.alpha) * vars_dvars_product + self.objective
        new_objective += scalar_ineq_dual_product + soc_dual_product + exp_dual_product + psd_dual_product

        self.active_eq_constraints = [
            cp.multiply(self.scalar_active_mask_params[j], self.scalar_ineq_functions[j]) == 0
            for j in self.non_pnorm_scalar_ids
        ]

        self.perturbed_problem = cp.Problem(
            cp.Minimize(new_objective),
            self.eq_constraints + self.active_eq_constraints + self.soc_lin_constraints + self.pnorm_tangent_constraints,
        )

        # ---- torch expressions for loss components ----
        # phi = \tilde g  (objective only)
        self.phi_torch = TorchExpression(
            self.objective,
            provided_vars_list=[*self.variables, *self.param_order],
        ).torch_expression

        # eq dual term: <lambda_eq, f>
        eq_terms = [cp.sum(cp.multiply(du, f)) for du, f in zip(self.eq_dual_params, self.eq_functions)]
        self.eq_dual_term_torch = TorchExpression(
            _cvx_sum_or_zero(eq_terms),
            provided_vars_list=[*self.variables, *self.param_order, *self.eq_dual_params],
        ).torch_expression

        # scalar ineq dual term: <lambda_ineq, g>
        ineq_terms = [cp.sum(cp.multiply(du, g)) for du, g in zip(self.scalar_ineq_dual_params, self.scalar_ineq_functions)]
        self.ineq_dual_term_torch = TorchExpression(
            _cvx_sum_or_zero(ineq_terms),
            provided_vars_list=[*self.variables, *self.param_order, *self.scalar_ineq_dual_params],
        ).torch_expression

        # ExpCone dual term
        exp_terms = [cp.sum(cp.multiply(du, c.expr)) for du, c in zip(self.exp_dual_params, self.exp_cones)]
        self.exp_dual_term_torch = TorchExpression(
            _cvx_sum_or_zero(exp_terms),
            provided_vars_list=[*self.variables, *self.param_order, *self.exp_dual_params],
        ).torch_expression

        # PSD dual term
        psd_terms = [cp.sum(cp.multiply(du, c.expr)) for du, c in zip(self.psd_dual_params, self.psd_cones)]
        self.psd_dual_term_torch = TorchExpression(
            _cvx_sum_or_zero(psd_terms),
            provided_vars_list=[*self.variables, *self.param_order, *self.psd_dual_params],
        ).torch_expression


def BLOLayerMT(
    problem_list,
    parameters_list,
    variables_list,
    alpha: float = 100.0,
    dual_cutoff: float = 1e-3,
    slack_tol: float = 1e-8,
    eps: float = 1e-13,
    compute_cos_sim: bool = False,
    max_workers: int | None = None,
    backward_eps: float = 1e-3,
):
    return _BLOLayerMT(
        problem_list=problem_list,
        parameters_list=parameters_list,
        variables_list=variables_list,
        alpha=alpha,
        dual_cutoff=dual_cutoff,
        slack_tol=slack_tol,
        eps=eps,
        compute_cos_sim=compute_cos_sim,
        max_workers=max_workers,
        backward_eps=backward_eps,
    )


class _BLOLayerMT(torch.nn.Module):
    def __init__(
        self,
        problem_list,
        parameters_list,
        variables_list,
        alpha,
        dual_cutoff,
        slack_tol,
        eps,
        compute_cos_sim,
        max_workers: int | None,
        backward_eps,
    ):
        super().__init__()
        if not (len(problem_list) == len(parameters_list) == len(variables_list)):
            raise ValueError("problem_list / parameters_list / variables_list must have the same length.")
        self.num_problems = len(problem_list)
        self.max_workers = max_workers or min(os.cpu_count() or 1, self.num_problems)

        layers = []
        for prob_i, params_i, vars_i in zip(problem_list, parameters_list, variables_list):
            layers.append(
                _BLOLayerSingle(
                    prob_i,
                    parameters=params_i,
                    variables=vars_i,
                    alpha=alpha,
                    dual_cutoff=dual_cutoff,
                    slack_tol=slack_tol,
                    eps=eps,
                    compute_cos_sim=compute_cos_sim,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

        self.alpha = float(alpha)
        self.dual_cutoff = float(dual_cutoff)
        self.slack_tol = float(slack_tol)
        self.eps = float(eps)
        self.backward_eps = float(backward_eps)
        self._compute_cos_sim = bool(compute_cos_sim)

        self.forward_solve_time = 0
        self.backward_solve_time = 0
        self.forward_setup_time = 0
        self.backward_setup_time = 0

    def forward(self, *params, solver_args=None):
        if solver_args is None:
            solver_args = {}
        solver = solver_args.get("solver", cp.SCS)
        if solver == cp.SCS:
            default_solver_args = dict(
                solver=cp.SCS,
                warm_start=False,
                ignore_dpp=True,
                max_iters=2500,
                eps=self.eps,
                verbose=False,
            )
        else:
            default_solver_args = dict(ignore_dpp=False)

        solver_args = {**default_solver_args, **solver_args}
        return _BLOLayerMTFn.apply(self, solver_args, *params)


class _BLOLayerMTFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mt: _BLOLayerMT, solver_args: dict, *params):
        if mt.num_problems == 0:
            raise ValueError("Empty problem_list.")

        ctx.mt = mt
        ctx.layers = mt.layers
        ctx.solver_args = solver_args
        ctx.dtype = params[0].dtype
        ctx.device = params[0].device

        # ---- infer batching from first layer's parameter templates ----
        ref = mt.layers[0]
        batch_sizes = []
        for i, (p, q) in enumerate(zip(params, ref.param_order)):
            if p.dtype != ctx.dtype or p.device != ctx.device:
                raise ValueError(f"Parameter {i} dtype/device mismatch.")
            if p.ndimension() == q.ndim:
                bs = 0
            elif p.ndimension() == q.ndim + 1:
                bs = int(p.size(0))
            else:
                raise ValueError(f"Invalid dim for parameter {i}: got {p.ndimension()}, expected {q.ndim} or {q.ndim+1}.")
            batch_sizes.append(bs)

            p_shape = p.shape if bs == 0 else p.shape[1:]
            if tuple(p_shape) != tuple(q.shape):
                raise ValueError(f"Parameter {i} shape mismatch: expected {q.shape}, got {p.shape}.")

        ctx.batch_sizes = np.array(batch_sizes, dtype=int)
        ctx.batch = bool(np.any(ctx.batch_sizes > 0))
        if ctx.batch:
            nonzero = ctx.batch_sizes[ctx.batch_sizes > 0]
            B = int(nonzero[0])
            if np.any(nonzero != B):
                raise ValueError(f"Inconsistent batch sizes: {ctx.batch_sizes}.")
        else:
            B = 1
        if B != mt.num_problems and ctx.batch:
            raise ValueError(f"Batch size ({B}) must equal number of problems ({mt.num_problems}).")
        ctx.batch_size = B

        # ---- alloc storages (stacked across problems) ----
        num_vars = len(ref.variables)
        sol_numpy = [np.empty((B,) + v.shape, dtype=float) for v in ref.variables]
        eq_dual = [np.empty((B,) + f.shape, dtype=float) for f in ref.eq_functions]
        scalar_ineq_dual = [np.empty((B,) + g.shape, dtype=float) for g in ref.scalar_ineq_functions]
        scalar_ineq_slack = [np.empty((B,) + g.shape, dtype=float) for g in ref.scalar_ineq_functions]

        soc_dual_0 = [np.empty((B,) + c.dual_variables[0].shape, dtype=float) for c in ref.soc_constraints]
        soc_dual_1 = [np.empty((B,) + c.dual_variables[1].shape, dtype=float) for c in ref.soc_constraints]

        exp_dual = [np.empty((B,) + c.dual_variables[0].shape, dtype=float) for c in ref.exp_cones]
        psd_dual = [np.empty((B,) + c.dual_variables[0].shape, dtype=float) for c in ref.psd_cones]

        # pnorm tangent caches
        pnorm_xstar = []
        pnorm_grad = []
        for _local_id in range(len(ref.pnorm_ineq_ids)):
            pnorm_xstar.append([np.empty((B,) + v.shape, dtype=float) for v in ref.variables])
            pnorm_grad.append([np.empty((B,) + v.shape, dtype=float) for v in ref.variables])

        def _slice_params_torch(i: int):
            if ctx.batch:
                return [p[i] if bs > 0 else p for p, bs in zip(params, ctx.batch_sizes)]
            return list(params)

        def _solve_one(i: int):
            layer = ctx.layers[i]

            params_i_t = _slice_params_torch(i)
            params_i_np = [to_numpy(t) for t in params_i_t]

            for pval, pparam in zip(params_i_np, layer.param_order):
                pparam.value = pval

            # solve forward problem
            try:
                layer.problem.solve(**ctx.solver_args)
            except Exception:
                layer.problem.solve(solver=cp.OSQP, warm_start=False, verbose=False)

            if layer.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"[forward] problem {i} status: {layer.problem.status}")

            # primal
            for v_id, v in enumerate(layer.variables):
                sol_numpy[v_id][i, ...] = v.value

            # eq duals
            for c_id, c in enumerate(layer.eq_constraints):
                eq_dual[c_id][i, ...] = c.dual_value

            # scalar ineq duals + slack
            for j, g_expr in enumerate(layer.scalar_ineq_functions):
                g_val = np.asarray(g_expr.value, dtype=float)
                scalar_ineq_dual[j][i, ...] = layer.scalar_ineq_constraints[j].dual_value
                scalar_ineq_slack[j][i, ...] = np.maximum(-g_val, 0.0)

            # SOC duals
            for c_id, c in enumerate(layer.soc_constraints):
                dv0, dv1 = c.dual_value
                soc_dual_0[c_id][i, ...] = dv0
                if hasattr(dv1, "shape") and len(dv1.shape) == 2 and dv1.shape[1] == 1:
                    soc_dual_1[c_id][i, ...] = dv1.reshape(-1)
                else:
                    soc_dual_1[c_id][i, ...] = dv1

            # Exp / PSD duals
            for c_id, c in enumerate(layer.exp_cones):
                exp_dual[c_id][i, ...] = c.dual_value
            for c_id, c in enumerate(layer.psd_cones):
                psd_dual[c_id][i, ...] = c.dual_value

            # pnorm tangent: compute ∇g(x*) wrt variables using TorchExpression + autograd
            if len(layer.pnorm_ineq_ids) > 0:
                with torch.enable_grad():
                    vars_star_t = [
                        torch.tensor(sol_numpy[v_id][i, ...], dtype=ctx.dtype, device=ctx.device, requires_grad=True)
                        for v_id in range(num_vars)
                    ]
                    params_i_det = [t.detach() for t in params_i_t]
                    for local_id in range(len(layer.pnorm_ineq_ids)):
                        g_t = layer.pnorm_g_torch[local_id](*vars_star_t, *params_i_det).reshape(())
                        grads = torch.autograd.grad(
                            g_t,
                            vars_star_t,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        )
                        for v_id, gv in enumerate(grads):
                            pnorm_xstar[local_id][v_id][i, ...] = to_numpy(vars_star_t[v_id].detach())
                            if gv is None:
                                pnorm_grad[local_id][v_id][i, ...] = 0.0
                            else:
                                pnorm_grad[local_id][v_id][i, ...] = to_numpy(gv.detach())

        # run problems in parallel
        with ThreadPoolExecutor(max_workers=mt.max_workers) as ex:
            futs = [ex.submit(_solve_one, i) for i in range(B)]
            for f in futs:
                f.result()

        ctx.sol_numpy = sol_numpy
        ctx.eq_dual = eq_dual
        ctx.scalar_ineq_dual = scalar_ineq_dual
        ctx.scalar_ineq_slack = scalar_ineq_slack
        ctx.soc_dual_0 = soc_dual_0
        ctx.soc_dual_1 = soc_dual_1
        ctx.exp_dual = exp_dual
        ctx.psd_dual = psd_dual
        ctx.pnorm_xstar = pnorm_xstar
        ctx.pnorm_grad = pnorm_grad
        ctx.params = params

        # warm start: per-problem primal values
        ctx._warm_vars = [[copy(v.value) for v in layer.variables] for layer in ctx.layers]

        sol_torch = [to_torch(arr, ctx.dtype, ctx.device) for arr in sol_numpy]
        return tuple(sol_torch)

    @staticmethod
    def backward(ctx, *dvars):
        mt = ctx.mt
        layers = ctx.layers
        B = ctx.batch_size

        ref = layers[0]
        num_vars = len(ref.variables)
        num_scalar_ineq = len(ref.scalar_ineq_functions)

        # outputs to fill
        new_sol_lagrangian = [np.empty_like(ctx.sol_numpy[k]) for k in range(num_vars)]
        new_eq_dual = [np.empty_like(ctx.eq_dual[k]) for k in range(len(ref.eq_constraints))]
        new_active_dual = [np.empty((B,) + c.shape, dtype=float) for c in ref.active_eq_constraints]
        new_soc_lam = [np.zeros((B,), dtype=float) for _ in ref.soc_lin_constraints]
        new_pnorm_lam = [np.zeros((B,), dtype=float) for _ in ref.pnorm_tangent_constraints]
        old_pnorm_lam = [np.zeros((B,), dtype=float) for _ in ref.pnorm_tangent_constraints]
        pnorm_masks = [np.zeros((B,), dtype=float) for _ in ref.pnorm_tangent_constraints]

        new_exp_dual = [np.empty_like(ctx.exp_dual[k]) for k in range(len(ref.exp_cones))]
        new_psd_dual = [np.empty_like(ctx.psd_dual[k]) for k in range(len(ref.psd_cones))]

        def _slice_params_torch(i: int, use_req=False, params_src=None):
            params_src = params_src if params_src is not None else ctx.params
            if ctx.batch:
                return [p[i] if bs > 0 else p for p, bs in zip(params_src, ctx.batch_sizes)]
            return list(params_src)

        def _solve_perturbed_one(i: int):
            layer = layers[i]

            # set parameters for this problem
            params_i_t = _slice_params_torch(i)
            params_i_np = [to_numpy(t) for t in params_i_t]
            for pval, pparam in zip(params_i_np, layer.param_order):
                pparam.value = pval

            # set dvar params + warm start
            for j, v in enumerate(layer.variables):
                dval = to_numpy(dvars[j][i] if ctx.batch else dvars[j])
                layer.dvar_params[j].value = dval
                v.value = ctx._warm_vars[i][j]

            # old eq dual params
            for j in range(len(layer.eq_functions)):
                layer.eq_dual_params[j].value = ctx.eq_dual[j][i]

            # scalar ineq dual params + active masks (same logic as before)
            y_dim = int(np.prod(to_numpy(dvars[0][i] if ctx.batch else dvars[0]).shape))
            num_eq = int(np.prod(ctx.eq_dual[0][i].shape)) if len(ctx.eq_dual) > 0 else 0
            cap = int(max(1, y_dim - num_eq))

            scalar_candidates = []
            for j in range(num_scalar_ineq):
                gshape = layer.scalar_ineq_functions[j].shape
                if int(np.prod(gshape)) != 1:
                    continue
                sl_s = float(np.asarray(ctx.scalar_ineq_slack[j][i]).reshape(()))
                lam_s = float(np.asarray(ctx.scalar_ineq_dual[j][i]).reshape(()))
                lam_s = 0.0 if lam_s < -1e-8 else max(lam_s, 0.0)
                if sl_s <= layer.slack_tol and lam_s >= layer.dual_cutoff:
                    scalar_candidates.append((lam_s, j))

            scalar_candidates.sort(key=lambda t: t[0])
            active_scalar = set([j for _, j in scalar_candidates[-cap:]]) if len(scalar_candidates) > cap else set([j for _, j in scalar_candidates])

            for j in range(num_scalar_ineq):
                lam = np.asarray(ctx.scalar_ineq_dual[j][i], dtype=float)
                lam = np.where(lam < -1e-8, lam, np.maximum(lam, 0.0))
                layer.scalar_ineq_dual_params[j].value = lam

                gshape = layer.scalar_ineq_functions[j].shape
                if int(np.prod(gshape)) == 1:
                    layer.scalar_active_mask_params[j].value = 1.0 if (j in active_scalar) else 0.0
                else:
                    sl = np.asarray(ctx.scalar_ineq_slack[j][i], dtype=float)
                    mask = (sl <= layer.slack_tol).astype(np.float64)
                    cap_vec = int(max(1, y_dim - num_eq))
                    if mask.sum() > cap_vec:
                        lam_flat = lam.reshape(-1)
                        idx = np.argpartition(lam_flat, -cap_vec)[-cap_vec:]
                        mask_flat = np.zeros_like(lam_flat, dtype=np.float64)
                        mask_flat[idx] = 1.0
                        mask = mask_flat.reshape(lam.shape)
                    layer.scalar_active_mask_params[j].value = mask

            # SOC dual params (old)
            for j in range(len(layer.soc_constraints)):
                layer.soc_dual_params_0[j].value = np.maximum(ctx.soc_dual_0[j][i], 0.0)
                layer.soc_dual_params_1[j].value = ctx.soc_dual_1[j][i]

            # Exp/PSD dual params (old)
            for j in range(len(layer.exp_cones)):
                layer.exp_dual_params[j].value = ctx.exp_dual[j][i]
            for j in range(len(layer.psd_cones)):
                layer.psd_dual_params[j].value = ctx.psd_dual[j][i]

            # pnorm tangent params (x*, grad) and record masks
            for local_id, j_scalar in enumerate(layer.pnorm_ineq_ids):
                for v_id in range(num_vars):
                    layer.pnorm_xstar_params[local_id][v_id].value = ctx.pnorm_xstar[local_id][v_id][i]
                    layer.pnorm_grad_params[local_id][v_id].value = ctx.pnorm_grad[local_id][v_id][i]
                mval = float(np.asarray(layer.scalar_active_mask_params[j_scalar].value).reshape(()))
                pnorm_masks[local_id][i] = mval

            # solve perturbed
            bargs = dict(ctx.solver_args)
            bargs["warm_start"] = bargs.get("warm_start", False)
            bargs["eps"] = mt.backward_eps
            layer.perturbed_problem.solve(**bargs)
            if layer.perturbed_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                layer.perturbed_problem.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, warm_start=True, verbose=False)

            # collect new primal
            for j, v in enumerate(layer.variables):
                new_sol_lagrangian[j][i, ...] = v.value

            # collect new eq duals
            for c_id, c in enumerate(layer.eq_constraints):
                new_eq_dual[c_id][i, ...] = c.dual_value

            # active_eq duals
            for c_id, c in enumerate(layer.active_eq_constraints):
                new_active_dual[c_id][i, ...] = c.dual_value

            # SOC lin duals (lam)
            for c_id, c in enumerate(layer.soc_lin_constraints):
                dv = c.dual_value
                new_soc_lam[c_id][i] = 0.0 if dv is None else float(np.asarray(dv).reshape(()))

            # pnorm tangent duals
            for c_id, c in enumerate(layer.pnorm_tangent_constraints):
                dv = c.dual_value
                lam_val = 0.0 if dv is None else float(np.asarray(dv).reshape(()))
                j_scalar = layer.pnorm_ineq_ids[c_id]
                mval = float(np.asarray(layer.scalar_active_mask_params[j_scalar].value).reshape(()))
                if mval < 0.5:
                    lam_val = 0.0
                new_pnorm_lam[c_id][i] = lam_val

            # Exp/PSD duals (new)
            for c_id, c in enumerate(layer.exp_cones):
                new_exp_dual[c_id][i, ...] = c.dual_value
            for c_id, c in enumerate(layer.psd_cones):
                new_psd_dual[c_id][i, ...] = c.dual_value

        with ThreadPoolExecutor(max_workers=mt.max_workers) as ex:
            futs = [ex.submit(_solve_perturbed_one, i) for i in range(B)]
            for f in futs:
                f.result()

        # torch conversions
        new_sol = [to_torch(v, ctx.dtype, ctx.device) for v in new_sol_lagrangian]
        vars_old = [to_torch(ctx.sol_numpy[j], ctx.dtype, ctx.device) for j in range(num_vars)]
        new_eq_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_eq_dual]
        old_eq_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in ctx.eq_dual]
        old_scalar_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in ctx.scalar_ineq_dual]
        new_active_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_active_dual]
        new_exp_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_exp_dual]
        old_exp_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in ctx.exp_dual]
        new_psd_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_psd_dual]
        old_psd_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in ctx.psd_dual]
        new_soc_lam_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_soc_lam]
        old_soc_lam_t = [to_torch(np.zeros_like(v), ctx.dtype, ctx.device) for v in new_soc_lam]  # old = 0

        new_pnorm_lam_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_pnorm_lam]
        old_pnorm_lam_t = [to_torch(v, ctx.dtype, ctx.device) for v in old_pnorm_lam]

        # ---- prepare params for autograd with minimal copies ----
        params_req = []
        req_grad_mask = []
        for p in ctx.params:
            need = bool(getattr(p, "requires_grad", False))
            q = p.detach()
            if need:
                q.requires_grad_(True)
            params_req.append(q)
            req_grad_mask.append(need)

        # ---- compute loss and grads ----
        loss = 0.0
        with torch.enable_grad():
            for i in range(B):
                layer = layers[i]

                vars_new_i = [v[i] for v in new_sol]
                vars_old_i = [v[i] for v in vars_old]
                params_i = slice_params_for_batch(params_req, ctx.batch_sizes, i) if ctx.batch else params_req

                new_eq_dual_i = [d[i] for d in new_eq_dual_t]
                old_eq_dual_i = [d[i] for d in old_eq_dual_t]

                non_pnorm_set = set(layer.non_pnorm_scalar_ids)
                pnorm_set = set(layer.pnorm_ineq_ids)
                pnorm_map = {j: local_id for local_id, j in enumerate(layer.pnorm_ineq_ids)}

                new_scalar_dual_full_i = []
                ptr = 0
                for j in range(num_scalar_ineq):
                    if j in non_pnorm_set:
                        new_scalar_dual_full_i.append(new_active_dual_t[ptr][i])
                        ptr += 1
                    elif j in pnorm_set:
                        lid = pnorm_map[j]
                        new_scalar_dual_full_i.append(new_pnorm_lam_t[lid][i])
                    else:
                        new_scalar_dual_full_i.append(old_scalar_dual_t[j][i])

                old_scalar_dual_full_i = [d[i] for d in old_scalar_dual_t]

                new_exp_dual_i = [d[i] for d in new_exp_dual_t]
                old_exp_dual_i = [d[i] for d in old_exp_dual_t]
                new_psd_dual_i = [d[i] for d in new_psd_dual_t]
                old_psd_dual_i = [d[i] for d in old_psd_dual_t]

                phi_new_i = layer.phi_torch(*vars_new_i, *params_i)
                phi_old_i = layer.phi_torch(*vars_old_i, *params_i)

                eq_new = layer.eq_dual_term_torch(*vars_old_i, *params_i, *new_eq_dual_i)
                eq_old = layer.eq_dual_term_torch(*vars_old_i, *params_i, *old_eq_dual_i)

                ineq_new = layer.ineq_dual_term_torch(*vars_old_i, *params_i, *new_scalar_dual_full_i)
                ineq_old = layer.ineq_dual_term_torch(*vars_old_i, *params_i, *old_scalar_dual_full_i)

                exp_new = layer.exp_dual_term_torch(*vars_old_i, *params_i, *new_exp_dual_i)
                exp_old = layer.exp_dual_term_torch(*vars_old_i, *params_i, *old_exp_dual_i)

                psd_new = layer.psd_dual_term_torch(*vars_old_i, *params_i, *new_psd_dual_i)
                psd_old = layer.psd_dual_term_torch(*vars_old_i, *params_i, *old_psd_dual_i)

                loss = loss + (phi_new_i + ineq_new + eq_new + exp_new + psd_new - phi_old_i - ineq_old - eq_old - exp_old - psd_old)

            loss = mt.alpha * loss

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
        return (None, None, *grads)


# Convenience aliases (single entry-point is MT)
BLOLayer = BLOLayerMT
FFOLayer = BLOLayerMT
