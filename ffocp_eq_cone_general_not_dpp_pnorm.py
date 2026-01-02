import time
import cvxpy as cp
import numpy as np
from copy import copy

import torch
from cvxtorch import TorchExpression

from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD

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


def BLOLayer(
    problem: cp.Problem,
    parameters,
    variables,
    alpha: float = 100.0,
    dual_cutoff: float = 1e-3,
    slack_tol: float = 1e-8,
    eps: float = 1e-13,
    compute_cos_sim: bool = False,
):
    """
    Fully-first-order layer.

    Supports:
      - Equalities
      - Scalar inequalities (<= 0)
      - SOC constraints (kept as in your original implementation)
      - For scalar inequalities that contain pnorm(...): build tangent space using
        first-order linearization (Eq.7 style), i.e., enforce
            mask * <∇g(x*), x - x*> == 0
        instead of mask * g(x) == 0.
    """
    obj = problem.objective
    if isinstance(obj, cp.Minimize):
        objective_expr = obj.expr
    elif isinstance(obj, cp.Maximize):
        objective_expr = -obj.expr
    else:
        objective_expr = getattr(obj, "expr", None)
        if objective_expr is None:
            raise ValueError("Unsupported objective type; expected Minimize/Maximize.")

    eq_funcs = []
    scalar_ineq_funcs = []

    cone_constraints = []
    cone_exprs_soc = []
    cone_exprs_exp = []
    cone_exprs_psd = []
    cone_exprs_other = []

    all_cone_exprs = {
        "soc": cone_exprs_soc,
        "exp": cone_exprs_exp,
        "psd": cone_exprs_psd,
        "other": cone_exprs_other,
    }

    for c in problem.constraints:
        if isinstance(c, cp.constraints.zero.Equality):
            eq_funcs.append(c.expr)
        elif isinstance(c, cp.constraints.nonpos.Inequality):
            scalar_ineq_funcs.append(c.expr)  # stores (lhs - rhs) <= 0 as expr
        else:
            cone_constraints.append(c)
            if isinstance(c, SOC):
                cone_exprs_soc.append(c)
            elif isinstance(c, ExpCone):
                cone_exprs_exp.append(c)
            elif isinstance(c, PSD):
                cone_exprs_psd.append(c)
            else:
                flat_blocks = []
                for arg in c.args:
                    flat_blocks.append(arg if arg.ndim == 1 else cp.vec(arg))
                g_expr = cp.hstack(flat_blocks)
                cone_exprs_other.append(g_expr)

    return _BLOLayer(
        objective=objective_expr,
        eq_functions=eq_funcs,
        scalar_ineq_functions=scalar_ineq_funcs,
        cone_constraints=cone_constraints,
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
    def __init__(
        self,
        objective,
        eq_functions,
        scalar_ineq_functions,
        cone_constraints,
        cone_exprs,
        parameters,
        variables,
        alpha,
        dual_cutoff,
        slack_tol,
        eps,
        _compute_cos_sim=False,
    ):
        super().__init__()

        self.objective = objective
        self.eq_functions = eq_functions
        self.scalar_ineq_functions = scalar_ineq_functions
        self.cone_constraints = cone_constraints
        self.cone_exprs = cone_exprs

        self.param_order = parameters
        self.variables = variables

        self.alpha = float(alpha)
        self.dual_cutoff = float(dual_cutoff)
        self.slack_tol = float(slack_tol)
        self.eps = float(eps)
        self._compute_cos_sim = bool(_compute_cos_sim)

        # original constraints
        self.eq_constraints = [f == 0 for f in self.eq_functions]
        self.scalar_ineq_constraints = [g <= 0 for g in self.scalar_ineq_functions]
        self.problem = cp.Problem(
            cp.Minimize(self.objective),
            self.eq_constraints + self.scalar_ineq_constraints + self.cone_constraints,
        )

        # dvar params
        self.dvar_params = [cp.Parameter(shape=v.shape) for v in self.variables]

        # dual params for eq and scalar ineq (placeholders used in ghost objective)
        self.eq_dual_params = [cp.Parameter(shape=f.shape) for f in self.eq_functions]
        self.scalar_ineq_dual_params = [
            cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions
        ]

        # active mask params for scalar inequalities
        self.scalar_active_mask_params = [
            cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions
        ]

        self.soc_constraints = list(cone_exprs.get("soc", []))
        self.soc_dual_params_0 = []
        self.soc_dual_params_1 = []
        self.soc_lam_params = []
        self.soc_lin_constraints = []
        soc_dual_product = cp.Constant(0.0)

        if len(self.soc_constraints) > 0:
            self.soc_dual_params_0 = [
                cp.Parameter(shape=c.dual_variables[0].shape, nonneg=True) for c in self.soc_constraints
            ]
            self.soc_dual_params_1 = [
                cp.Parameter(shape=c.dual_variables[1].shape) for c in self.soc_constraints
            ]

            self.soc_lam_params = [cp.Parameter(shape=()) for _ in self.soc_constraints]

            soc_dual_product = cp.sum([
                cp.multiply(cp.pnorm(c.args[1].expr, p=2) - c.args[0].expr, u)
                for u, c in zip(self.soc_dual_params_0, self.soc_constraints)
            ])

            self.soc_lin_constraints = [
                (self.soc_dual_params_1[j].T @ self.soc_constraints[j].args[1].expr
                 + cp.multiply(self.soc_constraints[j].args[0].expr, self.soc_dual_params_0[j])) == 0
                for j in range(len(self.soc_constraints))
            ]

        # For now, disallow exp/psd/other (same as your original)
        if (
            len(cone_exprs.get("exp", [])) > 0
            or len(cone_exprs.get("psd", [])) > 0
            or len(cone_exprs.get("other", [])) > 0
        ):
            raise ValueError("ExpCone/PSD/other cones are not implemented in this BLOLayer.")

        # pnorm will NOT use mask*g==0; instead they use mask*<∇g(x*), x-x*>==0.
        self.pnorm_ineq_ids = []
        self.non_pnorm_scalar_ids = []          
        self.pnorm_xstar_params = []            
        self.pnorm_grad_params = []
        self.pnorm_mask_params = []
        self.pnorm_tangent_constraints = []      # equality constraints
        self.pnorm_tangent_lam_params = []       # scalar multipliers for phi term
        self.pnorm_g_torch = []                  # TorchExpression for g (for grad wrt variables)

        for j, g in enumerate(self.scalar_ineq_functions):
            is_scalar = (int(np.prod(g.shape)) == 1)
            is_pnorm = is_scalar and _has_pnorm_atom(g)
            if is_pnorm:
                local_id = len(self.pnorm_ineq_ids)
                self.pnorm_ineq_ids.append(j)

                # x* and grad params per variable
                xs = []
                gs = []
                for v in self.variables:
                    xs.append(cp.Parameter(shape=v.shape))
                    gs.append(cp.Parameter(shape=v.shape))
                self.pnorm_xstar_params.append(xs)
                self.pnorm_grad_params.append(gs)

                # reuse the existing mask parameter for this scalar inequality
                self.pnorm_mask_params.append(self.scalar_active_mask_params[j])

                # one scalar Lagrange multiplier for tangent equality in phi
                self.pnorm_tangent_lam_params.append(cp.Parameter(shape=()))

                # tangent expression: sum_var <grad_v, v - v*> == 0
                lin = cp.Constant(0.0)
                for v_id, v in enumerate(self.variables):
                    dv = v - self.pnorm_xstar_params[local_id][v_id]
                    lin += cp.sum(cp.multiply(self.pnorm_grad_params[local_id][v_id], dv))

                self.pnorm_tangent_constraints.append(
                    cp.multiply(self.pnorm_mask_params[local_id], lin) == 0
                )

                # TorchExpression for g to get ∇g(x*) wrt variables
                self.pnorm_g_torch.append(
                    TorchExpression(
                        g,
                        provided_vars_list=[*self.variables, *self.param_order],
                    ).torch_expression
                )
            else:
                self.non_pnorm_scalar_ids.append(j)

        # ---------------- build perturbed problem ----------------
        vars_dvars_product = cp.sum([
            cp.sum(cp.multiply(dv, v)) for dv, v in zip(self.dvar_params, self.variables)
        ])

        scalar_ineq_dual_product = cp.sum([
            cp.sum(cp.multiply(lm, g))
            for lm, g in zip(self.scalar_ineq_dual_params, self.scalar_ineq_functions)
        ])

        self.new_objective = (1.0 / self.alpha) * vars_dvars_product \
                             + self.objective \
                             + scalar_ineq_dual_product \
                             + soc_dual_product

        # active eq constraints for NON-pnorm scalar ineq: mask * g == 0
        self.active_eq_constraints = []
        for j in self.non_pnorm_scalar_ids:
            g = self.scalar_ineq_functions[j]
            self.active_eq_constraints.append(cp.multiply(self.scalar_active_mask_params[j], g) == 0)

        self.perturbed_problem = cp.Problem(
            cp.Minimize(self.new_objective),
            self.eq_constraints
            + self.active_eq_constraints
            + self.soc_lin_constraints
            + self.pnorm_tangent_constraints
        )

        # ---------------- phi expression ----------------
        phi_expr = self.objective
        self.phi_torch = TorchExpression(
            phi_expr,
            provided_vars_list=[
                *self.variables,
                *self.param_order,
                *self.eq_dual_params,
                *self.scalar_ineq_dual_params,
                *self.soc_dual_params_0,
                *self.soc_dual_params_1,
                *self.soc_lam_params,
            ],
        ).torch_expression

        # ---------------- split dual terms: <lambda_eq, f> and <lambda_ineq, g> ----------------
        # eq dual term
        eq_terms = [cp.sum(cp.multiply(du, f)) for du, f in zip(self.eq_dual_params, self.eq_functions)]
        eq_dual_term_expr = _cvx_sum_or_zero(eq_terms)
        self.eq_dual_term_torch = TorchExpression(
            eq_dual_term_expr,
            provided_vars_list=[*self.variables, *self.param_order, *self.eq_dual_params],
        ).torch_expression

        # ineq dual term
        ineq_terms = [cp.sum(cp.multiply(du, g)) for du, g in zip(self.scalar_ineq_dual_params, self.scalar_ineq_functions)]
        ineq_dual_term_expr = _cvx_sum_or_zero(ineq_terms)
        self.ineq_dual_term_torch = TorchExpression(
            ineq_dual_term_expr,
            provided_vars_list=[*self.variables, *self.param_order, *self.scalar_ineq_dual_params],
        ).torch_expression

        self.info = {}

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
            )
        else:
            default_solver_args = dict(ignore_dpp=False)

        solver_args = {**default_solver_args, **solver_args}

        info = {}
        f = _BLOLayerFn(
            blolayer=self,
            solver_args=solver_args,
            _compute_cos_sim=self._compute_cos_sim,
            info=info,
        )
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

            # batch inference
            ctx.batch_sizes = []
            for i, (p, q) in enumerate(zip(params, blolayer.param_order)):
                if p.dtype != ctx.dtype:
                    raise ValueError(f"Parameter {i} dtype mismatch.")
                if p.device != ctx.device:
                    raise ValueError(f"Parameter {i} device mismatch.")

                if p.ndimension() == q.ndim:
                    bs = 0
                elif p.ndimension() == q.ndim + 1:
                    bs = p.size(0)
                    if bs == 0:
                        raise ValueError(f"Parameter {i} batch dimension is zero.")
                else:
                    raise ValueError(f"Invalid dim for parameter {i}.")
                ctx.batch_sizes.append(bs)

                p_shape = p.shape if bs == 0 else p.shape[1:]
                if not np.all(np.array(p_shape) == np.array(q.shape)):
                    raise ValueError(f"Parameter {i} shape mismatch. expected {q.shape}, got {p.shape}")

            ctx.batch_sizes = np.array(ctx.batch_sizes)
            ctx.batch = np.any(ctx.batch_sizes > 0)
            if ctx.batch:
                nonzero = ctx.batch_sizes[ctx.batch_sizes > 0]
                ctx.batch_size = int(nonzero[0])
                if np.any(nonzero != ctx.batch_size):
                    raise ValueError(f"Inconsistent batch sizes: {ctx.batch_sizes}")
            else:
                ctx.batch_size = 1

            B = ctx.batch_size
            params_numpy = [to_numpy(p) for p in params]

            # primal/dual storage
            sol_numpy = [np.empty((B,) + v.shape, dtype=float) for v in blolayer.variables]
            eq_dual = [np.empty((B,) + f.shape, dtype=float) for f in blolayer.eq_functions]

            scalar_ineq_dual = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.scalar_ineq_functions]
            scalar_ineq_slack = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.scalar_ineq_functions]

            # SOC dual storage
            soc_dual_0 = [np.empty((B,) + c.dual_variables[0].shape, dtype=float) for c in blolayer.soc_constraints]
            soc_dual_1 = [np.empty((B,) + c.dual_variables[1].shape, dtype=float) for c in blolayer.soc_constraints]
            soc_lam = [np.zeros((B,), dtype=float) for _ in blolayer.soc_lin_constraints]  # old = 0

            # pnorm tangent caches: x* and grad per (local_id, var_id)
            pnorm_xstar = []
            pnorm_grad = []
            for _local_id in range(len(blolayer.pnorm_ineq_ids)):
                pnorm_xstar.append([np.empty((B,) + v.shape, dtype=float) for v in blolayer.variables])
                pnorm_grad.append([np.empty((B,) + v.shape, dtype=float) for v in blolayer.variables])

            for i in range(B):
                if ctx.batch:
                    params_numpy_i = [p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)]
                else:
                    params_numpy_i = params_numpy

                for pval, pparam in zip(params_numpy_i, blolayer.param_order):
                    pparam.value = pval

                # solve forward problem
                try:
                    blolayer.problem.solve(**ctx.solver_args)
                except Exception:
                    blolayer.problem.solve(solver=cp.OSQP, warm_start=False, verbose=False)

                if blolayer.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"Forward problem status: {blolayer.problem.status}")

                # primal
                for v_id, v in enumerate(blolayer.variables):
                    sol_numpy[v_id][i, ...] = v.value

                # eq duals
                for c_id, c in enumerate(blolayer.eq_constraints):
                    eq_dual[c_id][i, ...] = c.dual_value

                # scalar ineq duals + slack
                for j, g_expr in enumerate(blolayer.scalar_ineq_functions):
                    g_val = np.asarray(g_expr.value, dtype=float)
                    scalar_ineq_dual[j][i, ...] = blolayer.scalar_ineq_constraints[j].dual_value
                    scalar_ineq_slack[j][i, ...] = np.maximum(-g_val, 0.0)

                # SOC duals
                for c_id, c in enumerate(blolayer.soc_constraints):
                    dv0, dv1 = c.dual_value
                    soc_dual_0[c_id][i, ...] = dv0
                    if hasattr(dv1, "shape") and len(dv1.shape) == 2 and dv1.shape[1] == 1:
                        soc_dual_1[c_id][i, ...] = dv1.reshape(-1)
                    else:
                        soc_dual_1[c_id][i, ...] = dv1

                # pnorm tangent: compute ∇g(x*) wrt variables using TorchExpression + autograd
                if len(blolayer.pnorm_ineq_ids) > 0:
                    with torch.enable_grad():
                        # vars at x*
                        vars_star_t = []
                        for v_id in range(len(blolayer.variables)):
                            t = torch.tensor(sol_numpy[v_id][i, ...], dtype=ctx.dtype, device=ctx.device, requires_grad=True)
                            vars_star_t.append(t)

                        # params for this batch (detached)
                        params_i_t = [t.detach() for t in slice_params_for_batch(params, ctx.batch_sizes, i)]

                        for local_id, _j in enumerate(blolayer.pnorm_ineq_ids):
                            g_t = blolayer.pnorm_g_torch[local_id](*vars_star_t, *params_i_t)
                            g_t = g_t.reshape(())  # scalar
                            grads = torch.autograd.grad(
                                g_t,
                                vars_star_t,
                                retain_graph=False,
                                create_graph=False,
                                allow_unused=False,
                            )
                            for v_id, gv in enumerate(grads):
                                if gv is None:
                                    pnorm_grad[local_id][v_id][i, ...] = 0.0
                                else:
                                    pnorm_grad[local_id][v_id][i, ...] = to_numpy(gv.detach())
                                pnorm_xstar[local_id][v_id][i, ...] = to_numpy(vars_star_t[v_id].detach())

            # save ctx
            ctx.sol_numpy = sol_numpy
            ctx.eq_dual = eq_dual
            ctx.scalar_ineq_dual = scalar_ineq_dual
            ctx.scalar_ineq_slack = scalar_ineq_slack
            ctx.soc_dual_0 = soc_dual_0
            ctx.soc_dual_1 = soc_dual_1
            ctx.soc_lam = soc_lam
            ctx.pnorm_xstar = pnorm_xstar
            ctx.pnorm_grad = pnorm_grad

            ctx.params_numpy = params_numpy
            ctx.params = params
            ctx.blolayer = blolayer

            # warm start primal vars
            ctx._warm_vars = [copy(v.value) for v in blolayer.variables]

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
            old_soc_lam = ctx.soc_lam

            params_numpy = ctx.params_numpy

            num_scalar_ineq = len(blolayer.scalar_ineq_functions)
            num_soc_cones = len(blolayer.soc_constraints)
            num_pnorm = len(blolayer.pnorm_ineq_ids)

            # allocate new primal + duals
            new_sol_lagrangian = [np.empty_like(sol_numpy[k]) for k in range(len(blolayer.variables))]
            new_eq_dual = [np.empty_like(eq_dual[k]) for k in range(len(blolayer.eq_constraints))]

            # duals for active_eq_constraints (NON-pnorm scalar ineq only)
            new_active_dual = [np.empty((B,) + c.shape, dtype=float) for c in blolayer.active_eq_constraints]

            # duals for SOC lin constraints
            new_soc_lam = [np.zeros((B,), dtype=float) for _ in blolayer.soc_lin_constraints]

            # duals for pnorm tangent constraints
            new_pnorm_lam = [np.zeros((B,), dtype=float) for _ in blolayer.pnorm_tangent_constraints]
            old_pnorm_lam = [np.zeros((B,), dtype=float) for _ in blolayer.pnorm_tangent_constraints]

            # store pnorm masks (for phi evaluation)
            pnorm_masks = [np.zeros((B,), dtype=float) for _ in blolayer.pnorm_tangent_constraints]

            for i in range(B):
                if ctx.batch:
                    params_numpy_i = [p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)]
                else:
                    params_numpy_i = params_numpy

                for j, _ in enumerate(blolayer.param_order):
                    blolayer.param_order[j].value = params_numpy_i[j]

                # set dvar params + warm start
                for j, v in enumerate(blolayer.variables):
                    blolayer.dvar_params[j].value = dvars_numpy[j][i]
                    v.value = ctx._warm_vars[j]

                # set eq dual params
                for j in range(len(blolayer.eq_functions)):
                    blolayer.eq_dual_params[j].value = eq_dual[j][i]

                # set scalar ineq dual params + active masks
                y_dim = int(np.prod(dvars_numpy[0][i].shape)) if dvars_numpy[0][i].ndim >= 1 else 1
                num_eq = int(np.prod(eq_dual[0][i].shape)) if len(eq_dual) > 0 else 0
                cap = int(max(1, y_dim - num_eq))

                scalar_candidates = []  # (lam, j)
                for j in range(num_scalar_ineq):
                    gshape = blolayer.scalar_ineq_functions[j].shape
                    if int(np.prod(gshape)) != 1:
                        continue
                    sl_s = float(np.asarray(scalar_ineq_slack[j][i]).reshape(()))
                    lam_s = float(np.asarray(scalar_ineq_dual[j][i]).reshape(()))
                    lam_s = 0.0 if lam_s < -1e-6 else max(lam_s, 0.0)

                    if sl_s <= blolayer.slack_tol and lam_s >= blolayer.dual_cutoff:
                        scalar_candidates.append((lam_s, j))

                scalar_candidates.sort(key=lambda t: t[0])
                active_scalar = set([j for _, j in scalar_candidates[-cap:]]) if len(scalar_candidates) > cap \
                            else set([j for _, j in scalar_candidates])

                pnorm_id_set = set(blolayer.pnorm_ineq_ids)
                for j in range(num_scalar_ineq):
                    lam = np.asarray(scalar_ineq_dual[j][i], dtype=float)
                    lam = np.where(lam < -1e-8, lam, np.maximum(lam, 0.0))
                    blolayer.scalar_ineq_dual_params[j].value = lam

                    gshape = blolayer.scalar_ineq_functions[j].shape
                    if int(np.prod(gshape)) == 1:
                        mask = 1.0 if (j in active_scalar) else 0.0
                        blolayer.scalar_active_mask_params[j].value = mask
                    else:
                        # keep your original vector-ineq behavior (elementwise + cap inside that block)
                        sl = np.asarray(scalar_ineq_slack[j][i], dtype=float)
                        mask = (sl <= blolayer.slack_tol).astype(np.float64)
                        cap_vec = int(max(1, y_dim - num_eq))
                        if mask.sum() > cap_vec:
                            lam_flat = lam.reshape(-1)
                            idx = np.argpartition(lam_flat, -cap_vec)[-cap_vec:]
                            mask_flat = np.zeros_like(lam_flat, dtype=np.float64)
                            mask_flat[idx] = 1.0
                            mask = mask_flat.reshape(lam.shape)
                        blolayer.scalar_active_mask_params[j].value = mask


                # set SOC dual params (old cone duals)
                for j in range(num_soc_cones):
                    u = np.maximum(soc_dual_0[j][i], 0.0)
                    v = soc_dual_1[j][i]
                    blolayer.soc_dual_params_0[j].value = u
                    blolayer.soc_dual_params_1[j].value = v

                # set pnorm tangent params (x* and grad) and record masks
                for local_id, j_scalar in enumerate(blolayer.pnorm_ineq_ids):
                    for v_id in range(len(blolayer.variables)):
                        blolayer.pnorm_xstar_params[local_id][v_id].value = ctx.pnorm_xstar[local_id][v_id][i]
                        blolayer.pnorm_grad_params[local_id][v_id].value = ctx.pnorm_grad[local_id][v_id][i]

                    mval = float(np.asarray(blolayer.scalar_active_mask_params[j_scalar].value).reshape(()))
                    pnorm_masks[local_id][i] = mval

                # solve perturbed problem
                bargs = dict(ctx.solver_args)
                bargs["warm_start"] = bargs.get("warm_start", False)
                blolayer.perturbed_problem.solve(**bargs)

                if blolayer.perturbed_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    blolayer.perturbed_problem.solve(
                        solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, warm_start=True, verbose=False
                    )

                # collect new primal
                for j, v in enumerate(blolayer.variables):
                    new_sol_lagrangian[j][i, ...] = v.value

                # collect eq duals
                for c_id, c in enumerate(blolayer.eq_constraints):
                    new_eq_dual[c_id][i, ...] = c.dual_value

                # collect active_eq duals (NON-pnorm only)
                for c_id, c in enumerate(blolayer.active_eq_constraints):
                    new_active_dual[c_id][i, ...] = c.dual_value

                # collect SOC lin duals (lam)
                for c_id, c in enumerate(blolayer.soc_lin_constraints):
                    dv = c.dual_value
                    new_soc_lam[c_id][i] = 0.0 if dv is None else float(np.asarray(dv).reshape(()))

                # collect pnorm tangent duals (lam)
                for c_id, c in enumerate(blolayer.pnorm_tangent_constraints):
                    dv = c.dual_value
                    lam_val = 0.0 if dv is None else float(np.asarray(dv).reshape(()))

                    j_scalar = blolayer.pnorm_ineq_ids[c_id]
                    mval = float(np.asarray(blolayer.scalar_active_mask_params[j_scalar].value).reshape(()))
                    if mval < 0.5:
                        lam_val = 0.0

                    new_pnorm_lam[c_id][i] = lam_val

            # convert to torch
            new_sol = [to_torch(v, ctx.dtype, ctx.device) for v in new_sol_lagrangian]

            new_eq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_eq_dual]
            old_eq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in eq_dual]

            old_scalar_ineq_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in scalar_ineq_dual]
            new_active_dual_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_active_dual]

            old_soc_dual_0_torch = [to_torch(v, ctx.dtype, ctx.device) for v in soc_dual_0]
            old_soc_dual_1_torch = [to_torch(v, ctx.dtype, ctx.device) for v in soc_dual_1]
            old_soc_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in old_soc_lam]
            new_soc_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_soc_lam]

            # pnorm x*, grad, mask, lam
            pnorm_xstar_torch = []
            pnorm_grad_torch = []
            for local_id in range(num_pnorm):
                for v_id in range(len(blolayer.variables)):
                    pnorm_xstar_torch.append(to_torch(ctx.pnorm_xstar[local_id][v_id], ctx.dtype, ctx.device))
                    pnorm_grad_torch.append(to_torch(ctx.pnorm_grad[local_id][v_id], ctx.dtype, ctx.device))
            pnorm_mask_torch = [to_torch(v, ctx.dtype, ctx.device) for v in pnorm_masks]
            new_pnorm_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in new_pnorm_lam]
            old_pnorm_lam_torch = [to_torch(v, ctx.dtype, ctx.device) for v in old_pnorm_lam]

            # prepare torch params for autograd
            params_req = []
            req_grad_mask = []
            for p in ctx.params:
                q = p.detach().clone()
                need = bool(p.requires_grad)
                if need:
                    q.requires_grad_(True)
                params_req.append(q)
                req_grad_mask.append(need)

            # compute loss via phi_new - phi_old
            loss = 0.0
            with torch.enable_grad():
                for i in range(B):
                    vars_new_i = [v[i] for v in new_sol]
                    vars_old_i = [to_torch(sol_numpy[j][i], ctx.dtype, ctx.device) for j in range(len(blolayer.variables))]

                    params_i = slice_params_for_batch(params_req, ctx.batch_sizes, i)

                    # eq duals
                    new_eq_dual_i = [d[i] for d in new_eq_dual_torch]
                    old_eq_dual_i = [d[i] for d in old_eq_dual_torch]

                    non_pnorm_set = set(blolayer.non_pnorm_scalar_ids)
                    pnorm_set = set(blolayer.pnorm_ineq_ids)
                    pnorm_map = {j: local_id for local_id, j in enumerate(blolayer.pnorm_ineq_ids)}

                    new_scalar_dual_full_i = []
                    ptr = 0
                    for j in range(num_scalar_ineq):
                        if j in non_pnorm_set:
                            new_scalar_dual_full_i.append(new_active_dual_torch[ptr][i])
                            ptr += 1
                        elif j in pnorm_set:
                            lid = pnorm_map[j]
                            new_scalar_dual_full_i.append(new_pnorm_lam_torch[lid][i])
                        else:
                            new_scalar_dual_full_i.append(old_scalar_ineq_dual_torch[j][i])

                    old_scalar_dual_full_i = [d[i] for d in old_scalar_ineq_dual_torch]

                    # SOC cone duals and lam
                    old_soc_dual_0_i = [d[i] for d in old_soc_dual_0_torch]
                    old_soc_dual_1_i = [d[i] for d in old_soc_dual_1_torch]
                    new_soc_lam_i = [d[i] for d in new_soc_lam_torch]
                    old_soc_lam_i = [d[i] for d in old_soc_lam_torch]

                    # pnorm x*, grad, mask
                    pnorm_xstar_i = [d[i] for d in pnorm_xstar_torch]
                    pnorm_grad_i = [d[i] for d in pnorm_grad_torch]
                    pnorm_mask_i = [d[i] for d in pnorm_mask_torch]

                    # pnorm tangent lam
                    new_pnorm_lam_i = [d[i] for d in new_pnorm_lam_torch]
                    old_pnorm_lam_i = [d[i] for d in old_pnorm_lam_torch]

                    phi_new_i = blolayer.phi_torch(
                        *vars_new_i,
                        *params_i,
                        *old_eq_dual_i,
                        *old_scalar_dual_full_i,
                        *old_soc_dual_0_i,
                        *old_soc_dual_1_i,
                        *new_soc_lam_i,
                    )

                    phi_old_i = blolayer.phi_torch(
                        *vars_old_i,
                        *params_i,
                        *old_eq_dual_i,
                        *old_scalar_dual_full_i,
                        *old_soc_dual_0_i,
                        *old_soc_dual_1_i,
                        *old_soc_lam_i,
                    )

                    # dual terms are evaluated at y* (vars_old_i), exactly like your attached program
                    eq_dual_term_new_i = blolayer.eq_dual_term_torch(
                        *vars_old_i, *params_i, *new_eq_dual_i
                    )
                    eq_dual_term_old_i = blolayer.eq_dual_term_torch(
                        *vars_old_i, *params_i, *old_eq_dual_i
                    )

                    ineq_dual_term_new_i = blolayer.ineq_dual_term_torch(
                        *vars_old_i, *params_i, *new_scalar_dual_full_i
                    )
                    ineq_dual_term_old_i = blolayer.ineq_dual_term_torch(
                        *vars_old_i, *params_i, *old_scalar_dual_full_i
                    )

                    loss = loss + (
                        phi_new_i
                        + ineq_dual_term_new_i
                        + eq_dual_term_new_i
                        - phi_old_i
                        - ineq_dual_term_old_i
                        - eq_dual_term_old_i
                    )

                loss = blolayer.alpha * loss

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

            return tuple(grads)

    return _BLOLayerFnFn.apply


# Alias
FFOLayer = BLOLayer
