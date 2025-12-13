import time
import cvxpy as cp
import numpy as np
import torch
from copy import copy
from cvxtorch import TorchExpression
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD
from utils import to_numpy, to_torch, slice_params_for_batch


def BLOLayer(problem: cp.Problem, parameters, variables, alpha=100.0, dual_cutoff=1e-3, slack_tol=1e-8, eps=1e-7, compute_cos_sim=False):
    obj = problem.objective
    if isinstance(obj, cp.Minimize):
        objective_expr = obj.expr
    elif isinstance(obj, cp.Maximize):
        objective_expr = -obj.expr
    else:
        objective_expr = getattr(obj, "expr", None)
        if objective_expr is None:
            raise ValueError("Unsupported objective type.")

    eq_funcs, scalar_ineq_funcs, cone_ineq_funcs = [], [], []
    cone_exprs_soc, cone_exprs_exp, cone_exprs_psd, cone_exprs_other = [], [], [], []
    all_cone_exprs = {"soc": cone_exprs_soc, "exp": cone_exprs_exp, "psd": cone_exprs_psd, "other": cone_exprs_other}

    for c in problem.constraints:
        if isinstance(c, cp.constraints.zero.Equality):
            eq_funcs.append(c.expr)
        elif isinstance(c, cp.constraints.nonpos.Inequality):
            scalar_ineq_funcs.append(c.expr)
        else:
            cone_ineq_funcs.append(c)
            if isinstance(c, SOC):
                cone_exprs_soc.append(c)
            elif isinstance(c, ExpCone):
                x, y, z = c.args
                g_ineq = x + cp.rel_entr(y, z)
                cone_exprs_exp.append(g_ineq)
                cone_exprs_exp.append(-y)
                cone_exprs_exp.append(-z)
            elif isinstance(c, PSD):
                X = c.args[0]
                g_ineq = -cp.lambda_min(X)
                cone_exprs_psd.append(g_ineq)
            else:
                flat_blocks = []
                for arg in c.args:
                    flat_blocks.append(arg if arg.ndim == 1 else cp.vec(arg))
                cone_exprs_other.append(cp.hstack(flat_blocks))

    if len(all_cone_exprs["exp"]) > 0 or len(all_cone_exprs["psd"]) > 0 or len(all_cone_exprs["other"]) > 0:
        raise ValueError("Only scalar inequalities + SOC supported.")

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
    def __init__(self, objective, eq_functions, scalar_ineq_functions, cone_constraints, cone_exprs, parameters, variables, alpha, dual_cutoff, slack_tol, eps, _compute_cos_sim=False):
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
        self._compute_cos_sim = bool(_compute_cos_sim)
        self.eps = float(eps)

        self.eq_constraints = [f == 0 for f in self.eq_functions]
        self.scalar_ineq_constraints = [g <= 0 for g in self.scalar_ineq_functions]
        self.problem = cp.Problem(cp.Minimize(objective), self.eq_constraints + self.scalar_ineq_constraints + self.cone_constraints)

        self.dvar_params = [cp.Parameter(shape=v.shape) for v in self.variables]
        self.eq_dual_params = [cp.Parameter(shape=f.shape) for f in self.eq_functions]
        self.scalar_ineq_dual_params = [cp.Parameter(shape=g.shape, nonneg=True) for g in self.scalar_ineq_functions]

        self.soc_constraints = list(self.cone_exprs["soc"])
        self.soc_dual_params_0 = []
        self.soc_residual_exprs = []
        for h in self.soc_constraints:
            self.soc_dual_params_0.append(cp.Parameter(shape=h.dual_variables[0].shape, nonneg=True))
            axis = getattr(h, "axis", None)
            resid = cp.pnorm(h.args[1].expr, p=2, axis=axis) - h.args[0].expr
            self.soc_residual_exprs.append(resid)

        vars_dvars_product = cp.sum([cp.sum(cp.multiply(dv, v)) for dv, v in zip(self.dvar_params, self.variables)])
        scalar_ineq_dual_product = cp.sum([cp.sum(cp.multiply(lm, g)) for lm, g in zip(self.scalar_ineq_dual_params, self.scalar_ineq_functions)])
        soc_dual_product = 0
        if len(self.soc_constraints) > 0:
            soc_dual_product = cp.sum([cp.sum(cp.multiply(u, r)) for u, r in zip(self.soc_dual_params_0, self.soc_residual_exprs)])

        self.new_objective = (1.0 / self.alpha) * vars_dvars_product + self.objective + scalar_ineq_dual_product + soc_dual_product

        self.ineq_exprs = [cp.vec(g) for g in self.scalar_ineq_functions] + [cp.vec(r) for r in self.soc_residual_exprs]
        self.ineq_torch = [
            TorchExpression(gv, provided_vars_list=[*self.variables, *self.param_order]).torch_expression
            for gv in self.ineq_exprs
        ]

        self.g0_params = [cp.Parameter(shape=gv.shape) for gv in self.ineq_exprs]
        self.mask_params = [cp.Parameter(shape=gv.shape, nonneg=True) for gv in self.ineq_exprs]

        self.Jy_params, self.y0_params = [], []
        self.Jx_params, self.x0_params = [], []
        for gv in self.ineq_exprs:
            m = int(np.prod(gv.shape))
            Jy_j, y0_j = [], []
            for v in self.variables:
                nv = int(np.prod(v.shape))
                Jy_j.append(cp.Parameter((m, nv)))
                y0_j.append(cp.Parameter((nv,)))
            self.Jy_params.append(Jy_j)
            self.y0_params.append(y0_j)

            Jx_j, x0_j = [], []
            for p in self.param_order:
                npd = int(np.prod(p.shape))
                Jx_j.append(cp.Parameter((m, npd)))
                x0_j.append(cp.Parameter((npd,)))
            self.Jx_params.append(Jx_j)
            self.x0_params.append(x0_j)

        self.active_eq_constraints = []
        for j, gv in enumerate(self.ineq_exprs):
            lin = self.g0_params[j]
            for k, v in enumerate(self.variables):
                lin = lin + self.Jy_params[j][k] @ (cp.vec(v) - self.y0_params[j][k])
            for k, p in enumerate(self.param_order):
                lin = lin + self.Jx_params[j][k] @ (cp.vec(p) - self.x0_params[j][k])
            self.active_eq_constraints.append(cp.multiply(self.mask_params[j], lin) == 0)

        self.perturbed_problem = cp.Problem(cp.Minimize(self.new_objective), self.eq_constraints + self.active_eq_constraints)

        phi_expr = self.objective
        if len(self.eq_functions) > 0:
            phi_expr = phi_expr + cp.sum([cp.sum(cp.multiply(du, f)) for du, f in zip(self.eq_dual_params, self.eq_functions)])
        if len(self.scalar_ineq_functions) > 0:
            phi_expr = phi_expr + cp.sum([cp.sum(cp.multiply(du, g)) for du, g in zip(self.scalar_ineq_dual_params, self.scalar_ineq_functions)])
        if len(self.soc_constraints) > 0:
            phi_expr = phi_expr + cp.sum([cp.sum(cp.multiply(u, r)) for u, r in zip(self.soc_dual_params_0, self.soc_residual_exprs)])

        self.phi_torch = TorchExpression(
            phi_expr,
            provided_vars_list=[
                *self.variables,
                *self.param_order,
                *self.eq_dual_params,
                *self.scalar_ineq_dual_params,
                *self.soc_dual_params_0,
            ],
        ).torch_expression

    def forward(self, *params, solver_args=None):
        if solver_args is None:
            solver_args = {}
        default_solver_args = dict(solver=cp.SCS, warm_start=False, ignore_dpp=True, max_iters=2500, eps=self.eps)
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

            batch_sizes = []
            for i, (p, q) in enumerate(zip(params, blolayer.param_order)):
                if p.dtype != ctx.dtype:
                    raise ValueError("Parameter dtypes mismatch.")
                if p.device != ctx.device:
                    raise ValueError("Parameter devices mismatch.")
                if p.ndimension() == q.ndim:
                    bs = 0
                elif p.ndimension() == q.ndim + 1:
                    bs = p.size(0)
                    if bs == 0:
                        raise ValueError("Zero batch dim.")
                else:
                    raise ValueError("Invalid parameter rank.")
                batch_sizes.append(bs)
                p_shape = p.shape if bs == 0 else p.shape[1:]
                if not np.all(np.array(p_shape) == np.array(q.shape)):
                    raise ValueError("Parameter shape mismatch.")

            ctx.batch_sizes = np.array(batch_sizes)
            ctx.batch = bool(np.any(ctx.batch_sizes > 0))
            if ctx.batch:
                nonzero = ctx.batch_sizes[ctx.batch_sizes > 0]
                B = int(nonzero[0])
                if np.any(nonzero != B):
                    raise ValueError("Inconsistent batch sizes.")
            else:
                B = 1
            ctx.batch_size = B

            params_numpy = [to_numpy(p) for p in params]

            sol_numpy = [np.empty((B,) + v.shape, dtype=float) for v in blolayer.variables]
            eq_dual = [np.empty((B,) + f.shape, dtype=float) for f in blolayer.eq_functions]
            scalar_ineq_dual = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.scalar_ineq_functions]
            scalar_ineq_slack = [np.empty((B,) + g.shape, dtype=float) for g in blolayer.scalar_ineq_functions]

            soc_dual_0 = [np.empty((B,) + h.dual_variables[0].shape, dtype=float) for h in blolayer.soc_constraints]
            soc_slack = [np.empty((B,) + r.shape, dtype=float) for r in blolayer.soc_residual_exprs]

            for i in range(B):
                if ctx.batch:
                    params_numpy_i = [p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)]
                else:
                    params_numpy_i = params_numpy

                for pval, q in zip(params_numpy_i, blolayer.param_order):
                    q.value = pval

                blolayer.problem.solve(**solver_args)
                if blolayer.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"Forward status {blolayer.problem.status}")

                for v_id, v in enumerate(blolayer.variables):
                    sol_numpy[v_id][i, ...] = v.value

                for c_id, c in enumerate(blolayer.eq_constraints):
                    eq_dual[c_id][i, ...] = c.dual_value if c.dual_value is not None else 0.0

                for j, g_expr in enumerate(blolayer.scalar_ineq_functions):
                    g_val = g_expr.value
                    scalar_ineq_dual[j][i, ...] = blolayer.scalar_ineq_constraints[j].dual_value if blolayer.scalar_ineq_constraints[j].dual_value is not None else 0.0
                    s_val = np.maximum(-g_val, 0.0)
                    scalar_ineq_slack[j][i, ...] = s_val

                for j, c in enumerate(blolayer.soc_constraints):
                    dv = c.dual_value
                    u = dv[0] if isinstance(dv, (list, tuple)) else dv
                    soc_dual_0[j][i, ...] = u
                    rv = blolayer.soc_residual_exprs[j].value
                    soc_slack[j][i, ...] = np.maximum(-rv, 0.0)

            ctx.sol_numpy = sol_numpy
            ctx.eq_dual = eq_dual
            ctx.scalar_ineq_dual = scalar_ineq_dual
            ctx.scalar_ineq_slack = scalar_ineq_slack
            ctx.soc_dual_0 = soc_dual_0
            ctx.soc_slack = soc_slack
            ctx.params_numpy = params_numpy
            ctx.params = params
            ctx.blolayer = blolayer

            ctx._warm_vars = [copy(v.value) for v in blolayer.variables]
            ctx._warm_eq_duals = [copy(c.dual_value) for c in blolayer.eq_constraints] if len(blolayer.eq_constraints) > 0 else []
            ctx._warm_ineq_duals = [copy(c.dual_value) for c in blolayer.scalar_ineq_constraints] if len(blolayer.scalar_ineq_constraints) > 0 else []

            sol_torch = [to_torch(arr, ctx.dtype, ctx.device) for arr in sol_numpy]
            return tuple(sol_torch)

        @staticmethod
        def backward(ctx, *dvars):
            bl = ctx.blolayer
            B = ctx.batch_size
            dvars_numpy = [to_numpy(dv) for dv in dvars]
            sol_numpy = ctx.sol_numpy
            eq_dual = ctx.eq_dual
            scalar_ineq_dual = ctx.scalar_ineq_dual
            soc_dual_0 = ctx.soc_dual_0
            params_numpy = ctx.params_numpy

            req_grad_mask = [bool(p.requires_grad) for p in ctx.params]
            params_req = []
            for p, need in zip(ctx.params, req_grad_mask):
                q = p.detach().clone()
                q.requires_grad_(need)
                params_req.append(q)

            new_sol = [np.empty_like(sol_numpy[k]) for k in range(len(bl.variables))]
            new_eq_dual = [np.empty((B,) + f.shape, dtype=float) for f in bl.eq_functions]
            num_scalar = len(bl.scalar_ineq_functions)
            num_soc = len(bl.soc_constraints)
            num_ineq_groups = num_scalar + num_soc
            new_ineq_dual = [np.empty((B,) + bl.ineq_exprs[j].shape, dtype=float) for j in range(num_ineq_groups)]

            y_dim = int(np.prod(bl.variables[0].shape)) if len(bl.variables) > 0 else 1
            num_eq_dim = int(sum(np.prod(f.shape) for f in bl.eq_functions)) if len(bl.eq_functions) > 0 else 0
            k_keep = int(max(1, y_dim - num_eq_dim))

            for i in range(B):
                if ctx.batch:
                    params_numpy_i = [p[i] if bs > 0 else p for p, bs in zip(params_numpy, ctx.batch_sizes)]
                else:
                    params_numpy_i = params_numpy

                for j in range(len(bl.param_order)):
                    bl.param_order[j].value = params_numpy_i[j]

                for j, v in enumerate(bl.variables):
                    bl.dvar_params[j].value = dvars_numpy[j][i]
                    v.value = ctx._warm_vars[j]

                for j, c in enumerate(bl.eq_constraints):
                    if c.dual_value is None and j < len(ctx._warm_eq_duals) and ctx._warm_eq_duals[j] is not None:
                        c.dual_value = ctx._warm_eq_duals[j]

                for j, c in enumerate(bl.scalar_ineq_constraints):
                    if c.dual_value is None and j < len(ctx._warm_ineq_duals) and ctx._warm_ineq_duals[j] is not None:
                        c.dual_value = ctx._warm_ineq_duals[j]

                for j in range(num_scalar):
                    lam = scalar_ineq_dual[j][i]
                    lam = np.maximum(lam, 0.0)
                    bl.scalar_ineq_dual_params[j].value = lam

                for j in range(num_soc):
                    u = soc_dual_0[j][i]
                    u = np.maximum(u, 0.0)
                    bl.soc_dual_params_0[j].value = u

                vars0 = [to_torch(sol_numpy[j][i], ctx.dtype, ctx.device).detach().requires_grad_(True) for j in range(len(bl.variables))]
                params0 = [to_torch(params_numpy_i[j], ctx.dtype, ctx.device).detach().requires_grad_(True) for j in range(len(bl.param_order))]

                for gid in range(num_ineq_groups):
                    gvec = bl.ineq_torch[gid](*vars0, *params0).reshape(-1)
                    g0 = gvec.detach().cpu().numpy().reshape(-1)
                    slack = np.maximum(-g0, 0.0)

                    if gid < num_scalar:
                        dual_old = np.maximum(scalar_ineq_dual[gid][i].reshape(-1), 0.0)
                    else:
                        dual_old = np.maximum(soc_dual_0[gid - num_scalar][i].reshape(-1), 0.0)

                    mask = (slack <= bl.slack_tol).astype(np.float64)
                    if mask.sum() > k_keep:
                        idx = np.argpartition(dual_old, -k_keep)[-k_keep:]
                        mf = np.zeros_like(mask, dtype=np.float64)
                        mf[idx] = 1.0
                        mask = mf

                    active_idx = np.flatnonzero(mask > 0.5)
                    m = g0.size

                    Jy_np = [np.zeros((m, int(np.prod(v.shape))), dtype=np.float64) for v in bl.variables]
                    Jx_np = [np.zeros((m, int(np.prod(p.shape))), dtype=np.float64) for p in bl.param_order]

                    for ridx in active_idx:
                        grads = torch.autograd.grad(
                            outputs=gvec[int(ridx)],
                            inputs=[*vars0, *params0],
                            retain_graph=True,
                            allow_unused=True,
                        )
                        for kk in range(len(bl.variables)):
                            if grads[kk] is not None:
                                Jy_np[kk][ridx, :] = grads[kk].detach().reshape(-1).cpu().numpy()
                        for kk in range(len(bl.param_order)):
                            gg = grads[len(bl.variables) + kk]
                            if gg is not None:
                                Jx_np[kk][ridx, :] = gg.detach().reshape(-1).cpu().numpy()

                    bl.g0_params[gid].value = g0.reshape(bl.ineq_exprs[gid].shape)
                    bl.mask_params[gid].value = mask.reshape(bl.ineq_exprs[gid].shape)

                    for kk in range(len(bl.variables)):
                        bl.y0_params[gid][kk].value = vars0[kk].detach().reshape(-1).cpu().numpy()
                        bl.Jy_params[gid][kk].value = Jy_np[kk]

                    for kk in range(len(bl.param_order)):
                        bl.x0_params[gid][kk].value = params0[kk].detach().reshape(-1).cpu().numpy()
                        bl.Jx_params[gid][kk].value = Jx_np[kk]

                back_args = dict(ctx.solver_args)
                if back_args.get("solver", None) != cp.MOSEK:
                    back_args["warm_start"] = True
                back_args["solver"] = cp.SCS
                back_args["ignore_dpp"] = True
                back_args.setdefault("max_iters", 2500)
                back_args.setdefault("eps", 1e-5)

                bl.perturbed_problem.solve(**back_args)
                st = bl.perturbed_problem.status
                if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"Backward status {st}")

                for j, v in enumerate(bl.variables):
                    new_sol[j][i, ...] = v.value

                for j, c in enumerate(bl.eq_constraints):
                    new_eq_dual[j][i, ...] = c.dual_value if c.dual_value is not None else 0.0

                for gid, c in enumerate(bl.active_eq_constraints):
                    dv = c.dual_value
                    if dv is None:
                        dv = 0.0
                    new_ineq_dual[gid][i, ...] = np.asarray(dv, dtype=float).reshape(bl.ineq_exprs[gid].shape)

            new_sol_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_sol]
            new_eq_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_eq_dual]
            old_eq_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in eq_dual]

            old_scalar_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in scalar_ineq_dual]
            old_soc_u_t = [to_torch(v, ctx.dtype, ctx.device) for v in soc_dual_0]

            new_scalar_dual_t = [to_torch(new_ineq_dual[j], ctx.dtype, ctx.device) for j in range(num_scalar)]
            new_soc_u_t = [to_torch(new_ineq_dual[num_scalar + j], ctx.dtype, ctx.device) for j in range(num_soc)]

            loss = 0.0
            with torch.enable_grad():
                for i in range(B):
                    vars_new_i = [v[i] for v in new_sol_t]
                    vars_old_i = [to_torch(sol_numpy[j][i], ctx.dtype, ctx.device) for j in range(len(bl.variables))]
                    params_i = slice_params_for_batch(params_req, ctx.batch_sizes, i)

                    new_eq_i = [d[i] for d in new_eq_dual_t]
                    old_eq_i = [d[i] for d in old_eq_dual_t]

                    new_sc_i = [d[i] for d in new_scalar_dual_t]
                    old_sc_i = [d[i] for d in old_scalar_dual_t]

                    new_soc_i = [d[i] for d in new_soc_u_t]
                    old_soc_i = [d[i] for d in old_soc_u_t]

                    phi_new = bl.phi_torch(*vars_new_i, *params_i, *new_eq_i, *new_sc_i, *new_soc_i)
                    phi_old = bl.phi_torch(*vars_old_i, *params_i, *old_eq_i, *old_sc_i, *old_soc_i)
                    loss = loss + (phi_new - phi_old)

                loss = bl.alpha * loss

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
