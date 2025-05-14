import numpy as np
import diffcp
import time
from dataclasses import dataclass
from typing import Any
import cvxpy as cp

def extract_nBatch(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndim == dim:
            return param.size(0)
    return 1

def expandParam(X, nBatch, nDim):
    if X.ndim in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndim == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")

# def forward_numpy(params_numpy, context):
#     """Forward pass in numpy."""
#     
#     info = {}
#     
#     if context.gp:
#         param_map = {}
#         # construct a list of params for the DCP problem
#         for param, value in zip(context.param_order, params_numpy):
#             if param in context.old_params_to_new_params:
#                 new_id = context.old_params_to_new_params[param].id
#                 param_map[new_id] = np.log(value)
#             else:
#                 new_id = param.id
#                 param_map[new_id] = value
#         params_numpy = [param_map[pid] for pid in context.param_ids]
#     
#     # canonicalize problem
#     start = time.time()
#     As, bs, cs, cone_dicts, shapes = [], [], [], [], []
#     for i in range(context.batch_size):
#         params_numpy_i = [
#             p if sz == 0 else p[i]
#             for p, sz in zip(params_numpy, context.batch_sizes)]
#         c, _, neg_A, b = context.compiler.apply_parameters(
#             dict(zip(context.param_ids, params_numpy_i)),
#             keep_zeros=True)
#         A = -neg_A  # cvxpy canonicalizes -A
#         As.append(A)
#         bs.append(b)
#         cs.append(c)
#         cone_dicts.append(context.cone_dims)
#         shapes.append(A.shape)
#     info['canon_time'] = time.time() - start
#     info['shapes'] = shapes
# 
#     # compute solution and derivative function
#     start = time.time()
#     try:
#         if context.solve_and_derivative:
#             xs, _, _, _, DT_batch = diffcp.solve_and_derivative_batch(
#                 As, bs, cs, cone_dicts, **context.solver_args)
#             info['DT_batch'] = DT_batch
#         else:
#             xs, _, _ = diffcp.solve_only_batch(
#                 As, bs, cs, cone_dicts, **context.solver_args)
#     except diffcp.SolverError as e:
#         print(
#             "Please consider re-formulating your problem so that "
#             "it is always solvable or increasing the number of "
#             "solver iterations.")
#         raise e
#     info['solve_time'] = time.time() - start
# 
#     # extract solutions and append along batch dimension
#     start = time.time()
#     sol = [[] for i in range(len(context.variables))]
#     for i in range(context.batch_size):
#         sltn_dict = context.compiler.split_solution(
#             xs[i], active_vars=context.var_dict)
#         for j, v in enumerate(context.variables):
#             sol[j].append(np.expand_dims(sltn_dict[v.id], axis=0))
#     sol = [np.concatenate(s, axis=0) for s in sol]
# 
#     if not context.batch:
#         sol = [np.squeeze(s, axis=0) for s in sol]
# 
#     if context.gp:
#         sol = [np.exp(s) for s in sol]
#         info['sol'] = sol
#             
#     return sol, info

def forward_single_np(Q, p, G, h, A, b):
    nz, neq, nineq = p.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    z_ = cp.Variable(nz)

    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T @ z_)
    eqCon = A @ z_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)
        ineqCon = G @ z_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cp.Problem(obj, cons)
    # prob.solve(solver=cp.GUROBI, verbose=True) # max_iters=5000)
    # prob.solve()
    prob.solve(adaptive_rho = False)  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    zhat = np.array(z_.value).ravel()
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        lam = slacks = None

    return prob.value, zhat, nu, lam, slacks
