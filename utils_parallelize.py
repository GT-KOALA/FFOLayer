import torch
import cvxpy as cp
import numpy as np
from scipy.sparse import block_diag as sp_block_diag
from scipy.linalg import block_diag

from joblib import Parallel, delayed
import os

def _to_numpy_cpu(X):
    if X is None:
        return None
    if isinstance(X, np.ndarray):
        return X
    try:
        import torch
        if torch.is_tensor(X):
            return X.cpu().contiguous().numpy()
    except Exception:
        pass
    raise TypeError(f"Unsupported type for conversion: {type(X)}")

def forward_batch_np(Q, p, G, h, A, b, solver=cp.GUROBI, solver_opts=None, verbose=False):
    """ -> kamo
    Q : (nb, nz, nz)
    p : (nb, nz)
    G : (nb, nineq, nz)
    h : (nb, nineq)
    A : (nb, neq, nz)
    b : (nb, neq)
    """
    nb = p.shape[0]
    nz = p.shape[1]
    neq = A.shape[1] if A is not None else 0
    nineq = G.shape[1] if G is not None else 0

    z_ = cp.Variable(nz * nb)
    Q_ = block_diag(*Q)
    p_ = p.reshape(-1)

    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q_) + p_.T @ z_)
    eqCon = None
    if neq > 0:
        A_ = block_diag(*A)
        b_ = b.reshape(-1)
        eqCon = A_ @ z_ == b_
    if nineq > 0:
        slacks = cp.Variable(nineq * nb)
        G_ = block_diag(*G)
        h_ = h.reshape(-1)
        ineqCon = G_ @ z_ + slacks == h_
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.GUROBI, verbose=False)
    assert 'optimal' in prob.status, f"status={prob.status}"

    zhat = np.asarray(z_.value, dtype=np.float64).reshape(nb, nz)
    nu = (None if eqCon is None else
            np.asarray(eqCon.dual_value, dtype=np.float64).reshape(nb, neq))
    if ineqCon is not None:
        lam = np.asarray(ineqCon.dual_value, dtype=np.float64).reshape(nb, nineq)
        slacks = np.asarray(slacks.value, dtype=np.float64).reshape(nb, nineq)
    else:
        lam = slacks = None

    return float(prob.value), zhat, nu, lam, slacks

def solve_in_chunks_parallel(Q, p, G, h, A, b, chunk_size=10, n_jobs=None,
                             solver=cp.GUROBI, solver_opts=None, verbose=False):
    """
    Split the batch into multiple chunks, solve each chunk in parallel, and return the concatenated results.
    """

    device = Q.device

    Q_np = _to_numpy_cpu(Q)
    p_np = _to_numpy_cpu(p)
    G_np = _to_numpy_cpu(G)
    h_np = _to_numpy_cpu(h)
    A_np = _to_numpy_cpu(A)
    b_np = _to_numpy_cpu(b)

    nb = p_np.shape[0]
    idx_slices = [slice(i, min(i + chunk_size, nb)) for i in range(0, nb, chunk_size)]

    if solver_opts is None:
        solver_opts = {}
    # solver_opts = {**{"Threads": 1, "OutputFlag": 0}, **solver_opts}

    def _get(X, sl):
        return None if X is None else X[sl]

    def _worker(sl):
        return forward_batch_np(
            _get(Q_np, sl), _get(p_np, sl), _get(G_np, sl), _get(h_np, sl), _get(A_np, sl), _get(b_np, sl),
            solver=solver, solver_opts=solver_opts, verbose=verbose
        )

    if n_jobs is None:
        try:
            n_jobs = max(1, os.cpu_count() - 1)
        except Exception:
            n_jobs = 4

    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes", batch_size=1)(
        delayed(_worker)(sl) for sl in idx_slices
    )

    vals   = [r[0] for r in results]
    zhats  = [r[1] for r in results]
    nus    = [r[2] for r in results]
    lams   = [r[3] for r in results]
    slacks = [r[4] for r in results]

    val_total = float(np.sum(vals))
    zhat_all  = np.vstack(zhats)

    zhat_all = torch.tensor(zhat_all, device=device, dtype=Q.dtype)

    def _stack_or_none(parts):
        return None if parts[0] is None else np.vstack(parts)

    nu_all = _stack_or_none(nus)
    nu_all = torch.tensor(nu_all, device=device, dtype=Q.dtype) if nu_all is not None else None
    lam_all = _stack_or_none(lams)
    lam_all = torch.tensor(lam_all, device=device, dtype=Q.dtype) if lam_all is not None else None
    slack_all = _stack_or_none(slacks)
    slack_all = torch.tensor(slack_all, device=device, dtype=Q.dtype) if slack_all is not None else None
    

    return val_total, zhat_all, nu_all, lam_all, slack_all
