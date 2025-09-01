import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch import Tensor
from torch.autograd import Function

import numpy as np
import scipy
import time
import cvxpy
# import solvers
# from qpthlocal.solvers.pdipm import batch as pdipm_b
# from qpthlocal.solvers.pdipm import spbatch as pdipm_spb
# from qpthlocal.solvers.cvxpy import forward_single_np
from utils import forward_single_np_eq_cst, forward_batch_np
from enum import Enum
from utils import extract_nBatch, expandParam
from typing import cast, List, Optional, Union

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# from cvxpylayers.torch import CvxpyLayer

# class QPSolvers(Enum):
#     PDIPM_BATCHED = 1
#     CVXPY = 2

# class ffoqp(torch.nn.Module):
#     def __init__(self, eps=1e-12, verbose=0, notImprovedLim=3, maxiter=20, solver=None, lamb=100):
#         super(ffoqp, self).__init__()
#         self.eps = eps
#         self.verbose = verbose
#         self.notImprovedLim = notImprovedLim
#         self.maxiter = maxiter
#         self.solver = solver if solver is not None else QPSolvers.CVXPY
#         self.lamb = lamb

def add_diag_(M, eps):
    if eps and eps > 0:
        d = M.diagonal(dim1=-2, dim2=-1)
        d.add_(eps)

def compact_active_rows(A):  # A: (B, m, n)
    B, m, n = A.shape
    As, idx = [], []
    for b in range(B):
        rowmask = (A[b].abs().amax(dim=-1) > 0)  # non-zero rows
        Ab = A[b][rowmask]
        As.append(Ab)
        idx.append(rowmask.nonzero(as_tuple=False).squeeze(-1))
    return As, idx 

def kkt_schur_complement(Q, A, delta):
    eps_q = 1e-8
    eps_s = 1e-12
    if delta.dim() == 2:
        delta = delta.unsqueeze(-1)          # (B,n,1)

    B, n, _ = Q.shape
    m = A.shape[1] if A.numel() > 0 else 0

    I_n = torch.eye(n, dtype=Q.dtype, device=Q.device)
    L = torch.linalg.cholesky(Q + eps_q * I_n)  # (B,n,n) supports batch

    if m == 0:
        dz = -torch.cholesky_solve(delta, L)    # (B,n,1)
        return dz.squeeze(-1), Q.new_zeros(B, 0)

    AT = A.transpose(-1, -2)                   # (B,n,m)

    Winv = torch.cholesky_solve(AT, L)         # (B,n,m) = Q^{-1} A^T
    y    = torch.cholesky_solve(delta, L)      # (B,n,1) = Q^{-1} delta

    S = A @ Winv                               # (B,m,m)
    if eps_s is not None and eps_s > 0:
        I_m = torch.eye(m, dtype=Q.dtype, device=Q.device)
        S = S + eps_s * I_m

    rhs = -(A @ y)                             # (B,m,1)
    try:
        Ls = torch.linalg.cholesky(S)
        dlam = torch.cholesky_solve(rhs, Ls)   # (B,m,1)
    except RuntimeError:
        # when the row rank is not full/ill-conditioned, QR-based is faster than gelsd
        dlam = torch.linalg.lstsq(S, rhs, driver='gels').solution

    dz = -torch.cholesky_solve(delta + AT @ dlam, L)  # (B,n,1)

    return dz.squeeze(-1), dlam.squeeze(-1)

def make_schur_op(A, L, eps_s):
    AT = [a.transpose(-1, -2).contiguous() for a in A]  # ragged list
    def Aop(v_list):  # v_list: list of (m_b,1)
        outs = []
        for a, at, vb in zip(A, AT, v_list):
            # w = A Q^{-1} A^T v
            w = torch.cholesky_solve(at @ vb, L)  # (n,1)
            out = a @ w                           # (m_b,1)
            if eps_s and eps_s > 0:
                out = out + eps_s * vb
            outs.append(out)
        return outs
    return Aop

def cg_solve_list(Aop, b_list, x0_list=None, maxit=50, tol=1e-6):
    xs = []
    for i, b in enumerate(b_list):
        m = b.shape[0]
        x = torch.zeros_like(b) if (x0_list is None or x0_list[i] is None) else x0_list[i]
        r = b - Aop([x])[0]
        p = r.clone()
        rsold = (r*r).sum()
        bnrm = b.norm()
        for _ in range(maxit):
            Ap = Aop([p])[0]
            denom = (p*Ap).sum()
            alpha = rsold / (denom + 1e-40)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = (r*r).sum()
            if rsnew.sqrt() <= tol * (bnrm + 1e-40):
                break
            p = r + (rsnew/rsold) * p
            rsold = rsnew
        xs.append(x)
    return xs

def kkt_schur_fast(Q, A, delta, L_cached=None, eps_q=1e-8, eps_s=1e-10,
                   cg_threshold=256, cg_maxit=50, cg_tol=1e-6, warm_dlam_list=None):
    if delta.dim() == 3 and delta.size(-1) == 1:
        delta = delta.squeeze(-1)                   # (B,n)
    B, n, _ = Q.shape

    Q = Q.contiguous()
    A = A.contiguous()
    delta = delta.contiguous()

    if L_cached is None:
        Q_ = Q.clone()                               # do not destroy the original tensor
        add_diag_(Q_, eps_q)
        L = torch.linalg.cholesky(Q_)               # (B,n,n)
    else:
        L = L_cached


    if A.numel() == 0:
        dz = -torch.cholesky_solve(delta.unsqueeze(-1), L).squeeze(-1)
        return dz, [Q.new_zeros(0) for _ in range(B)]

    Alist, idxlist = compact_active_rows(A)          # ragged each Ab:(m_b,n)

    y = -torch.cholesky_solve(delta.unsqueeze(-1), L)  # (B,n,1) with negative sign, corresponding to Q dz = -(...)

    dlam_list, dz_list = [], []
    Aop = make_schur_op(Alist, L, eps_s)

    rhs_list = [(a @ y[b]) for b, a in enumerate(Alist)]  # (m_b,1)

    for b, Ab in enumerate(Alist):
        m_b = Ab.shape[0]
        if m_b == 0:
            dlam_b = Ab.new_zeros(0, 1)
        elif m_b <= cg_threshold:
            ATb = Ab.transpose(-1, -2).contiguous()
            Winv_b = torch.cholesky_solve(ATb, L[b:b+1])  # (1,n,m_b)
            Sb = Ab @ Winv_b.squeeze(0)                   # (m_b,m_b)
            add_diag_(Sb, eps_s)
            Ls = torch.linalg.cholesky(Sb)
            dlam_b = torch.cholesky_solve(rhs_list[b], Ls)  # (m_b,1)
        else:
            raise NotImplementedError("CG not implemented")
            # x0 = None if warm_dlam_list is None else warm_dlam_list[b]
            # dlam_b = cg_solve_list(Aop, [rhs_list[b]], [x0], maxit=cg_maxit, tol=cg_tol)[0]
        dlam_list.append(dlam_b)

    dz = y.clone()  # (B,n,1)
    for b, (Ab, dlam_b) in enumerate(zip(Alist, dlam_list)):
        if dlam_b.numel() == 0:
            continue
        ATd = Ab.transpose(-1, -2) @ dlam_b            # (n,1)
        dz[b:b+1] -= torch.cholesky_solve(ATd.unsqueeze(0), L[b:b+1])

    dz = dz.squeeze(-1)

    M = A.shape[1]  # total #constraints before compaction
    dlam = Q.new_zeros((B, M))
    for b, (dl_b, idx_b) in enumerate(zip(dlam_list, idxlist)):
        if dl_b.numel():
            # dl_b is (m_b, 1) -> (m_b,)
            dlam[b, idx_b] = dl_b.squeeze(-1)

    return dz, dlam

def ffoqp(eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, alpha=100, check_Q_spd=True, chunk_size=100,
          solver='GUROBI', solver_opts={"verbose": False},
          exact_bwd_sol=True, dual_cutoff=1e-4):
    """ -> kamo
    change lamb to alpha to prevent confusion
    """

    class QPFunctionFn(torch.autograd.Function):
        @staticmethod
        @torch.no_grad()
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            # p_ = p_ + 1/alpha * torch.randn_like(p_)
            start_time = time.time()
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)

            if check_Q_spd:
                try:
                    torch.linalg.cholesky(Q)
                except:
                    raise RuntimeError('Q is not SPD.')

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert(neq > 0 or nineq > 0)
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            # if solver == QPSolvers.PDIPM_BATCHED:
            #     ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
            #     zhats, nus, lams, slacks = pdipm_b.forward(
            #         Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
            #         eps, verbose, notImprovedLim, maxIter)
            # elif solver == QPSolvers.CVXPY:
            # vals = torch.Tensor(nBatch).type_as(Q)
            zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
            lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
            nus = torch.Tensor(nBatch, ctx.neq).type_as(Q) \
                if ctx.neq > 0 else torch.Tensor()
            slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)

            for i in range(0, nBatch, chunk_size):
                if chunk_size > 1:
                    size = min(chunk_size, nBatch - i)
                    Ai, bi = (A[i:i+size], b[i:i+size]) if neq > 0 else (None, None)
                    _, zhati, nui, lami, si = forward_batch_np(
                        *[x.cpu().numpy() if x is not None else None
                          for x in (Q[i:i+size], p[i:i+size], G[i:i+size], h[i:i+size], Ai, bi)],
                        solver=solver, solver_opts=solver_opts)
                    # zhats[i:i+size] = torch.Tensor(zhati)
                    # lams[i:i+size] = torch.Tensor(lami)
                    # slacks[i:i+size] = torch.Tensor(si)
                    # if neq > 0:
                    #     nus[i:i+size] = torch.Tensor(nui)
                    i = slice(i, i + size)
                else:
                    Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                    _, zhati, nui, lami, si = forward_single_np_eq_cst(
                        *[x.cpu().numpy() if x is not None else None
                          for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
                # if zhati[0] is None:
                #     import IPython, sys; IPython.embed(); sys.exit(-1)
                zhats[i] = torch.Tensor(zhati)
                lams[i] = torch.Tensor(lami)
                slacks[i] = torch.Tensor(si)
                if neq > 0:
                    nus[i] = torch.Tensor(nui)

            # ctx.vals = vals
            ctx.lams = lams
            ctx.nus = nus
            ctx.slacks = slacks

            # else:
            #     raise NotImplementedError("Solver not implemented")

            # ctx.vals = vals
            ctx.lams = lams
            ctx.nus = nus
            ctx.slacks = slacks

            ctx.save_for_backward(zhats, lams, nus, Q_, p_, G_, h_, A_, b_)
            # print('value', vals)
            # print('solution', zhats)
            return zhats

        @staticmethod
        def backward(ctx, grad_output):
            # Backward pass to compute gradients with respect to inputs
            zhats, lams, nus, Q_, p_, G_, h_, A_, b_ = ctx.saved_tensors
            lams = torch.clamp(lams, min=0)

            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            # Formulate a different QP to solve
            # L = f + \alpha * (g + lams * h - g^*) + \alpha^2 * |h_+|^2
            Q, Q_e = expandParam(Q_, nBatch, 3)
            p, p_e = expandParam(p_, nBatch, 2)
            G, G_e = expandParam(G_, nBatch, 3)
            h, h_e = expandParam(h_, nBatch, 2)
            A, A_e = expandParam(A_, nBatch, 3)
            b, b_e = expandParam(b_, nBatch, 2)

            Q, p, G, h, A, b = Q.to(zhats.device), p.to(zhats.device), G.to(zhats.device), h.to(zhats.device), A.to(zhats.device), b.to(zhats.device)

            # Running gradient descent for a few iterations
            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0

            delta_directions = grad_output.unsqueeze(-1)
            zhats = zhats.unsqueeze(-1).detach()

            # this is a hack by kamo
            start_time = time.time()
            active_constraints = (lams > dual_cutoff).unsqueeze(-1).float()
            G_active = G * active_constraints
            #h_active = h.unsqueeze(-1) * active_constraints
            #newp = p.unsqueeze(-1) + delta_directions / alpha

            dzhat = torch.Tensor(nBatch, nz, 1).type_as(Q)
            dnu = torch.Tensor(nBatch, nineq + neq).type_as(Q)

            if neq > 0:
                G_active = torch.cat((G_active, A), dim=1)
                #h_active = torch.cat((h_active, b.unsqueeze(-1)), dim=1)

            if exact_bwd_sol:
                # kkt_schur_fast_fn = torch.compile(kkt_schur_fast, mode="max-autotune")

                _dzhat, _dnu = kkt_schur_fast(Q, G_active, delta_directions)
                dzhat.copy_(_dzhat.unsqueeze(-1))
                dnu.copy_(_dnu)
            else:
                for i in range(0, nBatch, chunk_size):
                    if chunk_size > 1:
                        size = min(chunk_size, nBatch - i)
                        i = slice(i, i + size)
                        _, zhati, nui, _, _ = forward_batch_np(
                            *[x.cpu().numpy() if x is not None else None
                              for x in (Q[i], grad_output[i], None, None, G_active[i], torch.zeros(G_active[i].shape[0], G_active[i].shape[1]))],
                            solver=solver, solver_opts=solver_opts)
                    else:
                        _, zhati, nui, _, _ = forward_single_np_eq_cst(
                            *[x.cpu().numpy() if x is not None else None
                              for x in (Q[i], grad_output[i], None, None, G_active[i], torch.zeros(G_active[i].shape[0]))])

                    dzhat[i, :, 0] = torch.Tensor(zhati)
                    dnu[i] = torch.Tensor(nui)

            start_time = time.time()
            with torch.enable_grad():
                Q_torch = Q.detach().clone().requires_grad_(True)
                p_torch = p.detach().clone().requires_grad_(True)
                G_torch = G.detach().clone().requires_grad_(True)
                h_torch = h.detach().clone().requires_grad_(True)
                A_torch = A.detach().clone().requires_grad_(True)
                b_torch = b.detach().clone().requires_grad_(True)
               
                objectives = (dzhat.transpose(-1,-2) @ Q_torch @ zhats + p_torch.unsqueeze(1) @ dzhat).squeeze(-1,-2)
                violations = G_torch @ zhats - h_torch.unsqueeze(-1)

                ineq_penalties = dnu[:, :nineq].unsqueeze(1) @ (violations * active_constraints)

                if neq > 0:
                    eq_violations = A_torch @ zhats - b_torch.unsqueeze(-1)
                    eq_penalties = dnu[:, nineq:].unsqueeze(1) @ eq_violations
                else:
                    eq_penalties = 0

                lagrangians = objectives + ineq_penalties + eq_penalties
                loss = torch.sum(lagrangians)
                loss.backward()

                Q_grad = Q_torch.grad.detach()
                p_grad = p_torch.grad.detach()
                G_grad = G_torch.grad.detach()
                h_grad = h_torch.grad.detach()
                if neq > 0:
                    # Somehow this is not working now...
                    A_grad = A_torch.grad.detach()
                    b_grad = b_torch.grad.detach()
                    # A_grad = torch.zeros_like(A)
                    # b_grad = torch.zeros_like(b)
                else:
                    A_grad = torch.zeros_like(A)
                    b_grad = torch.zeros_like(b)

            return (Q_grad, p_grad, G_grad, h_grad, A_grad, b_grad)  # (None,) * len(ctx.saved_tensors)

    return QPFunctionFn.apply

def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()
