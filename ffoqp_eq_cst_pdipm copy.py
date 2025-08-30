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
from enum import Enum
from utils import extract_nBatch, expandParam
from typing import cast, List, Optional, Union

from qpthlocal.solvers.pdipm import batch as pdipm_b
from qpthlocal.solvers.pdipm.batch import KKTSolvers
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

def solve_kkt_batched(Q, p, A, b, sigma=1e-8, ridge_Q=1e-12, sign=-1.0):
    B, n, _ = Q.shape
    m = A.size(1) if (A is not None and A.numel() > 0) else 0
    # if p.dim() == 2: p = p.unsqueeze(-1)
    # if b.dim() == 2: b = b.unsqueeze(-1)

    if m == 0:
        K11 = Q + ridge_Q * torch.eye(n, device=Q.device, dtype=Q.dtype)
        z = -torch.linalg.solve(K11, p).squeeze(-1)
        nu = Q.new_zeros(B, 0)
        return z, nu

    K = Q.new_zeros(B, n + m, n + m)
    K[:, :n, :n] = Q + ridge_Q * torch.eye(n, device=Q.device, dtype=Q.dtype)
    K[:, :n, n:] = A.transpose(1, 2)
    K[:, n:, :n] = A
    if sigma != 0.0:
        K[:, n:, n:] = sign * sigma * torch.eye(m, device=Q.device, dtype=Q.dtype)

    rhs = torch.cat((-p, b), dim=1)

    sol = torch.linalg.solve(K, rhs)
    z  = sol[:, :n].squeeze(-1)
    nu = sol[:, n:].squeeze(-1)
    return z, nu

def ffoqp(eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, alpha=100, check_Q_spd=False, chunk_size=None):
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

            if nineq > 0:
                ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
                zhats, nus, lams, slacks = pdipm_b.forward(
                    Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
                    eps, verbose, notImprovedLim, maxIter)
            else:
                # zhats, nus = solve_kkt_batched(Q, p, A, b)
                # lams  = Q.new_zeros(nBatch, 0)
                # slacks = Q.new_zeros(nBatch, 0)
                raise NotImplementedError("Not implemented")
            
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
            nBatch, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0

            delta_directions = grad_output.unsqueeze(-1)
            zhats = zhats.unsqueeze(-1).detach()

            # this is a hack by kamo
            start_time = time.time()
            active_constraints = (lams > 1e-3).unsqueeze(-1).float()
            G_active = G * active_constraints
            h_active = h.unsqueeze(-1) * active_constraints
            newp = p.unsqueeze(-1) + delta_directions / alpha

            if neq > 0:
                G_active = torch.cat((G_active, A), dim=1)
                h_active = torch.cat((h_active, b.unsqueeze(-1)), dim=1)

            newzhat, new_nu_both = solve_kkt_batched(Q, newp.squeeze(-1), G_active, h_active.squeeze(-1))
            # Lq, Qq = torch.linalg.eigh(Q)
            # rsqrtQ = Qq @ torch.diag_embed(torch.rsqrt(Lq)) @ Qq.transpose(-1, -2)
            # aapl = rsqrtQ @ -delta_directions
            # Aq = G_active @ rsqrtQ
            # pine = torch.linalg.lstsq(Aq, Aq @ aapl).solution
            # dlam = torch.linalg.lstsq(Aq.transpose(-1, -2), pine, driver='gelsd').solution
            # dz = rsqrtQ @ (aapl - pine)
            # newzhat = dz
            # new_nu_both = dlam[..., 0]

            newzhat = newzhat.unsqueeze(-1)
            newlam = new_nu_both[..., :nineq]
            newnu = new_nu_both[..., nineq:]

            start_time = time.time()
            with torch.enable_grad():
                Q_torch = Q.detach().clone().requires_grad_(True)
                p_torch = p.detach().clone().requires_grad_(True)
                G_torch = G.detach().clone().requires_grad_(True)
                h_torch = h.detach().clone().requires_grad_(True)
                A_torch = A.detach().clone().requires_grad_(True)
                b_torch = b.detach().clone().requires_grad_(True)
               
                objectives = (0.5 * newzhat.transpose(-1,-2) @ Q_torch @ newzhat + p_torch.unsqueeze(1) @ newzhat).squeeze(-1,-2)
                violations = G_torch @ zhats - h_torch.unsqueeze(-1)

                ineq_penalties = (newlam - lams).unsqueeze(1) @ (violations * active_constraints)

                optimal_objectives = (0.5 * zhats.transpose(-1,-2) @ Q_torch @ zhats + p_torch.unsqueeze(1) @ zhats).squeeze(-1,-2)
                if neq > 0:
                    eq_violations = A_torch @ zhats - b_torch.unsqueeze(-1)
                    eq_penalties = (newnu - nus).unsqueeze(1) @ eq_violations
                else:
                    eq_penalties = 0

                lagrangians = objectives - optimal_objectives + ineq_penalties + eq_penalties
                loss = torch.sum(lagrangians) * alpha
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
