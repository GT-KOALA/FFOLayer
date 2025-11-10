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

def ffoqp(eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, alpha=100, check_Q_spd=True, chunk_size=100,
          solver='GUROBI', solver_opts={"verbose": False},
          exact_bwd_sol=True, slack_cutoff=1e-8):
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
            return zhats,

        @staticmethod
        def backward(ctx, grad_output):
            # Backward pass to compute gradients with respect to inputs
            zhats, lams, nus, Q_, p_, G_, h_, A_, b_ = ctx.saved_tensors
            lams = torch.clamp(lams, min=0)
            slacks = torch.clamp(ctx.slacks, min=0)

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
            # active_constraints = (lams > dual_cutoff).unsqueeze(-1).float()
            active_constraints = (slacks <= slack_cutoff).unsqueeze(-1).to(Q.dtype)
            # import pdb; pdb.set_trace()
            G_active = G * active_constraints
            #h_active = h.unsqueeze(-1) * active_constraints
            #newp = p.unsqueeze(-1) + delta_directions / alpha

            dzhat = torch.Tensor(nBatch, nz, 1).type_as(Q)
            dnu = torch.Tensor(nBatch, nineq + neq).type_as(Q)

            if neq > 0:
                G_active = torch.cat((G_active, A), dim=1)
                #h_active = torch.cat((h_active, b.unsqueeze(-1)), dim=1)

            if exact_bwd_sol:
                sqrtQ = torch.linalg.cholesky(Q)
                aapl = torch.linalg.solve_triangular(
                    sqrtQ,
                    -delta_directions,
                    upper=False)
                Aq = torch.linalg.solve_triangular(
                    sqrtQ.mT,
                    G_active,
                    upper=True,
                    left=False)
                pine = torch.linalg.lstsq(Aq, Aq @ aapl).solution
                # dlam = torch.linalg.lstsq(Aq.mT, pine, driver='gelsd').solution
                # dlam = torch.linalg.lstsq(Aq.mT, pine, driver='gels').solution
                dlam = torch.linalg.lstsq(Aq.mT, pine, driver='gelsy').solution
                dz = torch.linalg.solve_triangular(sqrtQ.mT, aapl - pine, upper=True)
                dzhat[:] = dz
                dnu[:] = dlam[..., 0]
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
               
                objectives = (dzhat.transpose(-1,-2) @ Q_torch @ zhats + p_torch.unsqueeze(1) @ dzhat).squeeze(-1).squeeze(-1)
                violations = G_torch @ zhats - h_torch.unsqueeze(-1)

                ineq_penalties = (
                    dnu[:, :nineq].unsqueeze(1) @ (violations * active_constraints)
                    + lams.unsqueeze(1) @ G_torch @ dzhat
                )

                if neq > 0:
                    eq_violations = A_torch @ zhats - b_torch.unsqueeze(-1)
                    eq_penalties = (
                        dnu[:, nineq:].unsqueeze(1) @ eq_violations
                        + nus.unsqueeze(1) @ A_torch @ dzhat
                    )
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
