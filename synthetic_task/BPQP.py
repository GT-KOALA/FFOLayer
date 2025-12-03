import torch
import numpy as np
import sparse
from osqp import OSQP
import time

torch.set_default_dtype(torch.float)
device = "cuda" if torch.cuda.is_available() else "cpu"

def osqp_interface(P, q, A, lb, ub):
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=False, eps_abs=1e-5, eps_rel=1e-5, eps_prim_inf=1e-5, eps_dual_inf=1e-5)
    t0 = time.time()
    res = prob.solve()
    return res.x, res.y, time.time() - t0

def qp_osqp_backward(x_value, y_value, P, G, A, grad_output):
    nineq, ndim = G.shape
    neq = A.shape[0]
    lambs = y_value[:nineq]  # active set
    active_set = np.concatenate([np.argwhere(lambs > 1e-4), np.argwhere(x_value <= 1e-4)])
    bG = G[active_set, :].squeeze()
    bb = np.zeros(neq)
    bh = np.zeros(len(active_set))
    bq = -grad_output.detach().cpu().numpy()
    osnewA = np.vstack([bG, A])
    osnewA = sparse.csc_matrix(osnewA)
    l_new = np.hstack([bh, bb])
    u_new = np.hstack([bh, bb])
    x_grad, y_grad, time_spent_backward = osqp_interface(P, bq, osnewA, l_new, u_new)
    return x_grad, y_grad, time_spent_backward
    
def BPQP(args, sign=-1):
    class BPQPmethod(Function):
        @staticmethod
        def forward(ctx, P, q):
            n_dim = P.shape[0]
            n_ineq = n_dim
            G = torch.diag_embed(torch.ones(n_dim)).to(device)
            h = torch.zeros(n_ineq).to(device)
            A = torch.ones(n_dim).unsqueeze(0).to(device)
            b = torch.tensor([1]).to(device)

            _P, _q, _osA, _l, _u = create_qp_instances(P, sign * q, G, h, A, b)
            x_value, y_value, _ = osqp_interface(_P, _q, _osA, _l, _u)
            ctx.P = _P
            ctx.G = G.cpu().numpy()
            ctx.A = A.cpu().numpy()
            yy = torch.cat(
                [
                    torch.from_numpy(x_value).to(device).to(torch.float32),
                    torch.from_numpy(y_value).to(device).to(torch.float32),
                ],
                dim=0,
            )

            ctx.save_for_backward(yy)
            return yy[:n_dim]

        @staticmethod
        def backward(ctx, grad_output):
            P, G, A = ctx.P, ctx.G, ctx.A
            ndim = P.shape[0]
            nineq = G.shape[0]
            yy = ctx.saved_tensors[0]
            x_star = yy[:ndim]
            lambda_star = yy[ndim: (ndim + nineq)]
            x_grad, _, _ = qp_osqp_backward(
                x_star.detach().cpu().numpy(), lambda_star.detach().cpu().numpy(), P, G, A, grad_output
            )
            try:
                x_grad = torch.from_numpy(x_grad).to(torch.float32).to(device)
            except TypeError:
                print('No solution')
                x_grad = None
            grads = (None, x_grad)
            return grads

    return BPQPmethod.apply
