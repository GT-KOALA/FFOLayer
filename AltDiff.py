import torch
from torch.autograd import Function

def alt_diff_qp(P, q, A, b, G, h, rho=1.0, eps=1e-4, max_iter=1000):
    n = P.shape[0]
    m = b.shape[0]
    p = h.shape[0]
    rho = torch.tensor(rho, device=q.device, dtype=q.dtype)

    x = torch.zeros(n, device=q.device, dtype=q.dtype)
    s = torch.zeros(p, device=q.device, dtype=q.dtype)
    lam = torch.zeros(m, device=q.device, dtype=q.dtype)
    nu = torch.zeros(p, device=q.device, dtype=q.dtype)

    dx = torch.zeros((n, n), device=q.device, dtype=q.dtype)
    ds = torch.zeros((p, n), device=q.device, dtype=q.dtype)
    dlam = torch.zeros((m, n), device=q.device, dtype=q.dtype)
    dnu = torch.zeros((p, n), device=q.device, dtype=q.dtype)

    I = torch.eye(n, device=q.device, dtype=q.dtype)
    H = P + rho * (A.T @ A + G.T @ G)

    for _ in range(max_iter):
        x0 = x
        x = torch.linalg.solve(H, -(q + A.T @ lam + G.T @ nu - rho * (A.T @ b) + rho * (G.T @ (s - h))))
        dx = torch.linalg.solve(H, -(I + A.T @ dlam + G.T @ dnu + rho * (G.T @ ds)))

        s = torch.relu(-(nu / rho) - (G @ x - h))
        gate = (s > 0).to(q.dtype)
        ds = -(1.0 / rho) * gate[:, None] * (dnu + rho * (G @ dx))

        lam = lam + rho * (A @ x - b)
        dlam = dlam + rho * (A @ dx)

        nu = nu + rho * (G @ x + s - h)
        dnu = dnu + rho * (G @ dx + ds)

        if torch.linalg.norm(x - x0) <= eps * (1.0 + torch.linalg.norm(x0)):
            break

    return x, dx


def AltDiffLayer(eps=1e-4, max_iter=1000, rho=1.0):
    class L(Function):
        @staticmethod
        def forward(ctx, P, q, G=None, h=None, A=None, b=None):
            def bat(x, d):
                if x is None: return None
                return x.unsqueeze(0) if x.dim() == d else x

            P = bat(P, 2); q = bat(q, 1)
            B, n = q.shape

            A = bat(A, 2) if A is not None else None
            b = bat(b, 1) if b is not None else None
            G = bat(G, 2) if G is not None else None
            h = bat(h, 1) if h is not None else None

            if A is None or A.numel() == 0: A = torch.empty((1, 0, n), device=q.device, dtype=q.dtype)
            if b is None or b.numel() == 0: b = torch.empty((1, 0), device=q.device, dtype=q.dtype)
            if G is None or G.numel() == 0: G = torch.empty((1, 0, n), device=q.device, dtype=q.dtype)
            if h is None or h.numel() == 0: h = torch.empty((1, 0), device=q.device, dtype=q.dtype)

            if A.shape[0] == 1 and B > 1: A = A.expand(B, -1, -1)
            if b.shape[0] == 1 and B > 1: b = b.expand(B, -1)
            if G.shape[0] == 1 and B > 1: G = G.expand(B, -1, -1)
            if h.shape[0] == 1 and B > 1: h = h.expand(B, -1)

            P = P.detach(); qd = q.detach()
            A = A.detach(); b = b.detach(); G = G.detach(); h = h.detach()

            xs, Js = [], []
            for i in range(B):
                xi, Ji = alt_diff_qp(P[i], qd[i], A[i], b[i], G[i], h[i], rho=rho, eps=eps, max_iter=max_iter)
                xs.append(xi); Js.append(Ji)

            X = torch.stack(xs, 0)
            J = torch.stack(Js, 0)
            ctx.save_for_backward(J)
            ctx.unbatched = (q.dim() == 1 and P.dim() == 2)
            return X[0] if ctx.unbatched else X

        @staticmethod
        def backward(ctx, grad_output):
            (J,) = ctx.saved_tensors
            g = grad_output.unsqueeze(0) if grad_output.dim() == 1 else grad_output
            grad_q = torch.zeros_like(g)
            for i in range(g.shape[0]):
                grad_q[i] = J[i].T.mv(g[i])
            if grad_output.dim() == 1: grad_q = grad_q[0]
            return None, grad_q, None, None, None, None

    return L.apply
