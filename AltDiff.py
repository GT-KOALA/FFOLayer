import torch
from torch.autograd import Function

@torch.no_grad()
def alt_qp_solve_batched(P, q, A, b, G, h, rho=1.0, eps=1e-4, max_iter=1000, reg=1e-7):
    device, dtype = q.device, q.dtype
    B, n = q.shape
    rho_t = torch.as_tensor(rho, device=device, dtype=dtype)

    m = A.shape[1]
    p = G.shape[1]

    x = torch.zeros((B, n), device=device, dtype=dtype)
    s = torch.zeros((B, p), device=device, dtype=dtype)
    lam = torch.zeros((B, m), device=device, dtype=dtype)
    nu = torch.zeros((B, p), device=device, dtype=dtype)

    I = torch.eye(n, device=device, dtype=dtype).expand(B, n, n)

    AtA = torch.zeros((B, n, n), device=device, dtype=dtype)
    if m > 0:
        AtA = A.transpose(-1, -2) @ A

    GtG = torch.zeros((B, n, n), device=device, dtype=dtype)
    if p > 0:
        GtG = G.transpose(-1, -2) @ G

    H = P + rho_t * (AtA + GtG) + reg * I
    L = torch.linalg.cholesky(H)

    def solve_H(rhs):
        if rhs.dim() == 2:
            return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        return torch.cholesky_solve(rhs, L)

    Atb = torch.zeros((B, n), device=device, dtype=dtype)
    if m > 0:
        Atb = (A.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)

    alive = torch.ones((B,), device=device, dtype=torch.bool)

    for _ in range(max_iter):
        if not alive.any():
            break

        x0 = x
        rhs = q.clone()

        if m > 0:
            rhs = rhs + (A.transpose(-1, -2) @ lam.unsqueeze(-1)).squeeze(-1) - rho_t * Atb

        if p > 0:
            rhs = rhs + (G.transpose(-1, -2) @ nu.unsqueeze(-1)).squeeze(-1) \
                      + rho_t * (G.transpose(-1, -2) @ (s - h).unsqueeze(-1)).squeeze(-1)

        x = solve_H(-rhs)

        if p > 0:
            gx = (G @ x.unsqueeze(-1)).squeeze(-1)
            s = torch.relu(-(nu / rho_t) - (gx - h))

        if m > 0:
            lam = lam + rho_t * ((A @ x.unsqueeze(-1)).squeeze(-1) - b)

        if p > 0:
            nu = nu + rho_t * ((G @ x.unsqueeze(-1)).squeeze(-1) + s - h)

        dx_norm = torch.linalg.norm(x - x0, dim=-1)
        x0_norm = torch.linalg.norm(x0, dim=-1)
        conv = dx_norm <= eps * (1.0 + x0_norm)
        alive = alive & (~conv)

    return x, nu, s


def _kkt_vjp_single(P, A, G, active_mask, g, reg=1e-9):
    device, dtype = g.device, g.dtype
    n = P.shape[0]
    m = A.shape[0]

    if G.numel() == 0 or active_mask.numel() == 0 or active_mask.sum() == 0:
        Ga = G.new_zeros((0, n))
    else:
        Ga = G[active_mask]
    ka = Ga.shape[0]

    top_left = P + reg * torch.eye(n, device=device, dtype=dtype)

    K = torch.zeros((n + m + ka, n + m + ka), device=device, dtype=dtype)
    K[:n, :n] = top_left

    if m > 0:
        K[:n, n:n+m] = A.T
        K[n:n+m, :n] = A

    if ka > 0:
        K[:n, n+m:] = Ga.T
        K[n+m:, :n] = Ga

    rhs = torch.zeros((n + m + ka,), device=device, dtype=dtype)
    rhs[:n] = g

    sol = torch.linalg.solve(K, rhs)
    u = sol[:n]
    return -u


def AltDiffLayer(eps=1e-4, max_iter=1000, rho=1.0, reg_fwd=1e-7, active_tol=1e-6, reg_bwd=1e-9):
    class L(Function):
        @staticmethod
        def forward(ctx, P, q, G=None, h=None, A=None, b=None):
            def bat(x, d):
                if x is None:
                    return None
                return x.unsqueeze(0) if x.dim() == d else x

            P = bat(P, 2)
            q = bat(q, 1)
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

            P_d = P.detach()
            q_d = q.detach()
            A_d = A.detach()
            b_d = b.detach()
            G_d = G.detach()
            h_d = h.detach()

            x, nu, s = alt_qp_solve_batched(
                P_d, q_d, A_d, b_d, G_d, h_d,
                rho=rho, eps=eps, max_iter=max_iter, reg=reg_fwd
            )

            if s.numel() == 0:
                active = torch.zeros((B, 0), device=q.device, dtype=torch.bool)
            else:
                active = (s <= active_tol)

            ctx.save_for_backward(P_d, A_d, G_d, active)
            ctx.unbatched = (q.dim() == 1 and P.dim() == 2)

            return x[0] if ctx.unbatched else x

        @staticmethod
        def backward(ctx, grad_output):
            P, A, G, active = ctx.saved_tensors

            g = grad_output.unsqueeze(0) if grad_output.dim() == 1 else grad_output
            B, _ = g.shape

            grad_q = torch.empty_like(g)
            for i in range(B):
                act_i = active[i] if active.numel() > 0 else active.new_zeros((0,), dtype=torch.bool)
                grad_q[i] = _kkt_vjp_single(P[i], A[i], G[i], act_i, g[i], reg=reg_bwd)

            if grad_output.dim() == 1:
                grad_q = grad_q[0]

            return None, grad_q, None, None, None, None

    return L.apply
