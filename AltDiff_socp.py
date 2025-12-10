import torch
from torch.autograd import Function

def soc_proj_and_jac(y):
    k = y.numel()
    t = y[0]
    v = y[1:]
    r = torch.linalg.norm(v)
    I = torch.eye(k, device=y.device, dtype=y.dtype)
    Z = torch.zeros((k, k), device=y.device, dtype=y.dtype)

    if r <= t:
        return y, I
    if r <= -t:
        return torch.zeros_like(y), Z

    r = torch.clamp(r, min=torch.tensor(1e-12, device=y.device, dtype=y.dtype))
    a = 0.5 * (t + r)
    b = a / r
    out = torch.cat([a.view(1), b * v])

    k1 = k - 1
    J = torch.zeros((k, k), device=y.device, dtype=y.dtype)
    J[0, 0] = 0.5
    J[0, 1:] = 0.5 * (v / r)
    J[1:, 0] = (0.5 / r) * v
    vvT = v.view(k1, 1) @ v.view(1, k1)
    J[1:, 1:] = b * torch.eye(k1, device=y.device, dtype=y.dtype) - (t / (2.0 * r**3)) * vvT
    return out, J


def alt_diff_qp(P, q, A, b, G, h, soc_a=None, soc_b=None, rho=50.0, eps=1e-8, max_iter=2500):
    if rho <= 0:
        raise ValueError("rho must be > 0")

    n = P.shape[0]
    rho = torch.tensor(float(rho), device=q.device, dtype=q.dtype)

    m = b.shape[0]
    p = h.shape[0]

    x = torch.zeros(n, device=q.device, dtype=q.dtype)
    s = torch.zeros(p, device=q.device, dtype=q.dtype)
    lam = torch.zeros(m, device=q.device, dtype=q.dtype)
    nu = torch.zeros(p, device=q.device, dtype=q.dtype)

    dx = torch.zeros((n, n), device=q.device, dtype=q.dtype)
    ds = torch.zeros((p, n), device=q.device, dtype=q.dtype)
    dlam = torch.zeros((m, n), device=q.device, dtype=q.dtype)
    dnu = torch.zeros((p, n), device=q.device, dtype=q.dtype)

    I = torch.eye(n, device=q.device, dtype=q.dtype)

    C = 0
    if soc_a is not None and soc_b is not None and soc_a.numel() and soc_b.numel():
        if soc_a.dim() == 2: soc_a = soc_a.unsqueeze(0)  # (C,k,n)
        if soc_b.dim() == 1: soc_b = soc_b.unsqueeze(0)  # (C,k)
        C = soc_a.shape[0]

    H = P + rho * (A.T @ A + G.T @ G)
    if C:
        for c in range(C):
            Sc = soc_a[c]
            H = H + rho * (Sc.T @ Sc)

    z = []
    w = []
    dz = []
    dw = []
    if C:
        for c in range(C):
            k = soc_b[c].shape[0]
            z.append(torch.zeros(k, device=q.device, dtype=q.dtype))
            w.append(torch.zeros(k, device=q.device, dtype=q.dtype))
            dz.append(torch.zeros((k, n), device=q.device, dtype=q.dtype))
            dw.append(torch.zeros((k, n), device=q.device, dtype=q.dtype))

    for _ in range(max_iter):
        z_prev = [zi.clone() for zi in z] if C else None
        s_prev = s.clone()

        soc_rhs = torch.zeros(n, device=q.device, dtype=q.dtype)
        if C:
            for c in range(C):
                Sc = soc_a[c]
                soc_rhs = soc_rhs + (Sc.T @ w[c] + rho * (Sc.T @ (soc_b[c] - z[c])))

        x = torch.linalg.solve(
            H,
            -(q + A.T @ lam + G.T @ nu + soc_rhs - rho * (A.T @ b) + rho * (G.T @ (s - h)))
        )

        soc_rhs_dx = torch.zeros((n, n), device=q.device, dtype=q.dtype)
        if C:
            for c in range(C):
                Sc = soc_a[c]
                soc_rhs_dx = soc_rhs_dx + (Sc.T @ dw[c] - rho * (Sc.T @ dz[c]))

        dx = torch.linalg.solve(
            H,
            -(I + A.T @ dlam + G.T @ dnu + rho * (G.T @ ds) + soc_rhs_dx)
        )

        if p:
            s = torch.relu(-(nu / rho) - (G @ x - h))
            gate = (s > 0).to(q.dtype)
            ds = -(1.0 / rho) * gate[:, None] * (dnu + rho * (G @ dx))

        if C:
            for c in range(C):
                Sc = soc_a[c]
                y = Sc @ x + soc_b[c] + w[c] / rho
                zc, Jc = soc_proj_and_jac(y)
                z[c] = zc

                dy = Sc @ dx + dw[c] / rho
                dz[c] = Jc @ dy

                w[c] = w[c] + rho * (Sc @ x + soc_b[c] - z[c])
                dw[c] = dw[c] + rho * (Sc @ dx - dz[c])

        if m:
            lam = lam + rho * (A @ x - b)
            dlam = dlam + rho * (A @ dx)

        if p:
            nu = nu + rho * (G @ x + s - h)
            dnu = dnu + rho * (G @ dx + ds)

        # ---- correct stopping: primal feasibility ----
        r_eq = torch.linalg.norm(A @ x - b) if m else torch.tensor(0.0, device=q.device, dtype=q.dtype)
        r_in = torch.linalg.norm(G @ x + s - h) if p else torch.tensor(0.0, device=q.device, dtype=q.dtype)
        r_soc = torch.tensor(0.0, device=q.device, dtype=q.dtype)
        if C:
            for c in range(C):
                r_soc = r_soc + torch.linalg.norm(soc_a[c] @ x + soc_b[c] - z[c])

        if (r_eq + r_in + r_soc) <= eps:
            break

        if C:
            dz_norm = sum(torch.linalg.norm(z[c] - z_prev[c]) for c in range(C))
        else:
            dz_norm = torch.tensor(0.0, device=q.device, dtype=q.dtype)
        if torch.linalg.norm(s - s_prev) + dz_norm <= eps:
            break

    return x, dx



def AltDiffLayer(eps=1e-4, max_iter=1000, rho=1.0):
    class L(Function):
        @staticmethod
        def forward(ctx, P, q, G=None, h=None, A=None, b=None, soc_a=None, soc_b=None):
            unbatched = (q.dim() == 1 and P.dim() == 2)

            def bat(x, d):
                if x is None: return None
                return x.unsqueeze(0) if x.dim() == d else x

            def bat_socA(x):
                if x is None or x.numel() == 0: return None
                if x.dim() == 2: return x.unsqueeze(0).unsqueeze(0)
                if x.dim() == 3: return x.unsqueeze(0)
                return x

            def bat_socb(x):
                if x is None or x.numel() == 0: return None
                if x.dim() == 1: return x.unsqueeze(0).unsqueeze(0)
                if x.dim() == 2: return x.unsqueeze(0)
                return x

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

            socA = bat_socA(soc_a)
            socb = bat_socb(soc_b)
            if socA is not None and socb is not None:
                if socA.shape[0] == 1 and B > 1: socA = socA.expand(B, -1, -1, -1)
                if socb.shape[0] == 1 and B > 1: socb = socb.expand(B, -1, -1)

            P = P.detach(); qd = q.detach()
            A = A.detach(); b = b.detach(); G = G.detach(); h = h.detach()
            socA = socA.detach() if socA is not None else None
            socb = socb.detach() if socb is not None else None

            xs, Js = [], []
            for i in range(B):
                xi, Ji = alt_diff_qp(
                    P[i], qd[i], A[i], b[i], G[i], h[i],
                    soc_a=(socA[i] if socA is not None else None),
                    soc_b=(socb[i] if socb is not None else None),
                    rho=rho, eps=eps, max_iter=max_iter
                )
                xs.append(xi); Js.append(Ji)

            X = torch.stack(xs, 0)
            J = torch.stack(Js, 0)
            ctx.save_for_backward(J)
            ctx.unbatched = unbatched
            return X[0] if unbatched else X

        @staticmethod
        def backward(ctx, grad_output):
            (J,) = ctx.saved_tensors
            g = grad_output.unsqueeze(0) if grad_output.dim() == 1 else grad_output
            grad_q = torch.zeros_like(g)
            for i in range(g.shape[0]):
                grad_q[i] = J[i].T.mv(g[i])
            if grad_output.dim() == 1:
                grad_q = grad_q[0]
            return None, grad_q, None, None, None, None, None, None

    return L.apply
