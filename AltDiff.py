import torch
from torch.autograd import Function

def alt_diff_qp(P, q, A, b, G, h, rho=1.0, eps=1e-4, max_iter=1000, verbose=0):
    """
    QP (paper form):
        minimize 0.5 x^T P x + q^T x
        s.t.    A x = b
                G x <= h   (implemented via Gx + s = h, s>=0)

    Differentiation w.r.t. theta=q using the paper recurrences (7a)-(7d).
    Returns:
        x  : (n,)
        J  : (n,n)  where J[i,j] = dx_i / dq_j
    """
    device = q.device
    dtype = q.dtype

    n = P.shape[0]
    m = b.shape[0]
    p = h.shape[0]

    rho_t = torch.tensor(rho, device=device, dtype=dtype)

    x = torch.zeros(n, device=device, dtype=dtype)
    s = torch.zeros(p, device=device, dtype=dtype)
    lam = torch.zeros(m, device=device, dtype=dtype)
    nu = torch.zeros(p, device=device, dtype=dtype)

    dx = torch.zeros((n, n), device=device, dtype=dtype)
    ds = torch.zeros((p, n), device=device, dtype=dtype)
    dlam = torch.zeros((m, n), device=device, dtype=dtype)
    dnu = torch.zeros((p, n), device=device, dtype=dtype)

    I = torch.eye(n, device=device, dtype=dtype)
    
    H = P + rho_t * (A.T @ A + G.T @ G)
    # H = H + 1e-9 * I

    for it in range(max_iter):
        x_prev = x

        rhs_x = -(q + A.T @ lam + G.T @ nu - rho_t * (A.T @ b) + rho_t * (G.T @ (s - h)))
        x = torch.linalg.solve(H, rhs_x)

        rhs_dx = -(I + A.T @ dlam + G.T @ dnu + rho_t * (G.T @ ds))
        dx = torch.linalg.solve(H, rhs_dx)

        pre_s = -(nu / rho_t) - (G @ x - h)
        s = torch.relu(pre_s)

        gate = (s > 0).to(dtype)                 # (p,)
        ds = -(1.0 / rho_t) * gate[:, None] * (dnu + rho_t * (G @ dx))

        lam = lam + rho_t * (A @ x - b)
        dlam = dlam + rho_t * (A @ dx)

        nu = nu + rho_t * (G @ x + s - h)
        dnu = dnu + rho_t * (G @ dx + ds)

        diff = torch.linalg.norm(x - x_prev)
        denom = 1.0 + torch.linalg.norm(x_prev)
        if diff <= eps * denom:
            if verbose:
                print(f"[alt_diff_qp] converged at it={it}, rel_change={diff/denom:.3e}")
            break

    return x, dx


def AltDiffLayer(eps=1e-4, max_iter=1000, rho=1.0, verbose=0):
    class Newlayer(Function):
        @staticmethod
        def forward(ctx, P_, q_):
            if q_.dim() == 1:
                q_in = q_.unsqueeze(0)          # (1, n)
            else:
                q_in = q_
            if P_.dim() == 2:
                P_in = P_.unsqueeze(0)          # (1, n, n)
            else:
                P_in = P_
            B, n = q_in.shape

            assert P_in.shape[0] == B and P_in.shape[1] == n and P_in.shape[2] == n, \
                f"Shape mismatch: P {P_in.shape}, q {q_in.shape}"

            device = q_in.device
            dtype = q_in.dtype

            A_ = torch.ones(1, n, device=device, dtype=dtype)
            b_ = torch.tensor([1.0], device=device, dtype=dtype)
            G_ = -torch.eye(n, device=device, dtype=dtype)
            h_ = torch.zeros(n, device=device, dtype=dtype)

            P_det = P_in.detach()
            q_det = q_in.detach()
            A = A_.detach()
            b = b_.detach()
            G = G_.detach()
            h = h_.detach()

            xs = []
            Js = []
            for i in range(B):
                x_i, J_i = alt_diff_qp(
                    P_det[i], q_det[i], A, b, G, h,
                    rho=rho, eps=eps, max_iter=max_iter, verbose=verbose
                )
                xs.append(x_i)
                Js.append(J_i)

            X = torch.stack(xs, dim=0)      # (B, n)
            J = torch.stack(Js, dim=0)      # (B, n, n)

            ctx.save_for_backward(J)

            if q_.dim() == 1 and P_.dim() == 2:
                return X[0]
            return X

        @staticmethod
        def backward(ctx, grad_output):
            (J,) = ctx.saved_tensors  # (B, n, n)

            if grad_output.dim() == 1:
                g_in = grad_output.unsqueeze(0)   # (1, n)
            else:
                g_in = grad_output               # (B, n)

            B, n = g_in.shape
            grad_q = torch.zeros_like(g_in)

            for i in range(B):
                grad_q[i] = J[i].transpose(0, 1).mv(g_in[i])

            if grad_output.dim() == 1:
                grad_q = grad_q[0]

            return None, grad_q

    return Newlayer.apply

if __name__ == "__main__":
    layer = AltDiffLayer()

    B, n = 8, 10
    P = torch.eye(n).repeat(B,1,1).requires_grad_(False)
    q = torch.randn(B, n, requires_grad=True)

    x = layer(P, q)         # (B, n)
    loss = x.sum()
    loss.backward()
    print(q.grad.shape)     # (B, n)