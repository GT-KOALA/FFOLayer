import cvxpy as cp
import torch
import inspect
from cvxtorch import TorchExpression

torch.set_default_dtype(torch.float64)

# --- CVXPY model: SOC + hinge penalty ---
n, m = 5, 3
x = cp.Variable(n, name="x")
A = cp.Parameter((m, n), name="A")
b = cp.Parameter(m, name="b")
t = cp.Parameter(nonneg=True, name="t")

soc = cp.SOC(t, A @ x + b)              # ||A x + b||_2 <= t
residual = cp.norm(soc.args[1], 2) - soc.args[0]
penalty = cp.pos(residual)

# --- Torch leaf values ---
x_t = torch.randn(n, requires_grad=True)
A_t = torch.randn(m, n, requires_grad=True)
b_t = torch.randn(m, requires_grad=True)

with torch.no_grad():
    y0 = A_t @ x_t + b_t
    r0 = torch.linalg.norm(y0)
    t_val = (r0 - 0.7).clamp_min(1e-6)
t_t = t_val.detach().clone().requires_grad_(True)

# --- Build cvxtorch expression with fixed arg order if supported ---
te = TorchExpression(penalty, provided_vars_list=[x, A, b, t])
penalty_torch = te.torch_expression
val = penalty_torch(x_t, A_t, b_t, t_t)


# --- Autograd gradients ---
val.backward()
dx_auto = x_t.grad.detach().clone()
dA_auto = A_t.grad.detach().clone()
db_auto = b_t.grad.detach().clone()
dt_auto = t_t.grad.detach().clone()

# --- Analytic gradients (choose a concrete subgradient away from kinks) ---
with torch.no_grad():
    y = A_t.detach() @ x_t.detach() + b_t.detach()
    r = torch.linalg.norm(y)
    s = r - t_t.detach()
    # indicator for hinge active
    active = (s > 0).to(y.dtype)
    # avoid r=0
    u = y / (r + 1e-12)

    dx_ana = active * (A_t.detach().T @ u)                       # (n,)
    dA_ana = active * torch.outer(u, x_t.detach())               # (m,n)
    db_ana = active * u                                          # (m,)
    dt_ana = active * (-torch.ones_like(t_t.detach()))           # scalar

def max_abs(a, b):
    return (a - b).abs().max().item()

print("penalty =", float(val.detach()))
print("max|dx_auto - dx_ana| =", max_abs(dx_auto, dx_ana))
print("max|dA_auto - dA_ana| =", max_abs(dA_auto, dA_ana))
print("max|db_auto - db_ana| =", max_abs(db_auto, db_ana))
print("max|dt_auto - dt_ana| =", max_abs(dt_auto, dt_ana))

print("dx_auto:", dx_auto, "\n dx_ana:", dx_ana)
print("dt_auto:", dt_auto, "\n dt_ana:", dt_ana)
