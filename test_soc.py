import numpy as np
import cvxpy as cp

np.set_printoptions(precision=6, suppress=True)

# ----------------------------
# Problem data (choose something that makes the SOC constraint "matter")
# ----------------------------
n = 2
m = 2

A = np.eye(m)                 # u = x + b
b = np.array([1.0, 0.0])

c = np.array([2.0, 0.0])      # t = 2*x1 + 1
d = 1.0

p = np.array([1.0, 0.0])      # objective: 0.5||x||^2 + p^T x (unconstrained minimizer is x=-p)

# ----------------------------
# Primal: minimize 0.5||x||^2 + p^T x  s.t.  ||Ax+b|| <= c^T x + d
# ----------------------------
x = cp.Variable(n)

u = A @ x + b
t = c @ x + d

soc_con = cp.SOC(t, u)
primal_prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + p @ x), [soc_con])

primal_prob.solve(solver=cp.SCS)
# ----------------------------
# Read SOC dual variable from CVXPY
# For SOC(t,u): dual_value is stacked as [y0, yv...]
# ----------------------------
dv = soc_con.dual_value

if isinstance(dv, np.ndarray) and dv.dtype != object:
    y = np.asarray(dv).reshape(-1)
    y0 = float(y[0])
    yv = np.asarray(y[1:]).reshape(-1)

# Case B: dv is (y0, yv) or [y0, yv]
else:
    # dv is typically a length-2 container: (y0, yv)
    y0_part = dv[0]
    yv_part = dv[1]

    y0 = float(np.asarray(y0_part).reshape(-1)[0])
    yv = np.asarray(yv_part).reshape(-1)                               # shape (m,)

print("Dual from CVXPY (for SOC constraint):")
print("y0 =", y0)
print("||yv|| =", np.linalg.norm(yv))
print("dual cone check: y0 - ||yv|| =", y0 - np.linalg.norm(yv))
print()

# Also make primal values 1-D for safety
x_star = np.array(x.value).reshape(-1)
t_star = float(c @ x_star + d)
u_star = (A @ x_star + b).reshape(-1)

# ----------------------------
# KKT checks
# ----------------------------
stationarity_res = x_star + p - y0 * c - A.T @ yv
compl = float(y0 * t_star + yv @ u_star)        # now valid dot product

print("KKT residuals:")
print("stationarity residual (x + p - y0*c - A^T*yv):", stationarity_res)
print("complementarity <y,(t,u)> =", compl)
print()


# ----------------------------
# KKT checks for the Lagrangian convention:
# L(x,y) = 0.5||x||^2 + p^T x - y0*(c^T x + d) - yv^T*(A x + b),
# with y = (y0,yv) in SOC (self-dual).
#
# Stationarity: grad_x L = x + p - y0*c - A^T*yv = 0
# Complementarity (SOC boundary): <y, (t,u)> = y0*t + yv^T*u = 0  (at optimum under strong duality)
# ----------------------------
stationarity_res = x_star + p - y0 * c - A.T @ yv
compl = y0 * t_star + yv @ u_star

print("KKT residuals:")
print("stationarity residual (x + p - y0*c - A^T*yv):", stationarity_res)
print("complementarity <y,(t,u)> =", compl)
print()

# If constraint is active and y0>0, y should align with (t, -u): yv ≈ -(y0/t)*u
if t_star > 1e-9 and y0 > 1e-9:
    alpha = y0 / t_star
    align_res = yv + alpha * u_star
    print("Active-boundary direction check (yv + (y0/t)*u):", align_res)
    print("Interpretation: y is ~ alpha*(t, -u) with alpha =", alpha)
print()

# ----------------------------
# Build the explicit dual problem by minimizing L over x analytically.
#
# For f(x)=0.5||x||^2 + p^T x,
# g(y) = inf_x L(x,y) = -0.5||p - y0*c - A^T*yv||^2 - d*y0 - b^T*yv
# Dual: maximize g(y) s.t. (y0,yv) in SOC
# ----------------------------
y0_var = cp.Variable()
yv_var = cp.Variable(m)

q = p - y0_var * c - A.T @ yv_var
dual_obj = cp.Maximize(-0.5 * cp.sum_squares(q) - d * y0_var - b @ yv_var)
dual_con = [cp.SOC(y0_var, yv_var)]  # y0_var >= ||yv_var||

dual_prob = cp.Problem(dual_obj, dual_con)
dual_prob.solve(solver=cp.SCS)

print("DUAL status:", dual_prob.status)
print("DUAL optimal value:", dual_prob.value)
print("gap (primal - dual):", primal_prob.value - dual_prob.value)
print("y0_dual* =", y0_var.value)
print("yv_dual* =", yv_var.value)
