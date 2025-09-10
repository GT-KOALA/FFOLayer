import numpy as np
import cvxpy as cp
from cvxpylayers_local.cvxpylayer import CvxpyLayer
import torch
import ffoqp
import ffoqp_eq_cst

import numpy as np, torch
import cvxpy as cp

n = 2
Q = torch.eye(n)
# q = torch.tensor([1.0, 2.0])
q_param = cp.Parameter(n)
A = torch.empty(0, n); b = torch.empty(0)
G = -torch.eye(n)
# h = torch.zeros(n)
h = torch.tensor([1e3, 1e3])

z = cp.Variable(n)
obj = 0.5 * cp.quad_form(z, np.eye(n)) + q_param @ z
prob = cp.Problem(cp.Minimize(obj), [G.numpy() @ z <= h.numpy()])

g = torch.tensor([1.0, 1.0], dtype=torch.float32)
tau = 1e3

q = torch.tensor([-0.1, -0.2], dtype=torch.float32, requires_grad=True)
optimizer = torch.optim.SGD([q], lr=0.1)

layer_lpgd = CvxpyLayer(prob, parameters=[q_param], variables=[z], lpgd=True)
(z_star_lpgd,) = layer_lpgd(q)
upper_loss = torch.dot(g, z_star_lpgd) * tau
upper_loss.backward()
print("z*_lpgd =", z_star_lpgd.detach().numpy())
print("lpgd gradient =", q.grad.numpy())
optimizer.zero_grad()

layer_ffoqp = ffoqp.ffoqp()
(z_star_ffoqp,) = layer_ffoqp(Q, q, G, h, A, b)
upper_loss = torch.dot(g, z_star_ffoqp) * tau
upper_loss.backward()
print("z*_ffoqp =", z_star_ffoqp.detach().numpy())
print("ffoqp gradient =", q.grad.numpy())
optimizer.zero_grad()

layer_ffoqp_eq_cst = ffoqp_eq_cst.ffoqp()
(z_star_ffoqp_eq_cst,) = layer_ffoqp_eq_cst(Q, q, G, h, A, b)
upper_loss = torch.dot(g, z_star_ffoqp_eq_cst) * tau
upper_loss.backward()
print("z*_ffoqp_eq_cst =", z_star_ffoqp_eq_cst.detach().numpy())
print("ffoqp_eq_cst gradient =", q.grad.numpy())
optimizer.zero_grad()

layer = CvxpyLayer(prob, parameters=[q_param], variables=[z], lpgd=False)
(z_star_true,) = layer(q)
upper_loss = torch.dot(g, z_star_true) * tau
upper_loss.backward()
print("z*_true =", z_star_true.detach().numpy())
print("true gradient =", q.grad.numpy())