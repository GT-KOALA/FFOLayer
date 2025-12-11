import torch
import numpy as np

from dqp import dQP

# -----------------------------------------------------------------------------
#   minimize     (1/2) zᵀ P z + qᵀ z
#   subject to   C z ≤ d
#
#   P = 2 I₂
#   q = [0, 0]
#   C = [ [-1 -1],
#         [-1  0],
#         [ 0 -1] ]
#   d = [-1, 0, 0]
#
#   z* = [1/2, 1/2]
#   μ* = [1, 0, 0] (first constraint active)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Define differentiable problem parameters
# -----------------------------------------------------------------------------

rows_P = torch.LongTensor([0, 1])
cols_P = torch.LongTensor([0, 1])
# sparse matrix nonzero values are differentiable (0s assumed fixed)
vals_P = torch.tensor([2.0, 2.0], dtype=torch.float64, requires_grad=True) 
P = torch.sparse_coo_tensor(
    torch.stack([rows_P, cols_P]),
    vals_P,
    size=(2, 2),
    dtype=torch.float64,
)
q = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)


rows_C = torch.LongTensor([0, 0, 1, 2])
cols_C = torch.LongTensor([0, 1, 0, 1])
vals_C = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float64, requires_grad=True)
C = torch.sparse_coo_tensor(
    torch.stack([rows_C, cols_C]),
    vals_C,
    size=(3, 2),
    dtype=torch.float64,
)
d = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)

# -----------------------------------------------------------------------------
# Initialize and apply dQP
# -----------------------------------------------------------------------------
settings = dQP.build_settings(
    solve_type="sparse",
    qp_solver="osqp",
    lin_solver="qdldl",
    empty_batch=False,
)

layer = dQP.dQP_layer(settings=settings)

# Solve QP (forward pass)
z_star, lambda_star, mu_star, _, _ = layer(
    P.to_sparse_csc(), q, C.to_sparse_csc(), d
)

print("-"*80)
z_exact = np.array([0.5, 0.5])
mu_exact = np.array([1.0, 0.0, 0.0])
print("dQP active set J:", layer.active)
print("dQP z*:", z_star.detach().numpy())
print("Analytical z*:", z_exact)
print("dQP μ* :", mu_star.detach().numpy())
print("Analytical μ*:", mu_exact)
print("-"*80)

# -----------------------------------------------------------------------------
# Backpropogate (differentiate) through some scalar-valued loss. 
# A simple example is the optimal value function:
#       L(z*) = p* = (1/2) z*ᵀ P z* + qᵀ z*
#
# In this case, the Envelope theorem https://en.wikipedia.org/wiki/Envelope_theorem 
# yields a simple expression, independent of dL/dx^*, to use as reference
#       ∇_P L = (1/2) z* z*ᵀ 
#       ∇_q L = z*
#       ∇_C L = μ* z*ᵀ  
#       ∇_d L = -μ*
 # -----------------------------------------------------------------------------

# Evaluate differentiable loss and backpropogate using torch autograd
L = 0.5 * z_star @ (torch.sparse.mm(P, z_star.unsqueeze(1)).squeeze(1)) + q @ z_star
L.backward()

print("-"*80)
# gradient projected onto subspace of nonzero values
print("dQP ∇_P L:", vals_P.grad.detach().numpy()) 
grad_P_exact = 0.5 * np.outer(z_exact, z_exact)
print("Analytical ∇_P L:", grad_P_exact[rows_P, cols_P]) 

print("dQP ∇_q L:     ", q.grad.detach().numpy())
print("Analytical ∇_q L:     ", z_exact)

print("dQP ∇_C L:", vals_C.grad.detach().numpy())
grad_C_exact = np.outer(mu_exact, z_exact)
print("Analytical ∇_C L:", grad_C_exact[rows_C, cols_C])

print("dQP ∇_d L:     ", d.grad.detach().numpy())
print("Analytical ∇_d L:     ", -mu_exact)
print("-"*80)