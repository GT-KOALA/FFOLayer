from juliacall import Main as jl
import cupy
import numpy as np
import cvxpy as cp
from cupyx.scipy.sparse import csr_matrix
import time
# Load Clarabel in Julia
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
jl.seval('using CUDA, CUDA.CUSPARSE')

print("Julia project:", jl.seval("Base.active_project()"))

# jl.seval(r'''
# import Pkg

# Pkg.add(Pkg.PackageSpec(url="https://github.com/oxfordcontrol/Clarabel.jl.git", rev="CuClarabel"))
# Pkg.add("CUDA")
# Pkg.add("CUDSS")
         
# Pkg.precompile()
# ''')

# jl.seval("using CUDA; CUDA.versioninfo()")
# print("Installed packages:")
# print(jl.seval("import Pkg; sprint(Pkg.status)"))


jl.seval('''
    P = CuSparseMatrixCSR(sparse([2.0 1.0 0.0;
                1.0 2.0 0.0;
                0.0 0.0 2.0]))
    q = CuVector([0, -1., -1])
    A = CuSparseMatrixCSR(SparseMatrixCSC([1. 0 0; -1 0 0; 0 -1 0; 0 0 -1]))
    b = CuVector([1, 0., 0., 0.])

    # 0-cone dimension 1, one second-order-cone of dimension 3
    cones = Dict("f" => 1, "q"=> [3])

    settings = Clarabel.Settings(direct_solve_method = :cudss)
                                    
    solver   = Clarabel.Solver(P,q,A,b,cones, settings)
    Clarabel.solve!(solver)
    
    # Extract solution
    x = solver.solution
''')

# np.random.seed(0)
# m, n = 80, 160
# A = np.random.randn(m, n).astype(np.float64)
# b = np.random.randn(m).astype(np.float64)
# lam = 1e-2

# x = cp.Variable(n)
# obj = cp.Minimize(0.5*cp.sum_squares(A @ x - b) + 0.5*lam*cp.sum_squares(x))
# prob = cp.Problem(obj)

# try:
#     t0 = time.time()

#     val = prob.solve(
#         solver=cp.CUCLARABEL,
#         verbose=True,
#         # solver_opts={'direct_solve_method': 'cudss', 'time_limit': 60.0}
#     )
#     t1 = time.time()
#     print(f"[CuClarabel] status={prob.status}, obj={val:.6g}, wall={t1-t0:.3f}s")
#     print("solver stats:", prob.solver_stats.extra_stats)
# except Exception as e:
#     print("CuClarabel solve failed (check Julia/CuClarabel/cupy/juliacall installation):", e)
