import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import cvxpy as cp
from constants import *
from torch.nn.parameter import Parameter

from ffocp_eq_cone_general_not_dpp_MT import BLOLayer as BLOLayerGeneralMT
import ffoqp_eq_cst_schur

from dqp import dQP
from baselines.qpthlocal.qp import QPFunction
from baselines.cvxpylayers_local.cvxpylayer import CvxpyLayer
from baselines.cvxpylayers_local.cvxpylayer import CvxpyLayer as LPGDLayer
from baselines.BPQP import BPQPLayer
from baselines.BPQP_socp import BPQPLayer_socp
from baselines.AltDiff import AltDiffLayer
from baselines.AltDiff_socp import AltDiffLayer as AltDiffLayer_socp


class MLP(nn.Module):
    def __init__(self, input_dim=64, output_dim=10):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        self.activation = nn.ReLU()
        self.bound = 10

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        batch_size = x.shape[0]
        if batch_size > 1:
            x = self.activation(self.batch_norm1(self.fc1(x)))
            x = self.activation(self.batch_norm2(self.fc2(x)))
        else:
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
        x = torch.clamp(self.fc3(x), min=-self.bound, max=self.bound)
        return x
    
    
def setup_cvxpy_synthetic_problem(n, n_ineq_constraints, unconstrained=False):
    Q_cp = cp.Parameter((n, n), PSD=True)
    q_cp = cp.Parameter(n)
    G_cp = cp.Parameter((n_ineq_constraints, n))
    h_cp = cp.Parameter(n_ineq_constraints)
    z_cp = cp.Variable(n)

    objective_fn = 0.5 * cp.sum_squares(Q_cp @ z_cp) + q_cp.T @ z_cp
    variables = [z_cp]
    if not unconstrained:
    
        constraints = [G_cp @ z_cp <= h_cp]

        problem = cp.Problem(cp.Minimize(objective_fn), constraints)
        assert problem.is_dpp()
        
        parameters = [Q_cp, q_cp, G_cp, h_cp]
    
    else:
        parameters = [Q_cp, q_cp]
        constraints = []
        problem = cp.Problem(cp.Minimize(objective_fn), constraints)

    return problem, objective_fn, constraints, parameters, variables

def setup_cvxpy_synthetic_problem_with_cones(n, n_ineq_constraints, cone_dim, unconstrained=False):
    Q_cp = cp.Parameter((n, n), PSD=True)
    q_cp = cp.Parameter(n)
    G_cp = cp.Parameter((n_ineq_constraints, n))
    h_cp = cp.Parameter(n_ineq_constraints)
    z_cp = cp.Variable(n)

    objective_fn = 0.5 * cp.sum_squares(Q_cp @ z_cp) + q_cp.T @ z_cp
    variables = [z_cp]
    constraints = []
    parameters = []

    # t_cp = cp.Parameter(nonneg=True)

    # === SOC constraints ===
    # TODO: now only one cone, add multiple cones later
    constraints.append(cp.SOC(1, z_cp))

    if not unconstrained:
        constraints.append(G_cp @ z_cp <= h_cp)

    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    assert problem.is_dpp()

    if not unconstrained:
        parameters = [Q_cp, q_cp, G_cp, h_cp]
    else:
        parameters = [G_cp, h_cp]

    return problem, objective_fn, constraints, parameters, variables

def get_feasible_h(G, z0, s0):
    '''
    get a vector h such that the inequality constraint Gy<=h can be satisfied
    
    i.e. return h = G(z0) + s0 where s0 are all positive
    
    Args:
        - G: (num_ineq, y_dim)
        - s0: (num_ineq, )
        - z0: (y_dim,)
    '''
    assert(not torch.any(s0<0))
    return torch.matmul(G, z0) + s0


class OptModel(nn.Module):
    def __init__(self, input_dim, opt_dim, layer_type, constraint_learnable, device, batch_size, alpha=100, dual_cutoff=1e-3, slack_tol=1e-6, backward_eps=1e-8, is_QP=False):
        '''
        The architecture is {parameter - optLayer}.
            
        Args:
            - delta = 1/alpha, which is the perturbation constant for finite difference
        '''
        super().__init__()
        self.layer_type = layer_type
        assert(layer_type in [FFOCP_EQ, CVXPY_LAYER, LPGD, QPTH, LPGD_QP, FFOQP_EQ, FFOQP_EQ_SCHUR, FFOQP_EQ_PARALLELIZE, FFOQP_EQ_PDIPM, BPQP, ALTDIFF, DQP])
        
        self.constraint_learnable = constraint_learnable
        self.y_dim = opt_dim
        self.input_dim = input_dim
        self.num_ineq = 2*opt_dim + 1
        self.num_eq = 0
        self.is_QP = is_QP
        
        self.predictor = MLP(input_dim, self.y_dim)

        if self.is_QP:
            ### default optimization parameters
            self.Q = torch.eye(opt_dim).to(device)#.double()
            G = torch.cat([torch.eye(opt_dim), -torch.eye(opt_dim), torch.ones(1,opt_dim)], dim=0).to(device)#.double()
            h = torch.cat([torch.zeros(opt_dim), torch.ones(opt_dim), torch.Tensor([3])], dim=0).to(device)#.double()
            
            # self.Q = torch.eye(opt_dim).to(device)#.double()
            # G = torch.cat([torch.eye(opt_dim)], dim=0).to(device)#.double()
            # h = torch.cat([torch.zeros(opt_dim)], dim=0).to(device)#.double()
            
            
            ### simple 
            # G = torch.ones(1,opt_dim).to(device)
            # G[:,1:] = 0.0
            # h = torch.Tensor([0]).to(device)
            
            ### dense
            # self.Q = torch.ones(opt_dim, opt_dim).to(device) + torch.eye(opt_dim).to(device)
            # x_star = torch.zeros(opt_dim).to(device)
            # G = torch.ones(self.num_ineq, opt_dim).to(device)   
            # eps = 1.0                    
            # h = G @ x_star + eps        
            
            
            self.A = torch.Tensor().to(device)
            self.b = torch.Tensor().to(device)
            
            ##### learnable constraints
            if constraint_learnable:
                self.G = Parameter(torch.rand((self.num_ineq, self.y_dim)))
                self.z0_g = Parameter(torch.zeros((self.y_dim,)))
                self.log_s0 = Parameter(torch.rand((self.num_ineq,)))
                
            else:
                self.G = G.to(device)
                self.h = h.to(device)
                
            if self.layer_type not in [QPTH, LPGD_QP, BPQP, ALTDIFF, DQP]:
                problem, objective_fn, constraints, params, variables = setup_cvxpy_synthetic_problem(opt_dim, self.num_ineq)
        
                if layer_type==FFOCP_EQ:
                    self.optlayer = BLOLayerGeneralMT(problem, parameters=params, variables=variables, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol, eps=1e-12, backward_eps=backward_eps)
                        
                elif layer_type==CVXPY_LAYER:
                    self.optlayer = CvxpyLayer(problem, parameters=params, variables=variables)
                elif layer_type==LPGD:
                    self.optlayer = LPGDLayer(problem, parameters=params, variables=variables, lpgd=True)
                
                elif layer_type == FFOQP_EQ_SCHUR: ## use this ffoqp_cst
                    problem, objective_fn, constraints, params, variables = setup_cvxpy_synthetic_problem(opt_dim, self.num_ineq)
                    eq_funcs, ineq_funcs = [], []
                    for c in problem.constraints:
                        # Equality: g(x,θ) == 0  -> store g(x,θ)
                        if isinstance(c, cp.constraints.zero.Equality):
                            eq_funcs.append(c.expr)

                        # Inequality: g(x,θ) <= 0 -> store g(x,θ)
                        elif isinstance(c, cp.constraints.nonpos.Inequality):
                            ineq_funcs.append(c.expr)

                        else:
                            # save for PSD or SOC constraints
                            raise NotImplementedError(
                                f"Constraint type {type(c)} not supported in BLOLayer wrapper."
                            )
                    cvxpy_instance = {"variables":variables, "params":params, "problem":problem, "eq_constraints":[], "ineq_constraints":constraints,\
                        "eq_functions":eq_funcs, "ineq_functions":ineq_funcs}
            
                    # self.optlayer = ffoqp_eq_cst_schur.ffoqp(alpha=alpha, chunk_size=1, cvxpy_instance=cvxpy_instance)
                    self.optlayer = ffoqp_eq_cst_schur.ffoqp(alpha=alpha, chunk_size=1, cvxpy_instance=cvxpy_instance, solver='qpsolvers')
                    
            else:
                if self.layer_type==QPTH:
                    self.optlayer = QPFunction(verbose=-1)
                elif self.layer_type==BPQP:
                    self.optlayer = BPQPLayer(forward_eps=1e-12, backward_eps=1e-10)
                elif self.layer_type==ALTDIFF:
                    self.optlayer = AltDiffLayer()
                elif self.layer_type==DQP:
                    dQP_settings = dQP.build_settings(
                        solve_type="dense",
                        qp_solver="gurobi",
                        # lin_solver="scipy LU",
                    )
                    self.optlayer = dQP.dQP_layer(settings=dQP_settings)
                else:
                    raise NotImplementedError("Not implemented for layer type: {}".format(layer_type))
        else:
            self.Q = torch.eye(opt_dim).to(device)#.double()
            G = torch.cat([torch.eye(opt_dim), -torch.eye(opt_dim), torch.ones(1,opt_dim)], dim=0).to(device)#.double()
            h = torch.cat([torch.zeros(opt_dim), torch.ones(opt_dim), torch.Tensor([3])], dim=0).to(device)#.double()

            self.A = torch.Tensor().to(device)
            self.b = torch.Tensor().to(device)
            self.t = torch.tensor(10.0, device=device) 
            
            ##### learnable constraints
            if constraint_learnable:
                self.G = Parameter(torch.rand((self.num_ineq, self.y_dim)))
                self.z0_g = Parameter(torch.zeros((self.y_dim,)))
                self.log_s0 = Parameter(torch.rand((self.num_ineq,)))
                
            else:
                self.G = G.to(device)
                self.h = h.to(device)
            
            cone_dim = opt_dim
            problem, objective_fn, constraints, params, variables = setup_cvxpy_synthetic_problem_with_cones(opt_dim, self.num_ineq, cone_dim)
            if layer_type==FFOCP_EQ:
                self.optlayer = BLOLayerGeneralMT(problem, parameters=params, variables=variables, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol, eps=1e-12, backward_eps=backward_eps)
            elif layer_type==CVXPY_LAYER:
                self.optlayer = CvxpyLayer(problem, parameters=params, variables=variables)
            elif layer_type==LPGD:
                self.optlayer = LPGDLayer(problem, parameters=params, variables=variables, lpgd=True)
            elif layer_type==BPQP:
                self.optlayer = BPQPLayer_socp(forward_eps=1e-12, backward_eps=1e-10)
            elif layer_type==ALTDIFF:
                self.optlayer = AltDiffLayer_socp()
            else:
                raise NotImplementedError("Not implemented for layer type: {}".format(layer_type))

    def forward(self, x):
        nBatch = x.size(0)
        x = x.view(nBatch, -1) #(B, input_dim)
        
        out = self.predictor(x)
        q_pred = out[..., :self.y_dim]
        
        if self.constraint_learnable:
            h = get_feasible_h(self.G, self.z0_g, torch.exp(self.log_s0))
        else:
            h = self.h
        
        if self.is_QP:
            if self.layer_type in [QPTH, FFOQP_EQ, FFOQP_EQ_PARALLELIZE, FFOQP_EQ_PDIPM, FFOQP_EQ_SCHUR]:
                sol = self.optlayer(
                    self.Q, q_pred, self.G, h, self.A, self.b
                )
            elif self.layer_type==BPQP or self.layer_type==ALTDIFF:
                Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
                G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
                h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
                sol = self.optlayer(Q_batched, q_pred, G_batched, h_batched, self.A, self.b)
            elif self.layer_type==DQP:
                # Q: (y_dim, y_dim)
                Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)          # (batch, y_dim, y_dim)

                # G: (num_ineq, y_dim)
                G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)          # (batch, num_ineq, y_dim)

                # h: (num_ineq,)  ->  (batch, num_ineq)
                h_batched = h.unsqueeze(0).expand(nBatch, -1)                   # (batch, num_ineq)

                # q_pred: (batch, y_dim)
                q_pred_batched = q_pred                                         # (batch, y_dim)

                if self.A.numel() == 0:
                    A_batched = None
                    b_batched = None
                else:
                    # A: (num_eq, y_dim)
                    A_batched = self.A.unsqueeze(0).expand(nBatch, -1, -1)      # (batch, num_eq, y_dim)
                    # b: (num_eq,) -> (batch, num_eq)
                    b_batched = self.b.unsqueeze(0).expand(nBatch, -1)          # (batch, num_eq)

                sol, lambda_star, mu_star, _, _ = self.optlayer(
                    Q_batched, q_pred_batched, G_batched, h_batched, A_batched, b_batched
                )
            else:
                # Expand constant params along batch dimension
                Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
                G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
                h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
                
                params_batched = [Q_batched, q_pred, G_batched, h_batched]
                
                if self.layer_type==LPGD:
                    sol, = self.optlayer(*params_batched, solver_args={"eps": 1e-3}) #, solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0}
                elif self.layer_type==CVXPY_LAYER:
                    sol, = self.optlayer(*params_batched)
                else:
                    sol, = self.optlayer(*params_batched)
        else:
            if self.layer_type in [FFOCP_EQ, CVXPY_LAYER, LPGD]:
                Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
                G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
                h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
                # t_batched = self.t.unsqueeze(0).expand(nBatch, -1)   # (batch, y_dim)
                
                params_batched = [Q_batched, q_pred, G_batched, h_batched]
                sol = self.optlayer(*params_batched)
                if isinstance(sol, tuple):
                    sol = sol[0]
            elif self.layer_type==BPQP:
                Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
                G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
                h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)

                A_batched = torch.zeros((nBatch, 0, self.y_dim), device=self.Q.device, dtype=self.Q.dtype)
                b_batched = torch.zeros((nBatch, 0),        device=self.Q.device, dtype=self.Q.dtype)

                # Single SOC: ||z|| <= 1  -> soc_a = 0, soc_b = 1
                soc_a_batched = torch.zeros((nBatch, 1, self.y_dim), device=self.Q.device, dtype=self.Q.dtype)
                soc_b_batched = torch.ones((nBatch, 1), device=self.Q.device, dtype=self.Q.dtype)

                params_batched = [Q_batched, q_pred, G_batched, h_batched, A_batched, b_batched, soc_a_batched, soc_b_batched]
                sol = self.optlayer(*params_batched)
                if isinstance(sol, tuple):
                    sol = sol[0]
            elif self.layer_type==ALTDIFF:
                Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)
                G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)
                h_batched = h.unsqueeze(0).expand(nBatch, -1)

                A_batched = torch.zeros((nBatch, 0, self.y_dim), device=self.Q.device, dtype=self.Q.dtype)
                b_batched = torch.zeros((nBatch, 0), device=self.Q.device, dtype=self.Q.dtype)
                
                params_batched = [Q_batched, q_pred, G_batched, h_batched, A_batched, b_batched]
                sol = self.optlayer(*params_batched)
                if isinstance(sol, tuple):
                    sol = sol[0]
            else:
                raise NotImplementedError("Only FFOCP_EQ is supported for non-QP problems")
                
        return sol, q_pred