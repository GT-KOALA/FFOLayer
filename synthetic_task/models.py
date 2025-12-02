import torch
import torch.nn as nn
import cvxpy as cp
from constants import *

from torch.nn.parameter import Parameter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from ffocp_eq_timing import BLOLayer
# from ffocp_eq_multithread import BLOLayer as BLOLayerMT
from ffocp_eq_multithread_ghost import BLOLayer as BLOLayerMT

from qpthlocal.qp import QPFunction
# from cvxpylayers.torch import CvxpyLayer
from cvxpylayers_local.cvxpylayer import CvxpyLayer
from cvxpylayers_local.cvxpylayer import CvxpyLayer as LPGDLayer

import ffoqp_eq_cst
import ffoqp_eq_cst_schur
import ffoqp_eq_cst_parallelize
import ffoqp_eq_cst_pdipm


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
        # x = self.activation(self.fc1(x)) ###SHING HEI: removed batch norm for batch size 1
        x = self.activation(self.batch_norm1(self.fc1(x)))
        # x = self.activation(self.fc2(x))
        x = self.activation(self.batch_norm2(self.fc2(x)))
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
    def __init__(self, input_dim, opt_dim, layer_type, constraint_learnable, device, batch_size, alpha=100, dual_cutoff=1e-3, slack_tol=1e-6, backward_eps=1e-5):
        '''
        The architecture is {parameter - optLayer}.
            
        Args:
            - delta = 1/alpha, which is the perturbation constant for finite difference
        '''
        super().__init__()
        self.layer_type = layer_type
        assert(layer_type in [FFOCP_EQ, CVXPY_LAYER, LPGD, QPTH, LPGD_QP, FFOQP_EQ, FFOQP_EQ_SCHUR, FFOQP_EQ_PARALLELIZE, FFOQP_EQ_PDIPM])
        
        self.constraint_learnable = constraint_learnable
        self.y_dim = opt_dim
        self.input_dim = input_dim
        self.num_ineq = 2*opt_dim + 1
        self.num_eq = 0
        
        
        self.predictor = MLP(input_dim, self.y_dim)
        
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
            
        if self.layer_type not in [QPTH, LPGD_QP]:
            problem, objective_fn, constraints, params, variables = setup_cvxpy_synthetic_problem(opt_dim, self.num_ineq)
    
            multithread = True
            if layer_type==FFOCP_EQ:
                if not multithread:
                    self.optlayer = BLOLayer(problem, parameters=params, variables=variables, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol, eps=1e-12, solver_name="SCS")
                else:
                    problem_list = []
                    params_list = []
                    variables_list = []
                    for i in range(batch_size):
                        problem, objective_fn, constraints, params, variables = setup_cvxpy_synthetic_problem(opt_dim, self.num_ineq)
                        problem_list.append(problem)
                        params_list.append(params)
                        variables_list.append(variables)
                    
                    self.optlayer = BLOLayerMT(problem_list, parameters_list=params_list, variables_list=variables_list, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol, eps=1e-12, backward_eps=1e-5)
                    
            elif layer_type==CVXPY_LAYER:
                self.optlayer = CvxpyLayer(problem, parameters=params, variables=variables)
            elif layer_type==LPGD:
                self.optlayer = LPGDLayer(problem, parameters=params, variables=variables, lpgd=True)
            
            elif layer_type==FFOQP_EQ:
                self.optlayer = ffoqp_eq_cst.ffoqp(alpha=alpha, chunk_size=1)
            elif layer_type == FFOQP_EQ_PDIPM:
                self.optlayer = ffoqp_eq_cst_pdipm.ffoqp(alpha=alpha)
            elif layer_type == FFOQP_EQ_SCHUR: ## use this ffoqp_cst
                self.optlayer = ffoqp_eq_cst_schur.ffoqp(alpha=alpha, chunk_size=1)
            elif layer_type == FFOQP_EQ_PARALLELIZE:
                self.optlayer = ffoqp_eq_cst_parallelize.ffoqp(alpha=alpha, chunk_size=1)
        else:
            if self.layer_type==QPTH:
                self.optlayer = QPFunction(verbose=-1)
                
                
    def forward(self, x):
        nBatch = x.size(0)
        x = x.view(nBatch, -1) #(B, input_dim)
        
        out = self.predictor(x)
        q_pred = out[..., :self.y_dim]
        
        if self.constraint_learnable:
            h = get_feasible_h(self.G, self.z0_g, torch.exp(self.log_s0))
        else:
            h = self.h
            
        if self.layer_type in [QPTH, FFOQP_EQ, FFOQP_EQ_SCHUR, FFOQP_EQ_PARALLELIZE, FFOQP_EQ_PDIPM]:
            sol = self.optlayer(
                self.Q, q_pred, self.G, h, self.A, self.b
            )
        else:
            # Expand constant params along batch dimension
            Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
            G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
            h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
            
            params_batched = [Q_batched, q_pred, G_batched, h_batched]
            
            if self.layer_type==LPGD:
                sol, = self.optlayer(*params_batched, solver_args={"eps": 1e-3}) #, solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0}
            else:
                sol, = self.optlayer(*params_batched)
                
        return sol, q_pred
        
        
        
