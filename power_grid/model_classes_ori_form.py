#/usr/bin/env python3

import numpy as np
import scipy.stats as st
import operator
from functools import reduce
import sys
import os

# Add parent directory to path to import ffoqp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.optim as optim

from qpth.qp import QPFunction
import qpth
from constants import *
from cvxpylayers_local.cvxpylayer import CvxpyLayer
import cvxpy as cp

import ffoqp
import ffoqp_eq_cst
import ffoqp_eq_cst_parallelize
import ffoqp_eq_cst_pdipm
import ffoqp_eq_cst_schur

import ffocp_eq


class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()

        # Initialize linear layer with least squares solution
        X_ = np.hstack([X, np.ones((X.shape[0],1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))
        
        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W,b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1,:].T)
        b.data = torch.Tensor(Theta[-1,:])
        
        # Set up non-linear network of 
        # Linear -> BatchNorm -> ReLU -> Dropout layers
        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        self.sig = Parameter(torch.ones(1, Y.shape[1], device=DEVICE))
        
    def forward(self, x):
        return self.lin(x) + self.net(x), \
            self.sig.expand(x.size(0), self.sig.size(1))
    
    def set_sig(self, X, Y):
        Y_pred = self.lin(X) + self.net(X)
        var = torch.mean((Y_pred-Y)**2, 0)
        self.sig.data = torch.sqrt(var).data.unsqueeze(0)

class SolveScheduling(nn.Module):
    def __init__(self, params, mc_samples, layer_type):
        super().__init__()
        self.n = params["n"]
        self.mc_samples = mc_samples
        self.c_ramp = params["c_ramp"]
        self.gs, self.ge = params["gamma_under"], params["gamma_over"]

        z_var = cp.Variable(self.n)
        y_param = cp.Parameter((self.mc_samples, self.n))

        z_row = cp.reshape(z_var, (1, self.n))
        
        diff = y_param - z_row

        # under = cp.pos(diff)
        # over = cp.pos(-diff)
        under = cp.Variable((self.mc_samples, self.n))       # >= 0,  >= y - z
        over = cp.Variable((self.mc_samples, self.n))       # >= 0,  >= z - y
        quad = 0.5 * cp.sum_squares(diff)
        obj = (self.gs * cp.sum(under) + self.ge * cp.sum(over) + quad) / float(self.mc_samples)

        constraints = [
            z_var[1:] - z_var[:-1] <= self.c_ramp,
            z_var[:-1] - z_var[1:] <= self.c_ramp,
            under >= 0,
            over >= 0,
            under >= y_param - z_var,
            over >= z_var - y_param,
        ]
        problem = cp.Problem(cp.Minimize(obj), constraints)

        if layer_type=="ffocp":
            # self.optlayer = BLOLayer(objective=objective, equality_functions=eq_functions, inequality_functions=ineq_functions, parameters=params, variables=variables, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol)
            self.optlayer = ffocp_eq.BLOLayer(problem, parameters=[y_param], variables=[z_var], alpha=100.0, dual_cutoff=1e-3, slack_tol=1e-8)
        elif layer_type=="cvxpylayer":
            self.optlayer = CvxpyLayer(problem, parameters=[y_param], variables=[z_var], lpgd=False)
        elif layer_type=="cvxpylayer_lpgd":
            self.optlayer = CvxpyLayer(problem, parameters=[y_param], variables=[z_var], lpgd=True)
    
    def forward(self, mu, sig):
        nBatch, n = mu.size()
        assert n == self.n

        eps = torch.randn(nBatch, self.mc_samples, self.n, device=mu.device)
        y_samples = mu[:, None, :] + sig[:, None, :] * eps

        z_star, = self.optlayer(y_samples)

        return z_star