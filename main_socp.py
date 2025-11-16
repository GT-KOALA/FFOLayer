import numpy as np
import torch
import time
import sys
import os
import argparse
import tqdm
# import qpth
from qpth.qp import QPFunction
import pickle
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from ffocp_eq import BLOLayer
from ffocp_eq_multithread import BLOLayer as BLOLayerMT
import ffoqp
import ffoqp_eq_cst
import ffoqp_eq_cst_schur
import ffoqp_eq_cst_parallelize
import ffoqp_eq_cst_pdipm

from loss import *
from models import *
from data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ffocp_eq', help='ffoqp, ffocp_eq_mt, ffocp_eq, ts, qpth, ffoqp_eq_cst_pdipm, ffoqp_eq_cst, ffoqp_eq_cst_schur')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--eps', type=float, default=0.1, help='lambda for ffoqp')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--ydim', type=int, default=200, help='dimension of y')
    
    args = parser.parse_args()

    method = args.method
    seed = args.seed
    num_epochs = args.epochs
    eps = args.eps
    lamb = 1/(eps**2)
    learning_rate = args.lr

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim   = 640
    ydim  = args.ydim
    n = ydim
    num_samples = 2048
    batch_size = args.batch_size

    train_loader, test_loader = genData(input_dim, ydim, num_samples, batch_size)

    model = MLP(input_dim, ydim)
    if method == 'ffoqp':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    loss_fn = torch.nn.MSELoss()
    writer = SummaryWriter()

    m_socp = 32

    # Setup the optimization problem
    rng = np.random.default_rng(seed)
    A_socp_np = rng.standard_normal(size=(m_socp, n))
    A_socp_np = A_socp_np / (np.linalg.norm(A_socp_np, axis=1, keepdims=True) + 1e-12)
    b_socp_np = 1.5 + 0.5 * rng.random(size=(m_socp,))

    A_socp = torch.tensor(A_socp_np, dtype=torch.float32)
    b_socp = torch.tensor(b_socp_np, dtype=torch.float32)

    # deltas = [torch.zeros_like(parameter) for parameter in model.parameters()]
    # gradients = [torch.zeros_like(parameter) for parameter in model.parameters()]
    eta = learning_rate
    D = eps**3

    # Constructing cvxpylayer instance
    z_v  = cp.Variable(n)
    q_p  = cp.Parameter(n)
    A_p  = cp.Parameter((m_socp, n))
    b_p  = cp.Parameter(m_socp)

    constraints = [A_p[i, :] @ z_v + cp.norm(z_v, 2) <= b_p[i] for i in range(m_socp)]

    objective = cp.Minimize(q_p @ z_v)
    problem = cp.Problem(objective, constraints)
    cvxpylayer = CvxpyLayer(problem, parameters=[q_p, A_p, b_p], variables=[z_v])

    # Solver options
    # Note: the current version only works for the CVXPY solver
    # solver = QPSolvers.PDIPM_BATCHED
    if 'ffoqp' in method:
        if method == 'ffoqp':
            ffoqp_layer = ffoqp.ffoqp(lamb=lamb, verbose=-1)
        elif method == 'ffoqp_eq_cst_pdipm':
            ffoqp_layer = ffoqp_eq_cst_pdipm.ffoqp(alpha=100)
        elif method == 'ffoqp_eq_cst':
            ffoqp_layer = ffoqp_eq_cst.ffoqp(alpha=100, chunk_size=100)
        elif method == 'ffoqp_eq_cst_schur':
            ffoqp_layer = ffoqp_eq_cst_schur.ffoqp(alpha=100, chunk_size=100, solver='PDIPM', solver_opts={"verbose": False})
        elif method == 'ffoqp_eq_cst_parallelize':
            ffoqp_layer = ffoqp_eq_cst_parallelize.ffoqp(alpha=100, chunk_size=100)
        else:
            raise ValueError('Invalid method: {}'.format(method))
    elif 'ffocp' in method:
        if method == 'ffocp_eq':
            ffocp_layer = BLOLayer(problem, parameters=[q_p, A_p, b_p], variables=[z_v], alpha=100, dual_cutoff=1e-3, slack_tol=1e-8, eps=1e-12)
        elif method == 'ffocp_eq_mt':
            problem_list, params_list, variables_list = [], [], []
            n_ineq_constraints = 2 * n + 1
            for i in range(args.batch_size):
                z_i = cp.Variable(n)
                Q_i = cp.Parameter((n, n), PSD=True)
                q_i = cp.Parameter(n)
                G_i = cp.Parameter((n_ineq_constraints, n))
                h_i = cp.Parameter(n_ineq_constraints)
                obj_i = 0.5 * cp.quad_form(z_i, Q_i) + q_i.T @ z_i
                cons_i = [G_i @ z_i <= h_i]

                prob_i = cp.Problem(cp.Minimize(obj_i), cons_i)

                problem_list.append(prob_i)
                params_list.append([Q_i, q_i, G_i, h_i])
                variables_list.append([z_i])
            ffocp_layer = BLOLayerMT(problem_list, parameters_list=params_list, variables_list=variables_list, alpha=100, dual_cutoff=1e-3, slack_tol=1e-8, eps=1e-12)

    qpth_layer = QPFunction(verbose=-1)
    directory = 'results_{}/{}/'.format(args.batch_size, method)
    filename = '{}_ydim{}_lr{}_eps{}_seed{}.csv'.format(method, ydim, learning_rate, eps, seed)
    if os.path.exists(directory + filename):
        os.remove(directory + filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    file = open(directory + filename, 'w')
    file.write('epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss, forward_time, backward_time\n')

    s = 0
    ts_weight = 0
    norm_weight = 0
    for epoch in range(num_epochs):
        train_ts_loss_list, test_ts_loss_list = [], []
        train_df_loss_list, test_df_loss_list = [], []
        forward_time = 0
        backward_time = 0

        model.train()
        for i, (x, y) in enumerate(train_loader):
            start_time = time.time()
            y_pred = model(x)
            ts_loss = loss_fn(y_pred, y)
            if 'ffocp' in method:
                cur_batch_size = y_pred.shape[0]
                A_batch = A_socp.unsqueeze(0).expand(cur_batch_size, -1, -1).contiguous()
                b_batch = b_socp.unsqueeze(0).expand(cur_batch_size, -1).contiguous()
                sol = ffocp_layer(y_pred, A_batch, b_batch)
                z = sol[0]
                loss = torch.mean(y * z) + ts_loss * ts_weight + torch.norm(z) * norm_weight
            elif method == 'ts':
                # z = torch.zeros(n)
                z = qpth_layer(Q, y_pred.detach(), G, h, A, b)
                loss = ts_loss
            elif method == 'qpth':
                z = qpth_layer(Q, y_pred, G, h, A, b)
                loss = torch.mean(y * z) + ts_loss * ts_weight + torch.norm(z) * norm_weight
            elif method == 'cvxpylayer':
                sol = cvxpylayer(y_pred, A_socp, b_socp)
                z = sol[0]
                loss = torch.mean(y * z) + ts_loss * ts_weight + torch.norm(z) * norm_weight

            df_loss = torch.mean(y * z) # + ts_loss
            forward_time += time.time() - start_time

            start_time = time.time()
            loss.backward()
            backward_time += time.time() - start_time

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            if epoch > 0:
                optimizer.step()

            optimizer.zero_grad()

            train_ts_loss_list.append(ts_loss.item())
            train_df_loss_list.append(df_loss.item())

        print('Forward time {}, backward time {}'.format(forward_time, backward_time))

        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                y_pred = model(x)
                sol = cvxpylayer(y_pred, A_socp, b_socp)
                z = sol[0]
                ts_loss = loss_fn(y_pred, y)
                df_loss = torch.mean(y * z) 

                test_ts_loss_list.append(ts_loss.item())
                test_df_loss_list.append(df_loss.item())

        train_ts_loss = np.mean(train_ts_loss_list)
        train_df_loss = np.mean(train_df_loss_list)
        test_ts_loss = np.mean(test_ts_loss_list)
        test_df_loss = np.mean(test_df_loss_list)
        print("Epoch {}, Train TS Loss {}, Test TS Loss {}, Train DF Loss {}, Test DF Loss {}".format(epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss))

        writer.add_scalar('Loss/TS/train', train_ts_loss, epoch)
        writer.add_scalar('Loss/TS/test', test_ts_loss, epoch)
        writer.add_scalar('Loss/DF/train', train_df_loss, epoch)
        writer.add_scalar('Loss/DF/test', test_df_loss, epoch)

        file.write('{},{},{},{},{},{},{}\n'.format(epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss, forward_time, backward_time))

    writer.flush()
    file.close()

