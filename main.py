import numpy as np
import torch
import time
import sys
import os
import argparse
import tqdm
# import qpth
from qpth.qp import QPFunction, QPSolvers
import pickle
import cvxpylayers
import cvxpy as cp
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import ffoqp

from AdamFFO import AdamFFO

from loss import *
from models import *
from data import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ffoqp', help='ffoqp, ts, qpth')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--eps', type=float, default=0.1, help='lambda for ffoqp')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    
    args = parser.parse_args()

    method = args.method
    seed = args.seed
    num_epochs = args.epochs
    eps = args.eps
    lamb = 10
    learning_rate = args.lr

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim   = 640
    output_dim  = 32
    n = output_dim
    num_samples = 2048

    train_loader, test_loader = genData(input_dim, output_dim, num_samples)

    model = MLP(input_dim, output_dim)
    if method == 'ffoqp':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    loss_fn = torch.nn.MSELoss()
    writer = SummaryWriter()

    # Setup the optimization problem
    Q = torch.eye(n)
    # p = y_pred
    G = torch.cat([torch.eye(n), -torch.eye(n), torch.ones(1,n)], dim=0)
    h = torch.cat([torch.zeros(n), torch.ones(n), torch.Tensor([3])], dim=0)
    A = torch.Tensor()
    b = torch.Tensor()

    deltas = [torch.zeros_like(parameter) for parameter in model.parameters()]
    gradients = [torch.zeros_like(parameter) for parameter in model.parameters()]
    eta = learning_rate
    D = eps**3

    # solver = QPSolvers.PDIPM_BATCHED
    solver = QPSolvers.CVXPY

    ffoqp_layer = ffoqp.ffoqp(lamb=lamb, solver=solver, verbose=-1)
    qpth_layer = QPFunction(verbose=-1,  solver=solver)

    s = 0
    for epoch in range(num_epochs):
        train_ts_loss_list, test_ts_loss_list = [], []
        train_df_loss_list, test_df_loss_list = [], []
        train_food_loss_list, test_food_loss_list = [], []
        start_time = time.time()
        for i, (x, y) in enumerate(train_loader):
            y_pred = model(x)
            # y_pred.retain_grad()
            ts_loss = loss_fn(y_pred, y)
            # df_loss = df_loss_fn(y_pred, y)
            # food_loss = food_loss_fn(y_pred, y)
            if method == 'ffoqp':
                z = ffoqp_layer(Q, y_pred, G, h, A, b)
                loss = torch.mean(y * z) + ts_loss # + 0.01 * torch.norm(z)
                loss.backward()

                # s = torch.rand(1)
                # for i, parameter in enumerate(model.parameters()):
                #     deltas[i] -= eta * parameter.grad # Update delta
                #     deltas[i] = torch.clamp(deltas[i], min=-D, max=D) # Clip delta
                #     gradients[i] += deltas[i] * s
                #     parameter.grad = - gradients[i] / learning_rate
                #     gradients[i] = deltas[i] * (1 - s)

                # for i, parameter in enumerate(model.parameters()):
                #     s = torch.randn(parameter.shape)
                #     parameter.grad += s - deltas[i]
                #     deltas[i] = s

                # y_grad = y_pred.grad
                # print(y_grad)
                # if y_grad is not None:
                #     y_grad = torch.mean(y_grad, dim=0, keepdim=True)
                #     D = learning_rate
                #     delta = delta - learning_rate * y_grad
                #     delta = torch.clamp(delta, min=-D, max=D)
                #     y_pred.grad = - delta.repeat(y_pred.shape[0], 1) / learning_rate
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            elif method == 'ts':
                # z = torch.zeros(n)
                z = qpth_layer(Q, y_pred.detach(), G, h, A, b)
                loss = ts_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            elif method == 'qpth':
                z = qpth_layer(Q, y_pred, G, h, A, b)
                loss = torch.mean(y * z) + ts_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            df_loss = torch.mean(y * z)

            train_ts_loss_list.append(ts_loss.item())
            train_df_loss_list.append(df_loss.item())
            # train_food_loss_list.append(food_loss.item())

        # print('time elapsed:', time.time() - start_time)

        for i, (x, y) in enumerate(test_loader):
            y_pred = model(x)
            ts_loss = loss_fn(y_pred, y)
            df_loss = df_loss_fn(y_pred, y)
            food_loss = food_loss_fn(y_pred, y)

            test_ts_loss_list.append(ts_loss.item())
            test_df_loss_list.append(df_loss.item())
            test_food_loss_list.append(food_loss.item())

        train_ts_loss = np.mean(train_ts_loss_list)
        train_df_loss = np.mean(train_df_loss_list)
        train_food_loss = np.mean(train_food_loss_list)
        test_ts_loss = np.mean(test_ts_loss_list)
        test_df_loss = np.mean(test_df_loss_list)
        test_food_loss = np.mean(test_food_loss_list)
        print("Epoch {}, Train TS Loss {}, Test TS Loss {}, Train DF Loss {}, Test DF Loss {}, Train Food Loss {}, Test Food Loss {}".format(epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss, train_food_loss, test_food_loss))

        writer.add_scalar('Loss/TS/train', train_ts_loss, epoch)
        writer.add_scalar('Loss/TS/test', test_ts_loss, epoch)
        writer.add_scalar('Loss/DF/train', train_df_loss, epoch)
        writer.add_scalar('Loss/DF/test', test_df_loss, epoch)
        writer.add_scalar('Loss/Food/train', train_food_loss, epoch)
        writer.add_scalar('Loss/Food/test', test_food_loss, epoch)

    writer.flush()

