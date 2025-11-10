import numpy as np
import torch
import time
import sys
import os
import argparse

from models import *
from data import *


from torch.utils.tensorboard import SummaryWriter


       


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='bilevel layer method')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--ydim', type=int, default=200, help='dimension of y')
    
    parser.add_argument('--alpha', type=float, default=100, help='alpha')
    parser.add_argument('--dual_cutoff', type=float, default=1e-3, help='dual cutoff')
    parser.add_argument('--slack_tol', type=float, default=1e-8, help='slack tolerance')

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--learn_constraint', type=int, default=1, help='whether to learn constraint')
    parser.add_argument('--suffix', type=str, default="", help='suffix to the result directory')
    
    
    args = parser.parse_args()

    method = args.method
    seed = args.seed
    num_epochs = args.epochs
    learning_rate = args.lr
    alpha = args.alpha
    dual_cutoff = args.dual_cutoff
    slack_tol = args.slack_tol

    # Set random seed for reproducibility
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    input_dim   = 640
    ydim  = args.ydim
    n = ydim
    num_samples = 2000 #2048
    batch_size = args.batch_size

    train_loader, test_loader = genData(device, input_dim, ydim, num_samples, batch_size)
    # print(len(train_loader))
    # assert(len(train_loader)*batch_size == 1600)

    model = OptModel(input_dim, ydim, layer_type=method, constraint_learnable=(args.learn_constraint==1), batch_size=batch_size, device=device, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    loss_fn = torch.nn.MSELoss()
    
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    writer = SummaryWriter()

    deltas = [torch.zeros_like(parameter) for parameter in model.parameters()]
    gradients = [torch.zeros_like(parameter) for parameter in model.parameters()]

    # Note: the current version only works for the CVXPY solver
    # solver = QPSolvers.PDIPM_BATCHED

    directory = '../synthetic_results_{}{}/{}/'.format(args.batch_size, args.suffix, method)
    filename = '{}_ydim{}_lr{}_seed{}.csv'.format(method, ydim, learning_rate, seed)
    if os.path.exists(directory + filename):
        os.remove(directory + filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    
    
    ts_weight = 0
    norm_weight = 0
    with open(directory + filename, 'w') as file:
        file.write('epoch, train_ts_loss, test_ts_loss, train_df_loss, test_df_loss, forward_time, backward_time\n')
        file.flush()
        
        for epoch in range(num_epochs):
            print(f"##### epoch {epoch}: ")
            
            train_ts_loss_list, test_ts_loss_list = [], []
            train_df_loss_list, test_df_loss_list = [], []
            forward_time = 0
            backward_time = 0

            model.train()
            for i, (x, y) in enumerate(train_loader):
                if i%10==0:
                    print(f"\t\t train example: {i}/{len(train_loader)}")
                
                iter_start_time = time.time()
                
                start_time = time.time()

                z, y_pred = model(x) # (opt solution, predicted q)
                ts_loss = loss_fn(y_pred, y)
                
                df_loss = torch.mean(y * z)
                loss = df_loss + ts_loss * ts_weight + torch.norm(z) * norm_weight
                
                forward_time += time.time() - start_time

                start_time = time.time()
                loss.backward()
                backward_time += time.time() - start_time
                
                iter_time = time.time() - iter_start_time

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                #if epoch > 0:
                optimizer.step()

                optimizer.zero_grad()

                train_ts_loss_list.append(ts_loss.item())
                train_df_loss_list.append(df_loss.item())
                
                print(f"train loss: {loss.item()}, iter time: {iter_time}")

            print('Forward time {}, backward time {}'.format(forward_time, backward_time))

            # model.eval()
            # with torch.no_grad():
            #     for i, (x, y) in enumerate(test_loader):
            #         # print("batch size : ", x.shape[0])
            #         z, y_pred = model(x) #opt solution, predicted q
            #         ts_loss = loss_fn(y_pred, y)
            #         df_loss = torch.mean(y * z)
                    
            #         optimizer.zero_grad()

            #         test_ts_loss_list.append(ts_loss.item())
            #         test_df_loss_list.append(df_loss.item())
            model.eval()
            for i, (x, y) in enumerate(test_loader):
                # print("batch size : ", x.shape[0])
                z, y_pred = model(x) #opt solution, predicted q
                ts_loss = loss_fn(y_pred, y)
                df_loss = torch.mean(y * z)
                
                optimizer.zero_grad()

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
            file.flush()
            writer.flush()
            
        file.close()

