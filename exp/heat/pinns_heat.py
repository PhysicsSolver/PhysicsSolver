import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS
from tqdm import tqdm
import scipy.io
import sys
import os
import argparse

from util import *
from model.pinn import PINNs
from model.pinnsformer import PINNsformer


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print("cuda is available")
else:
    device = torch.device('cpu')
    print("cpu is available")

def get_data(x_range, y_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)
    x_mesh, t_mesh = np.meshgrid(x, t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    t = t_num_train
    data = data[0:t, :, :]
    b_left = data[0, :, :]  
    b_right = data[-1, :, :]
    b_upper = data[:, -1, :]  
    b_lower = data[:, 0, :]  
    res = data.reshape(-1, 2)

    return res, b_left, b_right, b_upper, b_lower


def get_data_2(x_range, y_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)
    x_mesh, t_mesh = np.meshgrid(x, t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    b_left = data[0, :, :] 
    b_right = data[-1, :, :]
    b_upper = data[:, -1, :]  
    b_lower = data[:, 0, :] 
    t = t_num_pred
    data = data[0:t, :, :]
    res = data.reshape(-1, 2)
    return res, b_left, b_right, b_upper, b_lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameters')

    parser.add_argument('--x_num_pred', default=101, type=int, help='x_num_pred')
    parser.add_argument('--t_num_pred', default=5, type=int, help='t_num_pred')
    parser.add_argument('--x_num_train', default=101, type=int, help='x_num_train')
    parser.add_argument('--t_num_train', default=4, type=int, help='t_num_train')
    pars = parser.parse_args()

    x_num_pred = pars.x_num_pred
    t_num_pred = pars.t_num_pred
    x_num_train = pars.x_num_train
    t_num_train = pars.t_num_train

    res, b_left, b_right, b_upper, b_lower = get_data([0, 1], [0, 0.2], x_num_train, t_num_train)

    res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
    b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
    b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
    b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
    b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

    x_res, t_res = res[:, 0:1], res[:, 1:2]
    x_left, t_left = b_left[:, 0:1], b_left[:, 1:2]
    x_right, t_right = b_right[:,  0:1], b_right[:,  1:2]
    x_upper, t_upper = b_upper[:,  0:1], b_upper[:,  1:2]
    x_lower, t_lower = b_lower[:,  0:1], b_lower[:,  1:2]

    res_test, _, _, _, _ = get_data_2([0, 1], [0, 0.2], x_num_pred, t_num_pred)
    res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
    x_test, t_test = res_test[:, 0:1], res_test[:, 1:2]


    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model = PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)
    optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

    loss_track = []

    nIter = 500

    for i in tqdm(range(nIter)):
        def closure():
            pred_res = model(x_res, t_res)
            pred_left = model(x_left, t_left)
            pred_right = model(x_right, t_right)
            pred_upper = model(x_upper, t_upper)
            pred_lower = model(x_lower, t_lower)

            u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]

            u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]

            u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                      create_graph=True)[0]

            loss_res = torch.mean((u_t -  u_xx) ** 2)
            loss_bc = torch.mean((pred_upper - 0) ** 2) + torch.mean((0 - pred_lower) ** 2)
            loss_ic = torch.mean((pred_left[:, 0] - torch.sin(torch.tensor(np.pi)*x_left[:, 0])) ** 2)
            loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

            loss = loss_res + loss_bc + loss_ic

            if i % 10 == 0:
                print("Iter: %d, Total loss: %.2e, Res loss: %.2e, BC loss: %.2e, IC loss: %.2e" % (
                i, loss.item(), loss_res.item(), loss_bc.item(), loss_ic.item()))

            optim.zero_grad()
            loss.backward()
            return loss

        optim.step(closure)

    with torch.no_grad():
        pred = model(x_test, t_test)[:, 0:1]
        pred = pred.cpu().detach().numpy()



