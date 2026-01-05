# code from https://github.com/HxSun08/Alt-Diff/blob/main/classification/newlayer.py

import torch
import math
import time

def relu(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = 0
    return ss

def sgn(s):
    ss = torch.zeros(len(s))
    for i in range(len(s)):
        if s[i]<=0:
            ss[i] = 0
        else:
            ss[i] = 1
    return ss

def proj(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = (ss[i] + math.sqrt(ss[i] ** 2 + 4 * 0.001)) / 2
    return ss

def alt_diff(Pi, qi, Ai, bi, Gi, hi, device="cuda"):
    
    n, m, d = qi.shape[0], bi.shape[0], hi.shape[0]
    xk = torch.zeros(n).to(device).to(torch.float64)
    sk = torch.zeros(d).to(device).to(torch.float64)
    lamb = torch.zeros(m).to(device).to(torch.float64)
    nu = torch.zeros(d).to(device).to(torch.float64)
    
    
    dxk = torch.zeros((n, n)).to(device).to(torch.float64)
    dsk = torch.zeros((d, n)).to(device).to(torch.float64)
    dlamb = torch.zeros((m, n)).to(device).to(torch.float64)
    dnu = torch.zeros((d, n)).to(device).to(torch.float64)
    
    rho = 1
    thres = 1e-3
    R = - torch.linalg.inv(Pi + rho * Ai.T @ Ai + rho * Gi.T @ Gi)
    
    res = [1000, -100]
    
    ATb = rho * Ai.T @ bi.double()
    GTh = rho * Gi.T @ hi
    begin2 = time.time()

    while abs((res[-1]-res[-2])/res[-2]) > thres:
        iter_time_start = time.time()
        #print((Ai.T @ lamb).shape)
        xk = R @ (qi + Ai.T @ lamb + Gi.T @ nu - ATb + rho * Gi.T @ sk - GTh)
        
        dxk = R @ (torch.eye(n).to(device) + Ai.T @ dlamb + Gi.T @ dnu + rho * Gi.T @ dsk)
        
        sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))
        dsk = (-1 / rho) * sgn(sk).to(device).reshape(d,1) @ torch.ones((1, n)).to(device) * (dnu + rho * Gi @ dxk)

        lamb = lamb + rho * (Ai @ xk - bi)
        dlamb = dlamb + rho * (Ai @ dxk)

        nu = nu + rho * (Gi @ xk + sk - hi)
        dnu = dnu + rho * (Gi @ dxk + dsk)

        res.append(0.5 * (xk.T @ Pi @ xk) + qi.T @ xk)      

    return (xk, dxk)