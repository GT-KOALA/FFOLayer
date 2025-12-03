import torch
from torch.autograd import Function
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode(X_):
    a = []
    X = X_.numpy()
    for i in range(len(X)):
        a.append(X[i])
    return a

def relu(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = 0
    return ss

def sgn(s):
    ss = torch.zeros(len(s)).to(device)
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

def alt_diff_qp(P, q, A, b, G, h):

    n, m, p = P.shape[0], b.shape[0], h.shape[0]
    xk = torch.zeros(n).to(device)
    sk = torch.zeros(p).to(device)
    lamb = torch.zeros(m).to(device)
    nu = torch.zeros(p).to(device)

    dxk = torch.zeros((n, n)).to(device)
    dsk = torch.zeros((p, n)).to(device)
    dlamb = torch.zeros((m, n)).to(device)
    dnu = torch.zeros((p,n)).to(device)

    rho = torch.tensor([1.0]).float().to(device)

    R = - torch.linalg.inv(P + rho * A.T @ A + rho * G.T @ G)

    res = [1000*torch.ones(1).to(device), -100*torch.ones(1).to(device)]
    # thres = 1e-5

    xk = torch.ones(n).to(device)
    thres = 1e-3
    while abs((torch.linalg.norm(res[-1]) - torch.linalg.norm(res[-2])) / torch.linalg.norm(res[-2])) > thres:
        # print(b)
        xk = R @ (q + A.T @ lamb + G.T @ nu - rho * A.T @ b + rho * G.T @ (sk - h))

        dxk = R @ (torch.ones(n).to(device) + A.T @ dlamb + G.T @ dnu  + rho * G.T @ dsk)
        sk = relu(- (1 / rho) * nu - (G @ xk - h))

        dsk = (-1 / rho) * sgn(sk).view(p, 1) @ (torch.ones((1,n)).to(device)) * (dnu + rho * G @ dxk)

        lamb = lamb + rho * (A @ xk - b)
        dlamb = dlamb + rho * (A @ dxk)
        nu = nu + rho * (G @ xk + sk - h)
        dnu = dnu + rho * (G @ dxk + dsk)
        res.append(xk)

    y_f = dxk
    return xk, y_f

def AltDiffLayer(eps=1e-3, verbose=0):
    class Newlayer(Function):
        @staticmethod
        def forward(ctx, P_, q_):
            n, m, d = q_.shape[0], 1, q_.shape[0]
            G_ = torch.diag_embed(torch.ones(n)).to(device)
            h_ = torch.zeros(n).to(device)
            A_ = torch.ones(n).unsqueeze(0).to(device)
            b_ = torch.tensor([1]).float().to(device)

            # print(n, m, d)
            P = P_.detach()
            q = q_.detach()
            G = G_.detach()
            h = h_.detach()
            A = A_.detach()
            b = b_.detach()
            # Define and solve the CVXPY problem.

            xk, dxk = alt_diff_qp(P, q, A, b, G, h)

            ctx.save_for_backward(dxk)
            return xk

        @staticmethod
        def backward(ctx, grad_output):
            # only call parameters q
            grad = ctx.saved_tensors
            grad_all = torch.mv(grad[0],grad_output)
            return (None, grad_all)

    return Newlayer.apply

if __name__ == "__main__":
    n = 20
    P = np.random.random((n,n))
    P = P.T@P+(0.0001*np.eye(n,n))
    q = np.random.random(n)
    P_ = torch.tensor(P).float().to(device)
    q_ = torch.tensor(q).float().to(device)
    q_.requires_grad = True
    net = AltDiff()
    pred = net(P_,q_)
    loss = pred@torch.ones(20).to(device)
    loss.backward()
    print(pred.grad)
    n, m, d = q.shape[0], 1, q.shape[0]
    G_ = torch.diag_embed(torch.ones(n)).to(device)
    h_ = torch.zeros(n).to(device)
    A_ = torch.ones(n).unsqueeze(0).to(device)
    b_ = torch.tensor([1]).float().to(device)
    xk, dxk = alt_diff_qp(P_, q_, A_, b_, G_, h_)
    print(xk)
    print(dxk)