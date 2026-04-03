import math, copy, argparse, random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('data', type=int)
args = parser.parse_args()

LR = 1e-3
LAMB_ADV = 0.01
LAMB_CONS = 1e-5
FEAT_EPS = 0.05
EDGE_P = 10
PGD_K = 3
STEP_SIZE_X = 0.01
STEP_SIZE_A = 1.0

root = '/tmp'
if args.data == 0:
    dataset = Planetoid(root, 'Cora')
elif args.data == 1:
    dataset = Planetoid(root, 'Citeseer')
elif args.data == 2:
    dataset = Planetoid(root, 'Pubmed')
elif args.data == 3:
    dataset = WikipediaNetwork(root, 'chameleon')
elif args.data == 4:
    dataset = WikipediaNetwork(root, 'squirrel')
else:
    dataset = Actor(root)

data = dataset[0].to(device)
num_cls = dataset.num_classes
out_dim = 16

labels = dict()

for x in range(len(data.y)):
    label = int(data.y[x])
    
    try:
        labels[label].append(x)
    except KeyError:
        labels[label] = [x]

train_mask, valid_mask, test_mask = [], [], []
for c in range(num_cls):
    train_mask.extend(labels[c][0:40])
    cut = int((len(labels[c]) - 40) / 2)
    valid_mask.extend(labels[c][40:40+cut])
    test_mask.extend(labels[c][40+cut:len(labels[c])])

train, valid, test = [], [], []
for x in range(len(data.y)):
    if x in train_mask:
        train.append(True)
        valid.append(False)
        test.append(False)
    elif x in valid_mask:
        train.append(False)
        valid.append(True)
        test.append(False)
    elif x in test_mask:
        train.append(False)
        valid.append(False)
        test.append(True)
    else:
        train.append(False)
        valid.append(False)
        test.append(True)

data.train_mask, data.val_mask, data.test_mask = torch.tensor(train).to(device), torch.tensor(valid).to(device), torch.tensor(test).to(device)
    
d_in = dataset.num_node_features

class SpecBranch(nn.Module):
    def __init__(self, K=3, hidden=64, out=out_dim):
        super().__init__()
        self.c1 = ChebConv(d_in, hidden, K)
        self.c2 = ChebConv(hidden, out, K)
    def forward(self, x, edge_index):
        return self.c2(F.relu(self.c1(x, edge_index)), edge_index)

class SpatBranch(nn.Module):
    def __init__(self, hidden=64, out=out_dim):
        super().__init__()
        self.g1 = GATConv(d_in, hidden, heads=8, concat=True, dropout=0.2)
        self.g2 = GATConv(hidden * 8, out, heads=1, concat=False, dropout=0.2)
    def forward(self, x, edge_index):
        return self.g2(F.elu(self.g1(x, edge_index)), edge_index)

class SpecSphere(nn.Module):
    def __init__(self, h_gate=out_dim, d_out=out_dim):
        super().__init__()
        self.spec = SpecBranch(out=d_out)
        self.spat = SpatBranch(out=d_out)
        gate_in = d_out * 2 + 2
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, h_gate),
            nn.ReLU(),
            nn.Linear(h_gate, d_out),
            nn.Sigmoid()
        )
        self.cls = nn.Linear(d_out, num_cls)
    @staticmethod
    def graph_energy(h, edge_index):
        diff = h[edge_index[0]] - h[edge_index[1]]
        return diff.pow(2).sum(dim=1).mean()
    def forward(self, data, need_grad_score=False):
        x, ei = data.x, data.edge_index
        z_spec = self.spec(x, ei)
        z_spat = self.spat(x, ei)
        if need_grad_score:
            with torch.enable_grad():
                x_s = x.clone().detach().requires_grad_()
                logits_s = self.cls(F.relu(self.spec(x_s, ei)))
                loss_s = F.nll_loss(F.log_softmax(logits_s, 1)[data.train_mask], data.y[data.train_mask])
                grad_x_s = torch.autograd.grad(loss_s, x_s)[0]
                r_x_spec = grad_x_s.abs().sum(dim=1, keepdim=True)
                x_p = x.clone().detach().requires_grad_()
                logits_p = self.cls(F.relu(self.spat(x_p, ei)))
                loss_p = F.nll_loss(F.log_softmax(logits_p, 1)[data.train_mask], data.y[data.train_mask])
                grad_x_p = torch.autograd.grad(loss_p, x_p)[0]
                r_x_spat = grad_x_p.abs().sum(dim=1, keepdim=True)
        else:
            r_x_spec = r_x_spat = torch.zeros((x.size(0), 1), device=x.device)
        alpha = self.gate(torch.cat([z_spec, z_spat, r_x_spec, r_x_spat], dim=1))
        z = alpha * z_spec + (1 - alpha) * z_spat
        logit = self.cls(F.relu(z))
        lp_reg = self.graph_energy(z_spec, ei)
        hp_reg = -self.graph_energy(z_spat, ei)
        return F.log_softmax(logit, 1), lp_reg, hp_reg, z_spec, z_spat

model = SpecSphere().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

def pgd_feature_attack(data, branch_forward, eps, step, k):
    x_orig = data.x.detach()
    x_adv = x_orig.clone()
    for _ in range(k):
        x_adv.requires_grad_()
        data.x = x_adv
        out = branch_forward(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + step * grad.sign()).clamp(x_orig - eps, x_orig + eps).detach()
    data.x = x_orig
    return x_adv

def pgd_edge_attack(data, branch_forward, p, step, k):
    ei = data.edge_index
    A = to_dense_adj(ei)[0]
    A_ = A.clone().detach()
    for _ in range(k):
        A_.requires_grad_()
        data.edge_index = dense_to_sparse(A_)[0]
        out = branch_forward(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        grad = torch.autograd.grad(loss, A_, allow_unused=True)[0]
        if grad is None:
            break
        A_ = (A_ + step * grad.sign()).clamp(0, 1).detach()
    diff = (A_ - A).abs().triu(1)
    idx = diff.flatten().topk(p).indices
    A.view(-1)[idx] = 1 - A.view(-1)[idx]
    ei_adv = dense_to_sparse(A)[0]
    data.edge_index = ei
    return ei_adv

best = 0.0
for epoch in range(1, 1501):
    model.train()
    out, lp_reg, hp_reg, z_spec, z_spat = model(data, need_grad_score=True)
    loss_ce = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    ei_adv = pgd_edge_attack(
        data,
        lambda d: F.log_softmax(model.cls(F.relu(model.spec(d.x, d.edge_index))), 1),
        EDGE_P, STEP_SIZE_A, PGD_K
    )
    data.edge_index = ei_adv
    out_spec_adv = model.cls(F.relu(model.spec(data.x, data.edge_index)))
    loss_adv_A = F.nll_loss(F.log_softmax(out_spec_adv, 1)[data.train_mask], data.y[data.train_mask])
    data.edge_index = dataset[0].edge_index.to(device)
    x_adv = pgd_feature_attack(
        data,
        lambda d: F.log_softmax(model.cls(F.relu(model.spat(d.x, d.edge_index))), 1),
        FEAT_EPS, STEP_SIZE_X, PGD_K
    )
    data.x = x_adv
    out_spat_adv = model.cls(F.relu(model.spat(data.x, data.edge_index)))
    loss_adv_X = F.nll_loss(F.log_softmax(out_spat_adv, 1)[data.train_mask], data.y[data.train_mask])
    data.x = dataset[0].x.to(device)
    cons_all = (z_spec - z_spat).pow(2).sum(dim=1)
    thr = cons_all.median()
    bu = (cons_all < thr).float()
    loss_cons = (bu * cons_all).mean()
    loss_comp = ((1 - bu) * F.relu(0.5 - (z_spec - z_spat).norm(dim=1)).pow(2)).mean()
    loss = loss_ce + LAMB_ADV * (loss_adv_A + loss_adv_X) + LAMB_CONS * (lp_reg + hp_reg + loss_cons + loss_comp)
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.eval()
    out, _, _, _, _ = model(data)
    pred = out.argmax(1)

    val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / int(data.val_mask.sum())
    if val_acc > best:
        best = val_acc
        print(f'{epoch}:{best:.4f}')
print(best)
