import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from evaluation import *
torch.manual_seed(2021)


def to_tensor(x):
    if not torch.is_tensor(x):
        x = torch.FloatTensor(x)
    return x


def to_numpy(x):
    if x.is_cuda:
        x = x.cpu()
    x = x.detach().numpy()
    return x


def logistic(x):
    return torch.log(1 + torch.exp(x))


def combine_featuers(s, X):
    s = to_tensor(s)
    X = to_tensor(X)
    return torch.cat((s.view(-1, 1), X), dim=1)



class FairModel(nn.Module):

    name = 'Long-term Fair Model'

    def __init__(self, n_features, lr, l2_reg, sf_reg, lf_reg):
        super().__init__()
        self.l2_reg = l2_reg
        self.sf_reg = sf_reg
        self.lf_reg = lf_reg

        self.linear = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.old_linear_weight = None
        self.old_linear_bias = None

        
        torch.manual_seed(2021)
        nn.init.normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, s, X):
        Xs = combine_featuers(s, X)
        h = self.linear(Xs)
        p = self.sigmoid(h)
        return h.squeeze(), p.squeeze()

    def prev_forward(self, s, X):
        Xs = combine_featuers(s, X)
        h = F.linear(Xs, self.old_linear_weight, self.old_linear_bias)
        p = self.sigmoid(h)
        return h.squeeze(), p.squeeze()

    def predict(self, s, X):
        _, p = self.forward(s, X)
        pred_y = torch.round(p)
        return to_numpy(pred_y), to_numpy(p)

    @property
    def params(self):
        w = to_numpy(self.linear.weight)[0]
        b = to_numpy(self.linear.bias)
        wb = np.hstack([w, b])
        return wb

    def save_params(self):
        self.old_linear_weight = self.linear.weight
        self.old_linear_bias = self.linear.bias

    def compute_loss(self, s, X, y):
        _, pred_y = self.forward(s, X)
        loss = self.loss(pred_y, y)
        l2 = torch.norm(self.linear.weight) ** 2
        return loss + self.l2_reg * l2

    def compute_short_fairness_from_cond_dist(self, s, X):
        X = X[s == 0]
        s = s[s == 0]
        s_pos = np.ones_like(s)
        s_neg = np.zeros_like(s)
        
        y_pos, _ = self.forward(s_pos, X)
        y_neg, _ = self.forward(s_neg, X)

        fair_cons1 = torch.mean(logistic(y_pos) + logistic(-y_neg) - 1)
        fair_cons2 = torch.mean(logistic(y_neg) + logistic(-y_pos) - 1)
        return torch.relu(fair_cons1), torch.relu(fair_cons2)

    def compute_post_long_fairness_from_cond_dist(self, s, Xs, Ys, probs):
        for k, v in probs.items():
            probs[k] = to_tensor(v)

        outputs = {}
        for i in range(len(Xs)-1):
            y_pos, _ = self.forward(s[s == 1], Xs[i][s == 1])
            y_neg, _ = self.forward(s[s == 0], Xs[i][s == 0])
            outputs[f'pos(y{i+1}=0)'] = logistic(-y_pos)
            outputs[f'pos(y{i+1}=1)'] = logistic(y_pos)
            outputs[f'neg(y{i+1}=0)'] = logistic(-y_neg)
            outputs[f'neg(y{i+1}=1)'] = logistic(y_neg)

        y_pos, _ = self.forward(np.zeros_like(s[s == 1]), Xs[-1][s == 1])
        y_neg, _ = self.forward(np.zeros_like(s[s == 0]), Xs[-1][s == 0])

        indices = [[0, 1]] * (len(Xs) - 1)

        part1, part2 = 0, 0
        for idx in product(*indices):
            pos, neg = 1, 1
            for i in range(len(Xs)-1):
                pos *= (probs[f'pos(y{i+1}={idx[i]})'] * outputs[f'pos(y{i+1}={idx[i]})'])
                neg *= (probs[f'neg(y{i+1}={idx[i]})'] * outputs[f'neg(y{i+1}={idx[i]})'])
            part1 += pos
            part2 += neg

        fair_cons1 = torch.mean(logistic(y_pos) * part1) + torch.mean(logistic(-y_neg) * part2) - 1
        fair_cons2 = torch.mean(logistic(-y_pos) * part1) + torch.mean(logistic(y_neg) * part2) - 1
        
        return torch.relu(fair_cons1)

    def train(self, s, OXs, OYs, Xs, Ys, epochs=0, plot=True, tol=1e-7, short_type='pos'):
        
        long_probs = compute_post_long_cond_probs(s, Xs, Ys)
        losses, o_losses, s_fairs, l_fairs = [], [], [], []

        gap = 1e30
        pre_loss = 1e30
        while gap > tol or epochs > 0:

            loss, o_loss, s_fair = 0, 0, 0
            for i, (OX, Oy, X, y) in enumerate(zip(OXs, OYs, Xs, Ys)):
                Oy = to_tensor(Oy)
                y = to_tensor(y)
                
                o_loss += self.compute_loss(s, OX, Oy)
                s_fair_pos, s_fair_neg = self.compute_short_fairness_from_cond_dist(s, X)
                if short_type == 'pos':
                    s_fair += s_fair_pos
                if short_type == 'neg':
                    s_fair += s_fair_neg

            l_fair = self.compute_post_long_fairness_from_cond_dist(s, Xs, Ys, long_probs)
            
            loss = o_loss + self.sf_reg * s_fair + self.lf_reg * l_fair

            losses.append(loss.item())
            o_losses.append(o_loss.item())
            s_fairs.append(s_fair.item())
            l_fairs.append(l_fair.item())

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            gap = pre_loss - loss
            pre_loss = loss
            epochs -= 1

        self.save_params()
        if plot:
            self.plot_data(losses, o_losses, s_fairs, l_fairs)
                

    def plot_data(self, losses, o_losses, s_fairs, l_fairs):
        fig = plt.figure(figsize=(16,4)) 
        plt.subplot(141)
        plt.plot(range(len(losses)), losses)
        plt.xlabel('epochs')
        plt.ylabel('totala_loss')
        plt.subplot(142)
        plt.plot(range(len(o_losses)), o_losses)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.subplot(143)
        plt.plot(range(len(s_fairs)), s_fairs)
        plt.xlabel('epochs')
        plt.ylabel('short_fair')
        plt.subplot(144)
        plt.plot(range(len(l_fairs)), l_fairs)
        plt.xlabel('epochs')
        plt.ylabel('long_fair')
        fig.tight_layout()
        plt.show()