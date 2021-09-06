import numpy as np
from utils import *



class Bank:
    
    name = 'Bank'
    params = np.array([2.5, 2, -1, -4.0])

    def __init__(self, params=None, seed=2021):
        self.seed = seed
        if params:
            self.params = params

    def predict(self, s, X):
        Xs = np.c_[s, X, np.ones(len(s))]
        p = sigmoid(Xs @ self.params / 3.)
        y = (p >= 0.5).astype(np.float)
        return y, p


class Agent:

    def __init__(self, n_samples, protect_ratio, eps, base, seed=2021):
        self.n_samples = n_samples
        self.protect_ratio = protect_ratio
        self.eps = eps
        self.base = base
        self.seed = seed

    def set_eps(self, eps):
        self.eps = eps

    def gen_init_profile(self):
        """
        Reference from https://github.com/mbilalzafar/fair-classification/blob/master/disparate_impact/synthetic_data_demo/generate_synthetic_data.py
        """
        np.random.seed(self.seed)

        def gen_gaussian(mean, cov, sen_label, sample_size):
            s = np.ones(sample_size, dtype=float) * sen_label
            X = np.random.multivariate_normal(mean=mean, cov=cov, size=sample_size)
            return s, X

        n_protects = int(self.protect_ratio * self.n_samples)

        # We will generate one gaussian cluster for each group
        mu0, sigma0 = [-2, -2], [[10, 1], [1, 5]]
        mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
        s0, X0 = gen_gaussian(mu0, sigma0, 0, n_protects) # protected group
        s1, X1 = gen_gaussian(mu1, sigma1, 1, self.n_samples - n_protects) #  non_protected groupx
        
        # join the posisitve and negative class clusters
        s = np.hstack((s0, s1))
        X = np.vstack((X0, X1))
        
        # shuffle the data
        perm = list(range(0, self.n_samples))
        np.random.shuffle(perm)
        s = s[perm]
        X = X[perm]

        return s, X

    def gen_next_profile(self, s, X, model):
        base = [[self.base[int(i)]] for i in s]
        _, prob = model.predict(s, X)
        sample_y = sampling(prob, coef=0.8)
        _, def_prob = Bank().predict(s, X)
        default = sampling(def_prob, values=[-1, 1.], coef=0.8)

        # X change
        change = self.eps * model.params[1:-1] * prob.reshape(-1, 1)   # test w/wo prob
        # Whether default
        default_change = change * default.reshape(-1, 1)
        # Whether getting the loan
        X_next = X + sample_y.reshape(-1, 1) * default_change + np.tile(base, 2)
        return X_next


def gen_multi_step_profiles(model, agent, steps, noise=(0.05, 0.1), seed=2021):
    np.random.seed(2021)
    noise_list = noise[0] + (noise[1] - noise[0]) * np.random.rand(steps)
    
    Xs, Ys = [], []

    s, init_X = agent.gen_init_profile()
    init_Y, prob = model.predict(s, init_X)
    init_Y = sampling(prob, coef=0.8)

    Xs.append(init_X)
    Ys.append(init_Y)

    for i in range(1, steps):
        next_X = agent.gen_next_profile(s, Xs[-1], model)
        next_Y, prob = model.predict(s, next_X)
        next_Y = sampling(prob, coef=0.8)

        Xs.append(next_X)
        Ys.append(next_Y)

    return s, Xs, Ys


def generate_y_from_bank(s, Xs, bank):
    Ys = []
    for X in Xs:
        y, _ = bank.predict(s, X)
        Ys.append(y)
    return Ys