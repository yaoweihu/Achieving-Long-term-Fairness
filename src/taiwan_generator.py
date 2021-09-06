import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils import *


def read_taiwan_data():
    file = "../data/default of credit card clients.xls"
    df = pd.read_excel(file, header=1)

    df = df[(df['PAY_AMT1'] < 10000) & (df['PAY_AMT1'] > 10)]
    df = df[(df['PAY_AMT2'] < 10000) & (df['PAY_AMT2'] > 10)]

    label0 = df[(df['default payment next month'] == 1) & (df['SEX'] == 1)].sample(n=1000, replace=False, random_state=2021)
    label1 = df[(df['default payment next month'] == 0) & (df['SEX'] == 1)].sample(n=1000, replace=False, random_state=2021)
    label2 = df[(df['default payment next month'] == 1) & (df['SEX'] == 2)].sample(n=1000, replace=False, random_state=2021)
    label3 = df[(df['default payment next month'] == 0) & (df['SEX'] == 2)].sample(n=1000, replace=False, random_state=2021)
    df = pd.concat([label0, label1, label2, label3], axis=0)

    X = df.iloc[:, 18:20].apply(lambda x: 3 * (x - np.min(x)) / (np.max(x) - np.min(x)))
    S = df['SEX'] - 1
    Y = df['default payment next month'].replace({0:1, 1:0})

    return np.array(S), np.array(X), np.array(Y)


def preprocess_data():
    np.random.seed(2021)

    S, X, Y = read_taiwan_data()
    lr = LogisticRegression()
    lr.fit(np.c_[S, X], Y)
    y = lr.predict(np.c_[S, X])
    
    Ss = []
    Xs = []
    Ys = []
    count = 0
    for i in range(len(S)):
        r = np.random.uniform() # generate a random number from 0 to 1
        if y[i] == Y[i] and S[i] == 0 and y[i] == 0:
            Ss.append(S[i])
            Xs.append(X[i])
            Ys.append(Y[i])
            count += 1
        if count >= 600:
            break

    count = 0
    for i in range(len(S)):
        r = np.random.uniform() # generate a random number from 0 to 1
        if y[i] == Y[i] and S[i] == 0 and y[i] == 1:
            Ss.append(S[i])
            Xs.append(X[i])
            Ys.append(Y[i])
            count += 1
        if count >= 600:
            break

    count = 0
    for i in range(len(S)):
        r = np.random.uniform() # generate a random number from 0 to 1
        if y[i] == Y[i] and S[i] == 1 and y[i] == 0:
            Ss.append(S[i])
            Xs.append(X[i])
            Ys.append(Y[i])
            count += 1
        if count >= 600:
            break

    count = 0
    for i in range(len(S)):
        r = np.random.uniform() # generate a random number from 0 to 1
        if y[i] == Y[i] and S[i] == 1 and y[i] == 1:
            Ss.append(S[i])
            Xs.append(X[i])
            Ys.append(Y[i])
            count += 1
        if count >= 600:
            break
    
    count = 0
    for i in range(len(S)):
        r = np.random.uniform() # generate a random number from 0 to 1
        if y[i] != Y[i]:
            Ss.append(S[i])
            Xs.append(X[i])
            Ys.append(Y[i])
            count += 1
        if count >= 600:
            break

    S = np.array(Ss)
    X = np.array(Xs)
    Y = np.array(Ys)

    # shuffle the data
    perm = list(range(0, len(S)))
    np.random.shuffle(perm)
    S = S[perm]
    X = X[perm]
    Y = Y[perm]

    params = list(lr.coef_[0]) + list(lr.intercept_)

    return S, X, Y, params


def split_data(S, X, Y):
    s_tr, s_te, X_tr, X_te, y_tr, y_te = train_test_split(S, X, Y, test_size=0.2, random_state=2020)
    return (s_tr, X_tr, y_tr), (s_te, X_te, y_te)


class Bank:
    
    name = 'Bank'
    params = None
    
    def __init__(self, params=None, seed=2021):
        self.seed = seed
        if params:
            self.params = np.array(params)

    def predict(self, s, X):
        Xs = np.c_[s, X, np.ones(len(s))]
        p = sigmoid(Xs @ self.params)
        y = (p >= 0.5).astype(np.float)
        return y, p


class Agent:

    def __init__(self, s, X, y, eps, base, seed=2021):
        self.eps = eps
        self.base = base
        self.seed = seed
        self.s = s
        self.X = X
        self.y = y

    def set_eps(self, eps):
        self.eps = eps

    def gen_init_profile(self):
        return self.s, self.X, self.y

    def gen_next_profile(self, s, X, model):
        base = [[self.base[int(i)]] for i in s]
        _, prob = model.predict(s, X)
        sample_y = sampling(prob)
        _, _, _, PARAMS = preprocess_data()
        _, def_prob = Bank(params=PARAMS).predict(s, X)
        default = sampling(def_prob, values=[-1, 1.])
        
        # X change
        change = self.eps * model.params[1:-1] * prob.reshape(-1, 1)   # test w/wo prob
        # Whether default
        default_change = change * default.reshape(-1, 1)
        # Whether getting the loan
        X_next = X + sample_y.reshape(-1, 1) * default_change + np.tile(base, len(model.params[1:-1]))
        return X_next


def gen_multi_step_profiles(model, agent, steps, noise=(0.05, 0.1), seed=2021):
    np.random.seed(seed)
    
    Xs, Ys = [], []

    s, init_X, init_Y = agent.gen_init_profile()
    
    Xs.append(init_X)
    Ys.append(init_Y)

    for i in range(1, steps):
        next_X = agent.gen_next_profile(s, Xs[-1], model)
        next_Y, prob = model.predict(s, next_X)
        next_Y = sampling(prob, coef=1.8, seed=seed)
        
        Xs.append(next_X)
        Ys.append(next_Y)

    return s, Xs, Ys


def generate_y_from_bank(s, Xs, bank):

    Ys = []
    for X in Xs:
        y, prob = bank.predict(s, X)
        y = sampling(prob, coef=1.8)
        Ys.append(y)
    return Ys