import numpy as np
import cvxpy as cp
from utils import sigmoid
from sklearn.linear_model import LogisticRegression



class LR:

    name = 'Logistic Regression'

    def __init__(self, l2_reg):
        self.model = LogisticRegression(C=1.0/l2_reg, random_state=2021)

    def train(self, s, X, y):
        Xs = np.c_[s, X]
        self.model.fit(Xs, y)

    def predict(self, s, X):
        Xs = np.c_[s, X]
        p = self.model.predict_proba(Xs)[:, 1]
        y = self.model.predict(Xs)
        return y, p

    @property
    def params(self):
        return np.r_[self.model.coef_[0], self.model.intercept_]


class CvxFairModel:

    name = 'Fair Model with Demographic Parity'

    def __init__(self, n_features, l2_reg, tao):
        self.l2_reg = l2_reg
        self.tao = tao
        self.w = cp.Variable(n_features)

    def add_intercept(self, s, X):
        return np.c_[s, X, np.ones_like(s)]

    def compute_loss(self, s, X, y):
        X = self.add_intercept(s, X)
        n = X.shape[0]

        # compute log likelihood
        t1 = 1.0/n * cp.sum(-1.0 * cp.multiply(y, X @ self.w) + cp.logistic(X @ self.w))
        # add l2_reg
        t2 = self.l2_reg * cp.norm(self.w[:-1]) ** 2
        return t1 + t2

    def compute_constraint(self, s, X):
        X = self.add_intercept(s, X)
        n = X.shape[0]
        
        X_pos = X[s == 1]
        X_neg = X[s == 0]

        h_pos = X_pos @ self.w
        h_neg = X_neg @ self.w
        c1 = 1.0 / len(X_pos) * cp.sum(cp.logistic(h_pos)) + 1.0 / len(X_neg) * cp.sum(cp.logistic(-h_neg))
        c2 = 1.0 / len(X_pos) * cp.sum(cp.logistic(-h_pos)) + 1.0 / len(X_neg) * cp.sum(cp.logistic(h_neg))
        return c1, c2

    def train(self, s, X, y):
        loss = self.compute_loss(s, X, y)
        c1, c2 = self.compute_constraint(s, X)
        cons = [c2 <= self.tao, c1 <= self.tao]
        obj = cp.Minimize(loss)
        prob = cp.Problem(obj, cons)
        prob.solve()
        print(prob.status)

    def predict(self, s, X):
        X = self.add_intercept(s, X)
        h = X @ self.w.value
        pred_y = (h >= 0).astype(float)
        p = sigmoid(h) 
        return pred_y, p 

    @property
    def params(self):
        return self.w.value


class EOFairModel:

    name = 'Fair Model with Equal Oppertunity'

    def __init__(self, n_features, l2_reg, tao):
        self.l2_reg = l2_reg
        self.tao = tao
        self.w = cp.Variable(n_features)

    def add_intercept(self, s, X):
        return np.c_[s, X, np.ones_like(s)]

    def compute_loss(self, s, X, y):
        X = self.add_intercept(s, X)
        n = X.shape[0]

        # compute log likelihood
        t1 = 1.0/n * cp.sum(-1.0 * cp.multiply(y, X @ self.w) + cp.logistic(X @ self.w))
        # add l2_reg
        t2 = self.l2_reg * cp.norm(self.w[:-1]) ** 2
        return t1 + t2

    def compute_constraint(self, s, X, y):
        X = self.add_intercept(s, X)
        n = X.shape[0]
        
        X_pos = X[(y == 1) == (s == 1)]
        X_neg = X[(y == 1) == (s == 0)]

        h_pos = X_pos @ self.w
        h_neg = X_neg @ self.w
        c1 = 1.0 / len(X_pos) * cp.sum(cp.logistic(h_pos)) + 1.0 / len(X_neg) * cp.sum(cp.logistic(-h_neg))
        c2 = 1.0 / len(X_pos) * cp.sum(cp.logistic(-h_pos)) + 1.0 / len(X_neg) * cp.sum(cp.logistic(h_neg))
        return c1, c2

    def train(self, s, X, y):
        loss = self.compute_loss(s, X, y)
        c1, c2 = self.compute_constraint(s, X, y)
        cons = [c2 <= self.tao, c1 <= self.tao]
        obj = cp.Minimize(loss)
        prob = cp.Problem(obj, cons)
        prob.solve()
        print(prob.status)

    def predict(self, s, X):
        X = self.add_intercept(s, X)
        h = X @ self.w.value
        pred_y = (h >= 0).astype(float)
        p = sigmoid(h) 
        return pred_y, p 

    @property
    def params(self):
        return self.w.value