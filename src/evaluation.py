import numpy as np
from itertools import product
from sklearn.linear_model import LogisticRegression

eps = 1e-8

def compute_accuracy(s, X, y, model):
    pred_y, _ = model.predict(s, X)
    acc = sum(pred_y == y) / len(y)
    return acc


def compute_equal_opportunity(s, X, y, model): 
    X_pos = X[(y == 1) == (s == 1)]
    X_neg = X[(y == 1) == (s == 0)]
    s_pos = np.ones(len(X_pos))
    s_neg = np.zeros(len(X_neg))

    y_pos, _ = model.predict(s_pos, X_pos)
    y_neg, _ = model.predict(s_neg, X_neg)
    eo_fairness = y_pos.mean() - y_neg.mean()
    return eo_fairness
    

def compute_total_cond_fairness(s, X, model):
    s_pos, s_neg = s[s == 1], s[s == 0]
    X_pos, X_neg = X[s == 1], X[s == 0]

    y_pos, _ = model.predict(s_pos, X_pos)
    y_neg, _ = model.predict(s_neg, X_neg) 
    fairness = y_pos.mean() - y_neg.mean()
    return fairness


def compute_short_cond_fairness(s, X, model):
    X = X[s == 0]
    s = s[s == 0]

    s_pos = np.ones_like(s)
    s_neg = np.zeros_like(s)
        
    y_pos, _ = model.predict(s_pos, X)
    y_neg, _ = model.predict(s_neg, X)

    fairness = y_pos.mean() - y_neg.mean()
    return fairness


def compute_post_long_cond_probs(s, Xs, Ys):
    probs = {}
    
    for i in range(len(Xs)-1):
        
        XXs_comb = np.c_[s[s == 1], Xs[i][s == 1], Xs[i+1][s == 1]]
        Xs_comb = np.c_[s[s == 1], Xs[i][s == 1]]
        lr_up = LogisticRegression(random_state=2021).fit(XXs_comb, Ys[i][s == 1])
        lr_dn = LogisticRegression(random_state=2021).fit(Xs_comb, Ys[i][s == 1])
        probs_up = lr_up.predict_proba(XXs_comb)
        probs_dn = lr_dn.predict_proba(Xs_comb)
        probs[f'pos(y{i+1}=0)'] = probs_up[:, 0] / probs_dn[:, 0]
        probs[f'pos(y{i+1}=1)'] = probs_up[:, 1] / probs_dn[:, 1]


        XXs_comb = np.c_[s[s == 0], Xs[i][s == 0], Xs[i+1][s == 0]]
        Xs_comb = np.c_[s[s == 0], Xs[i][s == 0]]
        lr_up = LogisticRegression(random_state=2021).fit(XXs_comb, Ys[i][s == 0])
        lr_dn = LogisticRegression(random_state=2021).fit(Xs_comb, Ys[i][s == 0])
        probs_up = lr_up.predict_proba(XXs_comb)
        probs_dn = lr_dn.predict_proba(Xs_comb)
        probs[f'neg(y{i+1}=0)'] = probs_up[:, 0] / probs_dn[:, 0]
        probs[f'neg(y{i+1}=1)'] = probs_up[:, 1] / probs_dn[:, 1]
    return probs
    

def compute_post_long_cond_fairness(s, Xs, model, prob=None):
    outputs = {}
    for i in range(len(Xs)-1):
        y_pos, _ = model.predict(s[s == 1], Xs[i][s == 1])
        y_neg, _ = model.predict(s[s == 0], Xs[i][s == 0])
        outputs[f'pos(y{i+1}=0)'] = 1 - y_pos
        outputs[f'pos(y{i+1}=1)'] = y_pos
        outputs[f'neg(y{i+1}=0)'] = 1 - y_neg
        outputs[f'neg(y{i+1}=1)'] = y_neg

    y_pos, _ = model.predict(np.zeros_like(s[s == 1]), Xs[-1][s == 1])
    y_neg, _ = model.predict(np.zeros_like(s[s == 0]), Xs[-1][s == 0])

    indices = [[0, 1]] * (len(Xs) - 1)

    part1, part2 = None, None
    for idx in product(*indices):
        pos, neg = 1, 1
        for i in range(len(Xs)-1):
            pos *= (prob[f'pos(y{i+1}={idx[i]})'] * outputs[f'pos(y{i+1}={idx[i]})'])
            neg *= (prob[f'neg(y{i+1}={idx[i]})'] * outputs[f'neg(y{i+1}={idx[i]})'])
        if part1 is None:
            part1 = pos
        else:
            part1 += pos
        if part2 is None:
            part2 = neg
        else:
            part2 += neg

    if part1 is not None or part2 is not None:
        fairness = np.mean(y_pos * part1) - np.mean(y_neg * part2)
    else:
        fairness = np.mean(y_pos) - np.mean(y_neg)
    return fairness


def compute_statistics(s, Xs, Ys, model, OYs=None):

    for i, (X, y) in enumerate(zip(Xs, Ys)):
        print("-" * 30, f"Step {i + 1} - {model.name}", "-" * 30)

        if OYs:
            acc = compute_accuracy(s, X, OYs[i], model)
        else:
            acc = compute_accuracy(s, X, y, model)
        print(f"Acc: {acc * 100:.1f}%")

        # op_fair = compute_equal_opportunity(s, X, y, model)
        # print(f"Equal Oppertunity: {abs(op_fair):.3f}")

        # cond_fair = compute_total_cond_fairness(s, X, model)
        # print(f"Total Fairness: {abs(cond_fair):.3f}")

        short_fair_cond = compute_short_cond_fairness(s, X, model)
        print(f"Short Fairness: {abs(short_fair_cond):.3f}")

        if i == 0:
            post_long_cond_fairness = compute_post_long_cond_fairness(s, Xs[:i+1], model)
        else:
            post_long_cond_prob = compute_post_long_cond_probs(s, Xs[:i+1], Ys[:i+1])
            post_long_cond_fairness = compute_post_long_cond_fairness(s, Xs[:i+1], model, post_long_cond_prob)

        print(f"Long fairness: {abs(post_long_cond_fairness):.3f}")
    print("\n")