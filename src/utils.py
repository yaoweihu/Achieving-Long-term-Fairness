import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sampling(probs, values=[0., 1.], coef=None, seed=2021):
    # np.random.seed(seed)

    #Synthetica data uses coef=0.8 and Taiwan data donesn't use coef
    if coef:
        probs = [pr * coef if pr > 0.5 else pr / coef for pr in probs]

    samples = []
    # probs is the probability of getting 1.
    for p in probs:
        r = np.random.uniform() # generate a random number from 0 to 1
        if r < p:
            samples.append(values[1])
        else:
            samples.append(values[0])
    return np.array(samples)


def gen_plot_data(model, **kargs):
    x_range = kargs['x_range'] if 'x_range' in kargs else [-5, 5]
    x = np.linspace(*x_range, 100)
    s0, s1 = np.zeros(len(x)), np.ones(len(x))
    y_s0 = (-model.params[-1] - model.params[0] * s0 - model.params[1] * x) / model.params[2]
    y_s1 = (-model.params[-1] - model.params[0] * s1 - model.params[1] * x) / model.params[2]
    
    line = {'x': x, 'y0': y_s0, 'y1': y_s1, **kargs}
    return line


def combine_tuples(s, Xs, Ys):
    n = len(Xs)
    S, X, Y = s, Xs[0], Ys[0]
    for x, y in zip(Xs[1:], Ys[1:]):
        S = np.hstack([S, s])
        X = np.vstack([X, x])
        Y = np.hstack([Y, y])
    return S, X, Y


def plot_data(s, X, y, n_samples=500, lines=None):
    """
    Reference from https://github.com/mbilalzafar/fair-classification/blob/master/disparate_impact/synthetic_data_demo/decision_boundary_demo.py
    """
    num_to_draw = n_samples
    x_draw = X[:num_to_draw]
    y_draw = y[:num_to_draw]
    s_draw = s[:num_to_draw]

    X_s_0 = x_draw[s_draw == 0.0]
    X_s_1 = x_draw[s_draw == 1.0]
    y_s_0 = y_draw[s_draw == 0.0]
    y_s_1 = y_draw[s_draw == 1.0]
    plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, label= "s=0 y=1", alpha=0.5)
    plt.scatter(X_s_0[y_s_0==0][:, 0], X_s_0[y_s_0==0][:, 1], color='red', marker='x', s=30, linewidth=1.5, label = "s=0. y=0", alpha=0.5)
    plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=30, label = "s=1 y=1", alpha=0.5)
    plt.scatter(X_s_1[y_s_1==0][:, 0], X_s_1[y_s_1==0][:, 1], color='red', marker='o', facecolors='none', s=30, label = "s=1, y=0", alpha=0.5)

    if lines:
        for line in lines:
            plt.plot(line['x'], line['y0'], linestyle=':', c=line['color'], label=line['label']+'(s=0)')
            plt.plot(line['x'], line['y1'], linestyle='-', c=line['color'], label=line['label']+'(s=1)')

    # dont need the ticks to see the data distribution
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1.0))
    plt.show()


def plot_axes(ax, s, X, y, n_samples=300, lines=None):

    num_to_draw = n_samples
    # num_to_draw = np.random.choice(len(X), n_samples, replace=False)
    x_draw = X[:num_to_draw]
    y_draw = y[:num_to_draw]
    s_draw = s[:num_to_draw]

    X_s_0 = x_draw[s_draw == 0.0]
    X_s_1 = x_draw[s_draw == 1.0]
    y_s_0 = y_draw[s_draw == 0.0]
    y_s_1 = y_draw[s_draw == 1.0]
    p1 = ax.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, alpha=0.5)
    p2 = ax.scatter(X_s_0[y_s_0==0][:, 0], X_s_0[y_s_0==0][:, 1], color='red', marker='x', s=30, linewidth=1.5, alpha=0.5)
    p3 = ax.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=30,  alpha=0.5)
    p4 = ax.scatter(X_s_1[y_s_1==0][:, 0], X_s_1[y_s_1==0][:, 1], color='red', marker='o', facecolors='none', s=30, alpha=0.5)

    if lines:
        for line in lines:
            ax.plot(line['x'], line['y0'], linestyle='--', c=line['color'], label="s=0")
            ax.plot(line['x'], line['y1'], linestyle='-', c=line['color'], label="s=1")

    ax.set_xlabel('$X_1$', fontsize=13, labelpad=1)
    ax.set_ylabel('$X_2$', fontsize=13, labelpad=1)
    ax.set_xticks([-10, -5, 0, 5, 10, 15])
    ax.set_yticks([-10, -5, 0, 5, 10, 15])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc=2, fontsize=13)

    return [p1, p2, p3, p4]