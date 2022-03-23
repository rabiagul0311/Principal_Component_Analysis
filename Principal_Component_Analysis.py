# -*- coding: utf-8 -*-
"""

@author: W10
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import mixture
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

import numpy as np

def main():
    # Database loaded and after that data and final values were seperated.
    mnist = datasets.load_digits()
    X, y = mnist.data, mnist.target

    # In order to plot_learning_curve to work better, a suffle split were
    # created and later on, sent as a parameter to that method.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    # Here some test scenarios were created and created as a list.
    layers = [1, 2]
    neurons = [10, 50, 100, 200]

    for i in layers:
        for j in neurons:

            # We create the hidden layers with the corresponding neurons.
            network_layers = []
            for z in range(i):
                network_layers.append(j)
            z = tuple(network_layers)

            print("{} layers with {} neurons -> {}".format(i, j, z))

            # Creating the neural network with the layer and neuron numbers
            # given as a tuple.
            network = MLPClassifier(hidden_layer_sizes=z)

            # Plotting the
            plot_learning_curve(network, "{} layers with {} neurons".format(i, j), X, y, cv=cv, n_jobs=4)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

if __name__=="__main__":
    main()
