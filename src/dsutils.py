# -*- coding: utf-8 -*-
from __future__ import division, print_function
from scipy import sparse, io, stats
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

def compute_cosine_sims(X, is_sparse=True):
    if is_sparse:
        Xnormed = X / sparse.linalg.norm(X, "fro")
        Xtnormed = X.T / sparse.linalg.norm(X.T, "fro")
        S = Xnormed * Xtnormed
    else:
        Xnormed = X / LA.norm(X, ord="fro")
        Xtnormed = X.T / LA.norm(X.T, ord="fro")
        S = np.dot(Xnormed, Xtnormed)
    return S


def save_vectors(X, filename, is_sparse=True):
    if is_sparse:
        io.mmwrite(filename, X)
    else:
        np.savetxt(filename, X, delimiter=",", fmt="%.5e")


def load_vectors(filename, is_sparse=True):
    if is_sparse:
        return io.mmread(filename)
    else:
        return np.loadtxt(filename, delimiter=",")


def get_upper_triangle(X, k=1, is_sparse=True):
    if is_sparse:
        return sparse.triu(X, k=k).toarray().flatten()
    else:
        return np.triu(X, k=k).flatten()


def plot_correlation(X, Y, title, corr=None):
    if corr == None:
        corr, _ = stats.pearsonr(X, Y)
    # extract 90-th percentile
    thresh = np.percentile(Y, 99)
    X90 = X[X > thresh]
    Y90 = Y[X > thresh]
    sample = np.random.choice(X90.shape[0], size=100, replace=False)
    Xsample = X90[sample]
    Ysample = Y90[sample]
    plt.scatter(Xsample, Ysample, color="red")
    plt.xlim([np.min(Xsample), np.max(Xsample)])
    plt.ylim([np.min(Ysample), np.max(Ysample)])
    plt.title("{:s} (corr: {:.3f})".format(title, corr))
    plt.xlabel("X")
    plt.ylabel("Y")

