# -*- coding: utf-8 -*-
from __future__ import division, print_function
from scipy import stats
import os
import time

import dsutils

DATA_DIR = "../data"

#VECTORIZER = "wordcount"
#VECTORIZER = "tfidf"
#VECTORIZER = "lsa"
#VECTORIZER = "glove"
VECTORIZER = "w2v"

X_IS_SPARSE = True
#Y_IS_SPARSE = True
Y_IS_SPARSE = False

NUM_FEATURES = 300

XFILE = os.path.join(DATA_DIR, "tag-vecs.mtx")
YFILE = os.path.join(DATA_DIR, "{:s}-{:d}-vecs.{:s}"
    .format(VECTORIZER, NUM_FEATURES, 
            "mtx" if Y_IS_SPARSE else "csv"))

print("Loading vectors...", end="")
start = time.time()
X = dsutils.load_vectors(XFILE, is_sparse=X_IS_SPARSE)
Y = dsutils.load_vectors(YFILE, is_sparse=Y_IS_SPARSE)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

print("Computing similarity matrix for LHS...")
start = time.time()
XD = dsutils.compute_cosine_sims(X, is_sparse=X_IS_SPARSE)
print("Computing similarity matrix for RHS...")
YD = dsutils.compute_cosine_sims(Y, is_sparse=Y_IS_SPARSE)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

print("Extracting upper triangles to array...", end="")
start = time.time()
XDT = dsutils.get_upper_triangle(XD, is_sparse=X_IS_SPARSE)
YDT = dsutils.get_upper_triangle(YD, is_sparse=Y_IS_SPARSE)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

corr, _ = stats.pearsonr(XDT, YDT)
print("Pearson correlation: {:.3f}".format(corr))

## create scatterplot using random sample of points from XDT and YDT
dsutils.plot_correlation(XDT, YDT, VECTORIZER, corr)
