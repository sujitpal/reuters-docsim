# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import time

import dsutils

DATA_DIR = "../data"
MAX_FEATURES = 300
VECTORS_FILE = os.path.join(DATA_DIR, 
    "tfidf-{:d}-vecs.mtx".format(MAX_FEATURES))

start = time.time()
texts = []
num_read = 0
ftext = open(os.path.join(DATA_DIR, "text.tsv"), "rb")
for line in ftext:
    if num_read % 100 == 0:
        print("{:d} lines of text read".format(num_read))
    docid, text = line.strip().split("\t")
    texts.append(text)
    num_read += 1

ftext.close()
elapsed = time.time() - start
print("{:d} lines of text read, COMPLETED in {:.3f}s"
    .format(num_read, elapsed))

print("vectorizing...", end="")
start = time.time()
tvec = TfidfVectorizer(max_features=MAX_FEATURES,
                       min_df=0.1, sublinear_tf=True,
                       stop_words="english",
                       binary=True)

X = tvec.fit_transform(texts)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

print("Saving TF-IDF vectors...", end="")
start = time.time()
dsutils.save_vectors(X, VECTORS_FILE, is_sparse=True)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

