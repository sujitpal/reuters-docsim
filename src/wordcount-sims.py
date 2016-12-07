# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.feature_extraction.text import CountVectorizer
import os
import time

import dsutils

DATA_DIR = "../data"
MAX_FEATURES = 50
VECTORS_FILE = os.path.join(DATA_DIR, 
    "wordcount-{:d}-vecs.mtx".format(MAX_FEATURES))

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
cvec = CountVectorizer(max_features=MAX_FEATURES,
                       stop_words="english")
X = cvec.fit_transform(texts)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

print("Saving wordcount vectors...", end="")
start = time.time()
dsutils.save_vectors(X, VECTORS_FILE, is_sparse=True)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

