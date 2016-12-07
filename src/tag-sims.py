# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
import time

import dsutils

DATA_DIR = "../data"
VECTORS_FILE = os.path.join(DATA_DIR, "tag-vecs.mtx")

start = time.time()
tags = []
num_read = 0
ftags = open(os.path.join(DATA_DIR, "tags.tsv"), "rb")
for line in ftags:
    if num_read % 100 == 0:
        print("{:d} lines of tags read".format(num_read))
    docid, taglist = line.strip().split("\t")
    taglist = re.sub(",", " ", taglist)
    tags.append(taglist)
    num_read += 1

ftags.close()
elapsed = time.time() - start
print("{:d} lines of text read, COMPLETED in {:.3f}s"
    .format(num_read, elapsed))

print("vectorizing tags...", end="")
start = time.time()
cvec = CountVectorizer()
X = cvec.fit_transform(tags)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

print("Saving tag vectors...", end="")
start = time.time()
dsutils.save_vectors(X, VECTORS_FILE, is_sparse=True)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))
