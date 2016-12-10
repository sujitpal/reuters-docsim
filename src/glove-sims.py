# -*- coding: utf-8 -*-
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
from __future__ import division, print_function
from sklearn.feature_extraction.text import CountVectorizer
import collections
import numpy as np
import os
import time

import dsutils

DATA_DIR = "../data"
EMBEDDING_SIZE = 200
VOCAB_SIZE = 5000
GLOVE_VECS = os.path.join(DATA_DIR, 
    "glove.6B.{:d}d.txt".format(EMBEDDING_SIZE))
VECTORS_FILE = os.path.join(DATA_DIR, 
    "glove-{:d}-vecs.csv".format(EMBEDDING_SIZE))

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

# read glove vectors
print("Reading GloVe vectors...", end="")
start = time.time()
glove = collections.defaultdict(lambda: np.zeros((EMBEDDING_SIZE,)))
fglove = open(GLOVE_VECS, "rb")
for line in fglove:
    cols = line.strip().split()
    word = cols[0]
    embedding = np.array(cols[1:], dtype="float32")
    glove[word] = embedding
fglove.close()
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

# use CountVectorizer to compute vocabulary
print("Extracting vocabulary...", end="")
start = time.time()
cvec = CountVectorizer(max_features=VOCAB_SIZE,
                       stop_words="english",
                       binary=True)
C = cvec.fit_transform(texts)
elapsed = time.time() - start
print("COMPLETED IN {:.3f}s".format(elapsed))

word2idx = cvec.vocabulary_
idx2word = {v:k for k, v in word2idx.items()}

# compute document vectors. This is just the sum of embeddings for
# individual words. Thus if a document contains the words "u u v"
# then the document vector is 2*embedding(u) + embedding(v).
print("Vectorizing...", end="")
X = np.zeros((C.shape[0], EMBEDDING_SIZE))
for i in range(C.shape[0]):
    row = C[i, :].toarray()
    wids = np.where(row > 0)[1]
    counts = row[:, wids][0]
    num_words = np.sum(counts)
    if num_words == 0:
        continue
    embeddings = np.zeros((wids.shape[0], EMBEDDING_SIZE))
    for j in range(wids.shape[0]):
        wid = wids[j]
        embeddings[j, :] = counts[j] * glove[idx2word[wid]]
    X[i, :] = np.sum(embeddings, axis=0) / num_words
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

print("Saving GloVe vectors...", end="")
start = time.time()
dsutils.save_vectors(X, VECTORS_FILE, is_sparse=False)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

