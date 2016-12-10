# -*- coding: utf-8 -*-
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
from __future__ import division, print_function
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import time

import dsutils

DATA_DIR = "../data"
MAX_FEATURES = 300
VOCAB_SIZE = 5000
WORD2VEC_MODEL = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin.gz")
VECTORS_FILE = os.path.join(DATA_DIR, 
    "w2v-{:d}-vecs.csv".format(MAX_FEATURES))

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

# read word2vec vectors
print("Reading Word2Vec vectors...", end="")
start = time.time()
word2vec = Word2Vec.load_word2vec_format(WORD2VEC_MODEL, binary=True)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

# use CountVectorizer to compute vocabulary
print("Extracting vocabulary...", end="")
start = time.time()
cvec = CountVectorizer(max_features=VOCAB_SIZE,
                       stop_words="english",
                       binary=True)
C = cvec.fit_transform(texts)

word2idx = cvec.vocabulary_
idx2word = {v:k for k, v in word2idx.items()}
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

# compute document vectors. This is just the sum of embeddings for
# individual words. Thus if a document contains the words "u u v"
# then the document vector is 2*embedding(u) + embedding(v).
print("vectorizing...", end="")
start = time.time()
X = np.zeros((C.shape[0], 300))
for i in range(C.shape[0]):
    row = C[i, :].toarray()
    wids = np.where(row > 0)[1]
    counts = row[:, wids][0]
    num_words = np.sum(counts)
    if num_words == 0:
        continue
    embeddings = np.zeros((wids.shape[0], MAX_FEATURES))
    for j in range(wids.shape[0]):
        wid = wids[j]
        try:
            emb = word2vec[idx2word[wid]]
            embeddings[j, :] = counts[j] * emb
        except KeyError:
            continue
    X[i, :] = np.sum(embeddings, axis=0) / num_words
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

print("Saving Word2Vec vectors...", end="")
start = time.time()
dsutils.save_vectors(X, VECTORS_FILE, is_sparse=False)
elapsed = time.time() - start
print("COMPLETED in {:.3f}s".format(elapsed))

