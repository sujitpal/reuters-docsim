# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.feature_extraction.text import CountVectorizer
import collections
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import re

DATA_DIR = "../data"
NUM_TOP_TAGS = 20

tags = []
num_docs = 0
ftags = open(os.path.join(DATA_DIR, "tags.tsv"), "rb")
for line in ftags:
    docid, taglist = line.strip().split("\t")
    tags.append(re.sub(",", " ", taglist))
    num_docs += 1
ftags.close()

### tag counts
cvec = CountVectorizer()
X = cvec.fit_transform(tags)

numtags = np.sum(X, axis=1)

print("Number of documents: {:d}".format(X.shape[0]))
print("Number of unique tags: {:d}".format(X.shape[1]))
print("Max #-tags per doc: {:d}".format(np.max(numtags)))
print("Mean #-tags per doc: {:.3f}".format(np.mean(numtags)))
print("Median #-tags per doc: {:.3f}".format(np.median(np.array(numtags))))

plt.subplot(211)
plt.hist(numtags, bins=np.max(numtags))
plt.title("Distribution of number of tags / doc")
plt.xlabel("#-tags per document")
plt.ylabel("count")

### top tags

cvec = CountVectorizer(max_features=NUM_TOP_TAGS)
X = cvec.fit_transform(tags)

term2id = cvec.vocabulary_
id2term = {v:k for k, v in term2id.items()}

freqs = np.sum(X, axis=0).tolist()[0]
terms = [id2term[i] for i in range(len(id2term))]
termfreqs = sorted(zip(terms, freqs), key=operator.itemgetter(1), 
                   reverse=True)
terms = [x[0] for x in termfreqs]
freqs = [x[1] for x in termfreqs]

# visualize top tags
plt.subplot(212)
plt.bar(range(len(terms)), freqs)
plt.xticks(np.arange(len(terms)) + 0.35, terms, rotation=90)
plt.title("Distribution of top {:d} tags".format(len(terms)))
plt.ylabel("counts")

plt.tight_layout()
plt.show()
