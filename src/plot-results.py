# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = "../data"
RESULTS_FILE = os.path.join(DATA_DIR, "results.tsv")

vecs, pmins, pmaxs = [], [], []
fres = open(RESULTS_FILE, "rb")
for line in fres:
    if line.startswith("#"):
        continue
    vec, pmin, pmax = line.strip().split("\t")
    vecs.append(vec)
    pmins.append(float(pmin))
    pmaxs.append(float(pmax) - float(pmin))
fres.close()

inds = np.arange(len(vecs))
plt.bar(inds, np.array(pmins), color="lightgray")
plt.bar(inds, np.array(pmaxs), bottom=np.array(pmins), color="b")
plt.xticks(inds + 0.35, vecs, rotation="30")
plt.xlabel("vectorizer")
plt.ylabel("pearson correlation")
plt.show()