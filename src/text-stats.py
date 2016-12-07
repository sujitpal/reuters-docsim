# -*- coding: utf-8 -*-
from __future__ import division, print_function
import nltk
import os

DATA_DIR = "../data"

ftext = open(os.path.join(DATA_DIR, "text.tsv"), "rb")
num_docs, num_sents, num_words = 0, 0, 0
max_words, max_sents = 0, 0
for line in ftext:
    docid, text = line.strip().split("\t")
    num_docs += 1
    sents = nltk.sent_tokenize(text)
    if max_sents < len(sents):
        max_sents = len(sents)
    for sent in sents:
        num_sents += 1
        words = nltk.word_tokenize(sent)
        if max_words < len(words):
            max_words = len(words)
        for word in words:
            num_words += 1

ftext.close()

print("Number of documents: {:d}".format(num_docs))
print("Average #-sentences / doc: {:.3f}".format(num_sents / num_docs))
print("Maximum #-sentences / doc: {:d}".format(max_sents))
print("Average #-words / doc: {:.3f}".format(num_words / num_sents))
print("Maximum #-words / sent: {:d}".format(max_words))
