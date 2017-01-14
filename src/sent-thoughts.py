# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from keras.preprocessing import sequence
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import re

def maybe_build_sentences(text_filename, sent_filename):
    sents = []
    if os.path.exists(sent_filename):
        fsent = open(sent_filename, "rb")
        for line in fsent:
            docid, sent_id, sent = line.strip().split("\t")
            sents.append(sent)
        fsent.close()
    else:
        ftext = open(text_filename, "rb")
        fsent = open(sent_filename, "wb")
        for line in ftext:
            docid, text = line.strip().split("\t")
            sent_id = 1
            for sent in nltk.sent_tokenize(text):
                sents.append(sent)
                fsent.write("{:d}\t{:d}\t{:s}\n"
                    .format(int(docid), sent_id, sent))
                sent_id += 1
        fsent.close()
        ftext.close()
    return sents
    
def is_number(n):
    temp = re.sub("[\.,-/]", "", n)
    return temp.isdigit()

def generate_sentence_batch(sents, word2id, embeddings, 
                            max_seqlen, batch_size, embed_size):
    while True:
        # loop once per epoch
        # shuffle the input
        indices = np.random.permutation(np.arange(len(sents)))
        shuffled_sents = [sents[ix] for ix in indices]
        # convert to list of list of word id        
        sent_wids = [[word2id[word] for word in sent.split()]
                                    for sent in shuffled_sents]
        num_batches = len(shuffled_sents) // batch_size
        for bid in range(num_batches):
            # loop once per batch
            batch = np.zeros((batch_size, max_seqlen, embed_size))
            sents_batch = sent_wids[bid * batch_size : (bid + 1) * batch_size]
            sents_batch_padded = sequence.pad_sequences(sents_batch, max_seqlen)
            for i in range(batch_size):
                for j in range(max_seqlen):
                    try:
                        batch[i, j] = embeddings[sents_batch_padded[i][j]]
                    except KeyError:
                        pass
            yield batch, batch


############################ main ###############################

DATA_DIR = "../data"

TEXT_FILENAME = os.path.join(DATA_DIR, "text.tsv")
SENT_FILENAME = os.path.join(DATA_DIR, "sents.txt")

sentences = maybe_build_sentences(TEXT_FILENAME, SENT_FILENAME)

# build vocabulary
word_freqs = collections.Counter()
sent_lens = []
parsed_sentences = []
for sent in sentences:
    words = nltk.word_tokenize(sent)
    parsed_words = []
    for word in words:
        if is_number(word):
            word = "9"
        word_freqs[word.lower()] += 1
        parsed_words.append(word)
    sent_lens.append(len(words))
    parsed_sentences.append(" ".join(parsed_words))

sent_lens = np.array(sent_lens)
print("number of sentences: {:d}".format(len(sent_lens)))
print("distribution of sentence lengths (number of words)")
print("min:{:d}, max:{:d}, mean:{:.3f}, med:{:.3f}".format(
    np.min(sent_lens), np.max(sent_lens), np.mean(sent_lens),
    np.median(sent_lens)))
print("vocab size (full): {:d}".format(len(word_freqs)))

# number of sentences: 131545
# sentence length distribution: (1, 429, 22.315, 21.000
# vocab size (full): 50751

VOCAB_SIZE = 50000
EMBED_SIZE = 300
SEQUENCE_LEN = 50
#LATENT_SIZE = 1024
LATENT_SIZE = 512
#LATENT_SIZE = 2048
#LATENT_SIZE = 256

BATCH_SIZE = 64
NUM_EPOCHS = 100

# lookup tables
print("building lookup tables...")
word2id = collections.defaultdict(lambda: 1)
word2id["PAD"] = 0
word2id["UNK"] = 1
for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2id[k] = v + 2
id2word = {v:k for k, v in word2id.items()}

# load glove vectors
print("loading embeddings...")
embeddings = {}
fglove = open(os.path.join(DATA_DIR, "glove.6B.300d.txt"), "rb")
for line in fglove:
    cols = line.split()
    word = cols[0]
    if word2id.has_key(word):        
        vec = np.array([float(x) for x in cols[1:]])
        embeddings[word2id[word]] = vec
fglove.close()

# set up generators
print("setting up generators...")
test_size = int(0.3 * len(parsed_sentences))
s_train, s_test = parsed_sentences[0:-test_size], parsed_sentences[-test_size:]
train_gen = generate_sentence_batch(s_train, word2id, embeddings,
                                    SEQUENCE_LEN, BATCH_SIZE, EMBED_SIZE)
test_gen = generate_sentence_batch(s_test, word2id, embeddings,
                                   SEQUENCE_LEN, BATCH_SIZE, EMBED_SIZE)

# define autoencoder
print("defining autoencoder...")
inputs = Input(shape=(SEQUENCE_LEN, EMBED_SIZE), name="input")
encoded = LSTM(LATENT_SIZE, name="encoder_lstm")(inputs)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = LSTM(EMBED_SIZE, return_sequences=True, 
               name="decoder_lstm")(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# display autoencoder model summary
for layer in autoencoder.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

# train autoencoder
print("training autoencoder...")
num_train_samples = len(s_train) // BATCH_SIZE
num_test_samples = len(s_test) // BATCH_SIZE
checkpoint = ModelCheckpoint(filepath=os.path.join(
    DATA_DIR, "sent-thoughts-autoencoder.h5"),
    save_best_only=True)
history = autoencoder.fit_generator(train_gen, 
                                    samples_per_epoch=num_train_samples,
                                    nb_epoch=NUM_EPOCHS,
                                    validation_data=test_gen, 
                                    nb_val_samples=num_test_samples,
                                    callbacks=[checkpoint])

# saving history for charting (headless)
fchart = open(os.path.join(DATA_DIR, "sent-thoughts-loss.csv"), "wb")
trg_losses = history.history["loss"]
val_losses = history.history["val_loss"]
fchart.write("#loss\tval_loss\n")
for trg_loss, val_loss in zip(trg_losses, val_losses):
    fchart.write("{:.5f}\t{:.5f}\n".format(trg_loss, val_loss))
fchart.close()

#plt.title("Loss")
#plt.plot(history.history["loss"], color="g", label="train")
#plt.plot(history.history["val_loss"], color="b", label="validation")
#plt.legend(loc="best")
#plt.show()

