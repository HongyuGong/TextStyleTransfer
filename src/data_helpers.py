"""
data_helpers.py
"""
import pickle
import numpy as np
from params import *
from collections import Counter
import itertools
import params
import argparse
import sys
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
from preprocess import preprocessText


PADDING = "<EOS>"
UNK = "<UNK>"


# called by train_gan_general.py
def padSents(word_sents, max_len, padding_word=PADDING):
    len_list = np.array([len(sent) for sent in word_sents])
    print("max len of all sentences:", max(len_list))
    # normalize length
    len_list = [min(max_len, l) for l in len_list]
    print("max len is set as:", max_len)
    padded_word_sents = []
    for i in range(len(word_sents)):
        sent = word_sents[i][:max_len]
        num_padding = max_len - len(sent)
        new_sent = sent + [padding_word] * num_padding
        padded_word_sents.append(new_sent[:])
    return padded_word_sents, len_list


def digitalizeDataUsingSingleVocab(words, vocab, max_len):
    # func: convert word sequence to index sequence with a single vocab
    # padding
    padded_words, sent_len = padSents(words, max_len)
    # allow unk token
    sents = []
    for word_seq in padded_words:
        sent = []
        for word in word_seq:
            try:
                token = vocab[word]
            except:
                token = params.unk_token
            sent.append(token)
        sents.append(sent[:])
    return sents, sent_len


def loadTrainInputs(max_sent_len, save_folder):
    """
    return sequence of word indices
    """
    train_orig_fn = save_folder+"train_orig.pkl"
    train_tsf_fn = save_folder+"train_tsf.pkl"
    # load vocab (joint_vocab) & tsf_vocab (decoder_vocab)
    with open(save_folder+"vocab.pkl", "rb") as handle:
        vocab = pickle.load(handle)
    with open(save_folder+"tsf_vocab.pkl", "rb") as handle:
        tsf_vocab = pickle.load(handle)
    # read words for encoder inputs
    with open(train_orig_fn, "rb") as handle:
        train_orig_words = pickle.load(handle)
    with open(train_tsf_fn, "rb") as handle:
        train_tsf_words = pickle.load(handle)
    train_orig_sents, train_orig_len = digitalizeDataUsingSingleVocab(train_orig_words, vocab, max_sent_len)
    train_tsf_encoder_sents, train_tsf_encoder_len = digitalizeDataUsingSingleVocab(train_tsf_words, vocab, max_sent_len)
    train_tsf_decoder_sents, train_tsf_decoder_len = digitalizeDataUsingSingleVocab(train_tsf_words, tsf_vocab, max_sent_len)
    return train_orig_sents, train_orig_words, train_orig_len, train_tsf_encoder_sents, train_tsf_encoder_len, \
           train_tsf_decoder_sents, train_tsf_decoder_len


def loadDevInputs(max_sent_len, save_folder):
    orig_fn = save_folder + "dev_orig.pkl"
    with open(save_folder+"vocab.pkl", "rb") as handle:
        vocab = pickle.load(handle)
    with open(orig_fn, "rb") as handle:
        orig_words = pickle.load(handle)

    orig_sents, orig_len = digitalizeDataUsingSingleVocab(orig_words, vocab, max_sent_len)
    return orig_sents, orig_words, orig_len


def loadTestInputs(max_sent_len, save_folder):
    orig_fn = save_folder + "test_orig.pkl"
    with open(save_folder+"vocab.pkl", "rb") as handle:
        vocab = pickle.load(handle)
    with open(orig_fn, "rb") as handle:
        orig_words = pickle.load(handle)

    orig_sents, orig_len = digitalizeDataUsingSingleVocab(orig_words, vocab, max_sent_len)
    return orig_sents, orig_words, orig_len


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    print("data size:", data_size)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            gap = data_size - (batch_num + 1) * batch_size
            start_index = batch_num * batch_size
            if (gap < 0):
                start_index += gap
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def convertIdxToWords(sents, vocab_inv):
    words = [[vocab_inv[idx] for idx in seq] for seq in sents]
    return words    


def cleanGeneratorOutputs(generator_outputs, max_sent_len):
    clean_generator_outputs = []
    outputs_len = []
    eos = params.end_token
    for sent in generator_outputs:
        sent = list(sent)
        if (len(sent) >= max_sent_len):
            clean_sent = sent[:max_sent_len]
        else:
            clean_sent = sent[:] + [eos] * (max_sent_len - len(sent))
        # replace end tokens with all eos
        end_ind = max_sent_len
        if (eos in clean_sent):
            end_ind = clean_sent.index(eos)
            for i in range(end_ind, max_sent_len):
                clean_sent[i] = eos
        clean_generator_outputs.append(clean_sent[:])
        outputs_len.append(end_ind)
    return np.array(clean_generator_outputs), np.array(outputs_len)


def cleanTexts(generator_outputs, max_sent_len):
    clean_generator_outputs = []
    eos = params.end_token
    for sent in generator_outputs:
        # padding or removal
        sent = list(sent)
        clean_sent = sent[:max_sent_len]
        end_ind = max_sent_len
        if eos in clean_sent:
            end_ind = clean_sent.index(eos)
            clean_sent = clean_sent[:end_ind]
        clean_generator_outputs.append(clean_sent[:])
    return np.array(clean_generator_outputs)
    
    
