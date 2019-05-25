"""
corpus_helper
1. prepare corpus to .pkl
2. prepare vocab and embedidng for joint train corpus and transfer train corpus
3. tune embedding
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


def shuffleData(sent_list):
    np.random.seed(31)
    shuffled_indices = np.random.permutation(np.arange(len(sent_list)))
    shuffled_sent_list = np.array(sent_list)[shuffled_indices]
    return shuffled_sent_list

    
def convertDataToPickle(fn, pickle_fn, is_shuffle=False):
    f = open(fn, "r")
    sents = []
    for line in f:
        seq = line.strip().split()
        sents.append(seq[:])
    f.close()
    # shuffle sents
    if (is_shuffle):
        sents = shuffleData(sents)
    print("fn size: {}".format(len(sents)))
    with open(pickle_fn, "wb") as handle:
        pickle.dump(sents, handle)
    print("done shuffling and transforming txt to pickle...")


# ************** Build Vocab & Init Embedding ****************

def buildVocab(sentences, min_cnt=2):
    # vocab: dict, vocab_inv: list
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocab_inv = extra_words + [x[0] for x in word_counts.most_common() if x[1] >= min_cnt]
    print("Extra words:", extra_words)
    # Mapping from word to index
    vocab = {x: i for i, x in enumerate(vocab_inv)}
    print("<EOS> in vocab:", vocab["<EOS>"])
    return [vocab, vocab_inv]


def buildEmbed(vocab_list, embed_fn, word_limit=250000):
    # sanity check: vocab_list should be a list instead of dictionary
    if (type(vocab_list) != type([1,2])):
        print("vocab type: {}, not a list!".format(type(vocab_list)))
        sys.exit(0)
    
    print("load pretrained word embeddings...")
    f = open(embed_fn, "r")
    header = f.readline()
    vocab_size, dim = np.array(header.strip().split(), "int")

    # read pretrained embedding
    all_vocab = []
    all_embed = []
    i = 0
    while (i < word_limit):
        line = f.readline()
        seq = line.strip().split()
        if (seq == []):
            break
        all_vocab.append(seq[0])
        vec = list(np.array(seq[1:], "float"))
        all_embed.append(vec[:])
        i += 1
    f.close()
    print("pretrain vocab:", all_vocab[:10])

    # adapt to dataset
    print("dataset vocabulary:", len(vocab_list))
    init_embed = []
    unknown_word = []
    for w in vocab_list:
        try:
            ind = all_vocab.index(w)
            vec = all_embed[ind]
        except:
            vec = (np.random.rand(dim) - 0.5) * 2
            unknown_word.append(w)
        init_embed.append(vec[:])
    print("unknown word:", len(unknown_word), unknown_word[:10])
    init_embed = np.array(init_embed)
    print("vocab size: {}, embedding size: {}".format(len(vocab_list), np.shape(init_embed)))
    return init_embed


def buildVocabEmbed(train_pkl_list, train_tsf_pkl, embed_fn, raw_vec_path, save_folder):
    all_sents = []
    for pkl in train_pkl_list:
        with open(pkl, "rb") as handle:
            sents = pickle.load(handle)
        all_sents += list(sents)
    # joint vocab of orig and tsf data, vocab (dictionary), vocab_inv (list)
    vocab, vocab_inv = buildVocab(all_sents)
    with open(save_folder+"vocab.pkl", "wb") as handle:
        pickle.dump(vocab, handle)
    with open(save_folder+"vocab_inv.pkl", "wb") as handle:
        pickle.dump(vocab_inv, handle)
    print("Joint vocab size:", len(vocab))
    print("Example vocab words:", vocab_inv[:10])

    
    # joint word embedding for train corpus
    init_embed = buildEmbed(vocab_inv, embed_fn, 40000)
    # save raw_dataset_vec.txt for tuning
    f =  open(raw_vec_path, "w")
    vocab_size = len(vocab)
    f.write(str(vocab_size)+" "+str(VEC_DIM)+"\n")
    for (word, vec) in zip(vocab_inv, init_embed):
        f.write(word+" "+" ".join([str(val) for val in vec])+"\n")
    f.close()
    print("saving raw vecs for tuning...")
    del all_sents, vocab, vocab_inv

    # tsf_vocab, tsf_vocab_inv
    with open(train_tsf_pkl, "rb") as handle:
        train_tsf_sents = pickle.load(handle)
    train_tsf_sents = list(train_tsf_sents)
    tsf_vocab, tsf_vocab_inv = buildVocab(train_tsf_sents)
    with open(save_folder+"tsf_vocab.pkl", "wb") as handle:
        pickle.dump(tsf_vocab, handle)
    with open(save_folder+"tsf_vocab_inv.pkl", "wb") as handle:
        pickle.dump(tsf_vocab_inv, handle)
    print("Transfer vocab size:", len(tsf_vocab))
    print("Examples tsf vocab:", tsf_vocab_inv[:10])


# ************** Tune Embedding ****************

def combineTrainCorpus(orig_txt, tsf_txt, save_path=None):
    sents = []
    orig_f = open(orig_txt, "r")
    sents += [line.strip() for line in orig_f.readlines()]
    orig_f.close()
    tsf_f = open(tsf_txt, "r")
    sents += [line.strip() for line in tsf_f.readlines()]
    tsf_f.close()
    shuffled_sents = shuffleData(sents)
    g = open(save_path, "w")
    for sent in shuffled_sents:
        g.write(sent+"\n")
    g.close()
    print("Saving train corpus to {}".format(save_path))
    return len(sents)
        
    
def tuneEmbed(train_corpus, total_lines, raw_vec_path, tune_vec_path):
    sentences = LineSentence(train_corpus)
    model = Word2Vec(sentences, size=VEC_DIM, window=6, iter=20, workers=10, min_count=1)
    model.intersect_word2vec_format(raw_vec_path, lockf=1.0, binary=False)
    # measure runing time
    start = time.time()
    model.train(sentences, total_examples=total_lines, epochs=20)
    end = time.time()
    print("done retraining using time {} s.".format(end-start))
    #word_vectors = model.mv
    model.wv.save_word2vec_format(tune_vec_path)
    print("done saving tuned word embeddings...")
    

def saveTuneEmbed(save_folder, tune_vec_path):
    with open(save_folder+"vocab_inv.pkl", "rb") as handle:
        vocab_list = pickle.load(handle)
    init_embed = buildEmbed(vocab_list, tune_vec_path, word_limit=250000)
    with open(save_folder+"init_embed.pkl", "wb") as handle:
        pickle.dump(init_embed, handle)
    print("saving init_embed shape: {}".format(np.shape(init_embed)))
    del vocab_list, init_embed
    
    with open(save_folder+"tsf_vocab_inv.pkl", "rb") as handle:
        tsf_vocab_list = pickle.load(handle)
    tsf_init_embed = buildEmbed(tsf_vocab_list, tune_vec_path, word_limit=250000)
    with open(save_folder+"tsf_init_embed.pkl", "wb") as handle:
        pickle.dump(tsf_init_embed, handle)
    print("saving tsf_init_embed shape: {}".format(np.shape(tsf_init_embed)))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="gyafc_family") #gyafc_family, yelp
    parser.add_argument("--tokenize", default=False, action="store_true")
    parser.add_argument("--vec_dim", type=int, default=100)
    parser.add_argument("--embed_fn", type=str, default=None)
    args = parser.parse_args()
    data_type = args.data_type
    tok_flag = args.tokenize
    VEC_DIM = args.vec_dim
    embed_fn = args.embed_fn

    data_folder = "../data/"+str(data_type)+"/"
    dump_folder = "../dump/"+str(data_type)+"/"
    if not os.path.exists("../data/"):
        os.mkdir("../data/")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if not os.path.exists("../dump/"):
        os.mkdir("../dump/")
    if not os.path.exists(dump_folder):
        os.mkdir(dump_folder)
        
    fn_names = ["corpus.train.orig", "corpus.train.tsf", \
                "corpus.dev.orig", "corpus.dev.tsf", \
                "corpus.test.orig", "corpus.test.tsf"]

    if tok_flag:
        print("tokenization...")
        raw_fn_list = [data_folder + fn for fn in fn_names]
        fn_list = [dump_folder + fn for fn in fn_names]
        # tokenization + normalization
        for raw_fn, proc_fn in zip(raw_fn_list, fn_list):
            preprocessText(raw_fn, proc_fn)
    else:
        fn_list = [data_folder + fn for fn in fn_names]

    print("saving corpus data to pkl...")
    pkl_names=  ["train_orig.pkl", "train_tsf.pkl", \
                 "dev_orig.pkl", "dev_tsf.pkl", \
                 "test_orig.pkl", "test_tsf.pkl"]
    pkl_list = [dump_folder + pkl for pkl in pkl_names]
    for fn, pkl in zip(fn_list, pkl_list):
        is_shuffle = ("train" in fn)
        # convert .txt to .pkl
        convertDataToPickle(fn, pkl, is_shuffle=is_shuffle)


    print("build vocabulary and embedding...")
    raw_vec_path = dump_folder + "raw_vec.txt"
    train_tsf_pkl = pkl_list[1]
    buildVocabEmbed(pkl_list[:2], train_tsf_pkl, embed_fn, raw_vec_path, dump_folder)
    

    # combine and shuffle corpus for embedding tuning
    print("tune embedding on the dataset...")
    train_orig_txt, train_tsf_txt = fn_list[:2]
    train_txt = dump_folder + "corpus.train"
    total_lines = combineTrainCorpus(train_orig_txt, train_tsf_txt, train_txt)
    tune_vec_path = dump_folder + "tune_vec.txt"
    tuneEmbed(train_txt, total_lines, raw_vec_path, tune_vec_path)
    # save embedding as init_embed & tsf_init_embed
    saveTuneEmbed(dump_folder, tune_vec_path)
    
    os.system("rm {}".format(raw_vec_path))
    os.system("rm {}".format(train_txt))
    
