"""
preprocessing utility
"""
import re
import numpy as np

def tokenize(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  string = re.sub(r"[^A-Za-z0-9()$,!?\'\`]", " ", string) # add $ to indicate special token
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def normalize(string):
    # replace numbers with specific token _num_
    num_token = "_num_"
    seq = string.split()
    new_seq = []
    for word in seq:
        if (word.isdigit()):
            new_seq.append(num_token)
        else:
            new_seq.append(word)
    return " ".join(new_seq)


def preprocessText(fn, proc_fn):
    g = open(proc_fn, "w")
    with open(fn, "r") as f:
        for line in f:
            tok_line = tokenize(line)
            proc_line = normalize(tok_line)
            g.write(proc_line+"\n")
    g.close()
    print("process {} and save to {}".format(fn, proc_fn))
    
    
  


    
