"""
semantic discriminator
"""

from gensim.models import KeyedVectors
import numpy as np


class SemanticDiscriminator(object):
    
    def __init__(self, embed_fn):
        self.model = KeyedVectors.load_word2vec_format(embed_fn, binary=False, limit=10000)

    def getSemanticReward(self, orig_words, tsf_words):
        reward_list = []
        for (os, ts) in zip(orig_words, tsf_words):
            dist = min(100, self.model.wmdistance(os, ts))
            reward = -1.0 * dist / float(len(os))
            reward_list.append(reward)
        return np.array(reward_list)
    
    
    
        
        
        
        
        
