"""
RL based style transfer
"""

import numpy as np
import tensorflow as tf
import random
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
import data_helpers
from generator import Generator
from rnnlm import RNNLM
from style_discriminator import StyleDiscriminator
from semantic_discriminator import SemanticDiscriminator
from rollout import ROLLOUT
import pretrain
import params
import pickle
import sys
import os


# Set Parameters
# Language model
tf.app.flags.DEFINE_integer('lm_rnn_size', 50, 'same as embedding dim')
tf.app.flags.DEFINE_integer('lm_num_layers', 2, 'number of layers in language modeling')
tf.app.flags.DEFINE_string('lm_model', 'gru', 'neuron types in lm')
tf.app.flags.DEFINE_integer('lm_seq_length', 30, 'sequence length')
tf.app.flags.DEFINE_integer('lm_epochs', 4, 'epochs in training lm')
tf.app.flags.DEFINE_float('lm_grad_clip', 5.0, 'clip gradients at this value')
tf.app.flags.DEFINE_float('lm_learning_rate', 1e-5, 'learning rate in lm')
tf.app.flags.DEFINE_float('lm_decay_rate', 0.97, 'decay rate for rmsprop')
# Generator
tf.app.flags.DEFINE_string('cell_type', 'GRU', "encoder-decoder cell")
tf.app.flags.DEFINE_integer('hidden_units', 50, "dimension of hidden cell")
tf.app.flags.DEFINE_integer('depth', 1, "the depth of LSTM")
tf.app.flags.DEFINE_string('attention_type', 'Loung', "attention type")
tf.app.flags.DEFINE_integer('embedding_dim', 100, "dimension of embedding")
tf.app.flags.DEFINE_integer('max_sent_len', 18, "max length of sentence")
tf.app.flags.DEFINE_boolean('attn_input_feeding', False, "")
tf.app.flags.DEFINE_boolean('use_dropout', True, "")
tf.app.flags.DEFINE_float('dropout_rate', 0.1, "")
tf.app.flags.DEFINE_string('optimizer', 'adam', "")
tf.app.flags.DEFINE_float('learning_rate', 1e-4, "")
tf.app.flags.DEFINE_float('rl_learning_rate', 1e-5, "")
tf.app.flags.DEFINE_boolean('use_beamsearch_decode', False, "")
tf.app.flags.DEFINE_integer('beam_width', 4, "")
tf.app.flags.DEFINE_float('grad_clip', 5.0, "")
# Style Discriminator
tf.app.flags.DEFINE_integer('style_num_classes', 1, "")
tf.app.flags.DEFINE_integer('style_hidden_size', 50, "")
tf.app.flags.DEFINE_integer('style_attention_size', 20, "")
tf.app.flags.DEFINE_float('style_keep_prob', 0.9, "")
# Train
tf.app.flags.DEFINE_string('data_type', 'yelp', 'data type: either yelp or gyafc_family')
tf.app.flags.DEFINE_string("model_path", None, "path of trained model")
tf.app.flags.DEFINE_string("output_path", None, "path of transferred sentences")
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')


print("\nParameters:")
FLAGS = tf.app.flags.FLAGS
print(FLAGS.__flags.items())



def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    generator, rnnlm, style_discriminator, semantic_discriminator, rollout, vocab, tsf_vocab_inv = \
               pretrain.create_model(sess, save_folder, FLAGS, embed_fn)
    saver = tf.train.Saver(tf.all_variables())

    MODEL = FLAGS.model_path
    try:
        saver.restore(sess, MODEL)
    except:
        print("Error: No model found in {}".format(MODEL))
        sys.exit(0)

    # load test data
    test_orig_sents, test_orig_words, test_orig_len = data_helpers.loadTestInputs(FLAGS.max_sent_len, save_folder)
    print("test size: {}".format(len(test_orig_sents)))

    dump_folder = "../dump/" + str(FLAGS.data_type) + "/"
    output_path = dump_folder + FLAGS.output_path
    log_path = dump_folder + "logs.txt"
    f = open(output_path, "w")
    g = open(log_path, "w")
    ind = 0
    while (ind < len(test_orig_sents)):
        input_sents = test_orig_sents[ind:ind+FLAGS.batch_size]
        input_len = test_orig_len[ind:ind+FLAGS.batch_size]
        orig_words = test_orig_words[ind:ind+FLAGS.batch_size]
        # pad to batch size
        if (ind+FLAGS.batch_size > len(test_orig_sents)):
            input_sents = test_orig_sents[ind:] + [test_orig_sents[ind]] * (ind+FLAGS.batch_size-len(test_orig_sents))
            input_len = test_orig_len[ind:]+[test_orig_len[ind]]*(ind+FLAGS.batch_size-len(test_orig_sents))
            orig_words = test_orig_words[ind:] + [test_orig_words[ind]] * (ind+FLAGS.batch_size-len(test_orig_sents))
        # generator_outputs: [batch_size, time, beam_width]
        beam_generator_outputs = generator.generate(sess, input_sents, input_len)
        # generator_outputs: [batch_size, time]
        generator_outputs = np.array(beam_generator_outputs)[:,:,0]
        generator_outputs = data_helpers.cleanTexts(generator_outputs, FLAGS.max_sent_len)
        tsf_words = data_helpers.convertIdxToWords(generator_outputs, tsf_vocab_inv)
        tmp_ind = ind
        ind += FLAGS.batch_size
        for (orig_word_seq, tsf_word_seq) in zip(orig_words, tsf_words):
            if (tmp_ind >= len(test_orig_sents)):
                break
            # output
            f.write(" ".join(tsf_word_seq)+"\n")
            # log
            g.write("orig:\t"+" ".join(orig_word_seq)+"\n")
            g.write("tsf:\t"+" ".join(tsf_word_seq)+"\n")
            g.write("\n")
            tmp_ind += 1
    f.close()
    g.close()
    print("done saving tsf sents to", output_path)
    print("done saving both orig and tsf logs to", log_path)
    


if __name__ == "__main__":
    save_folder = "../dump/" + FLAGS.data_type + "/"
    embed_fn = save_folder + "tune_vec.txt"
    tf.app.run()
            
        
        
                
                
            
        
        

    
