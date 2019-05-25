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
#from params import *
import params
import pickle
import sys
import pretrain
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
tf.app.flags.DEFINE_integer('beam_width', 2, "")
tf.app.flags.DEFINE_float('grad_clip', 5.0, "")
# Style Discriminator
tf.app.flags.DEFINE_integer('style_num_classes', 1, "")
tf.app.flags.DEFINE_integer('style_hidden_size', 50, "")
tf.app.flags.DEFINE_integer('style_attention_size', 20, "")
tf.app.flags.DEFINE_float('style_keep_prob', 0.9, "")
tf.app.flags.DEFINE_integer('style_epochs', 3, 'epochs in pretraining style discriminator')
# Train
tf.app.flags.DEFINE_string('data_type', 'yelp', 'data type: either yelp or gyafc_family')
tf.app.flags.DEFINE_boolean('use_pretrained_model', False, 'load pretrained model')
tf.app.flags.DEFINE_string("pretrained_model_path", None, "path of pretrained model")
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_integer('pretrain_epochs', 4, 'number of pre-training epoches')
tf.app.flags.DEFINE_integer('epochs', 2, 'number of training epoches')
tf.app.flags.DEFINE_integer('rollout_num', 2, 'iterations of sampled rollouts')


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

    # create pretrained_model folder
    pretrained_model_folder = "../pretrained_model/" + FLAGS.data_type + "/"
    if not os.path.exists(pretrained_model_folder):
        os.mkdir(pretrained_model_folder)
        
    # create model folder
    model_folder = "../model/" + FLAGS.data_type + "/"
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    

    # load data
    #orig_sents, orig_words, orig_sent_len, tsf_sents, tsf_sent_len = data_helpers.loadTrainInputs(save_folder)
    # load train data
    orig_sents, orig_words, orig_sent_len, tsf_encoder_sents, tsf_encoder_sent_len, \
                tsf_decoder_sents, tsf_decoder_sent_len = data_helpers.loadTrainInputs(FLAGS.max_sent_len, save_folder)
    print("# of train orig_sents: {}".format(len(orig_sents)))
    print("example train tsf_encoder_sents: {}".format(tsf_encoder_sents[0]))
    print("example train tsf_decoder_sents: {}".format(tsf_decoder_sents[0]))
    
    # load dev data
    dev_orig_sents, dev_orig_words, dev_orig_sent_len = data_helpers.loadDevInputs(FLAGS.max_sent_len, save_folder)
    print("# of dev orig_sents: {}".format(len(dev_orig_sents)))
    
    
    PRETRAINED_MODEL = pretrained_model_folder + FLAGS.pretrained_model_path
    if (FLAGS.use_pretrained_model):
        saver.restore(sess, PRETRAINED_MODEL)
        print("restore an existing model...")
    else:
        print("pretrain a new model...")
        sess.run(tf.global_variables_initializer())

        
        # pretrain rnnlm
        print("pretrain rnnlm...")
        # Better change: tsf_decoder_sents as input -> also change init embed in training RNNLM
        pretrain.pretrainRNNLM(sess, rnnlm, tsf_encoder_sents, FLAGS.lm_epochs, FLAGS.lm_learning_rate, \
                      FLAGS.lm_decay_rate, FLAGS.batch_size)
        
        # pretrain discriminator
        print("pretrain discriminator...")
        pretrain.pretrainStyleDiscriminator(sess, style_discriminator, orig_sents, orig_sent_len, \
                                   tsf_encoder_sents, tsf_encoder_sent_len, \
                                   FLAGS.style_epochs, FLAGS.batch_size)
        

        print("pretrain generator...")
        pretrain.pretrainGenerator(sess, generator, tsf_encoder_sents, tsf_encoder_sent_len, \
                                   tsf_decoder_sents, tsf_decoder_sent_len, \
                                   dev_orig_words, dev_orig_sents, dev_orig_sent_len, \
                                   tsf_vocab_inv, saver, rnnlm, style_discriminator, semantic_discriminator, rollout, \
                                   FLAGS.max_sent_len, FLAGS.pretrain_epochs, FLAGS.batch_size, PRETRAINED_MODEL)
        sys.exit(0)


    print("adversarial training...")
    best_dev_reward = float("-inf")
    # generator batches
    generator_batches = data_helpers.batch_iter(list(zip(orig_sents, orig_words, orig_sent_len)), FLAGS.batch_size, FLAGS.epochs)
    batch_count = 0
    dev_size = len(dev_orig_sents)
    for batch in generator_batches:
        # load data
        batch_orig_sents, batch_orig_words, batch_orig_sent_len = zip(*batch)
        # beam outputs
        generator_outputs = generator.generate(sess, batch_orig_sents, batch_orig_sent_len)
        # most likely outputs
        generator_outputs = np.array(generator_outputs)[:,:,0]
        generator_outputs, outputs_len = data_helpers.cleanGeneratorOutputs(generator_outputs, FLAGS.max_sent_len)
        rewards = rollout.get_reward(sess, generator, batch_orig_sents, batch_orig_words, batch_orig_sent_len, \
                                     generator_outputs, rnnlm, style_discriminator, semantic_discriminator, \
                                     FLAGS.max_sent_len, FLAGS.rollout_num)            

        # train generator
        feed = {generator.encoder_inputs: batch_orig_sents, \
                generator.encoder_inputs_length: batch_orig_sent_len, \
                generator.decoder_inputs: generator_outputs, \
                generator.decoder_inputs_length: outputs_len, \
                generator.rewards: 1.0 * rewards} # +/-rewards
        _ = sess.run(generator.g_updates, feed_dict=feed)
        avg_reward = np.mean(rewards)

        # evaluate on dev data
        if (batch_count % 100 == 0):
            dev_rewards = []
            dev_style_rewards = []
            dev_sem_rewards = []
            dev_lm_rewards = []
            for itera in range(int(dev_size/FLAGS.batch_size)):
                start_ind = itera*FLAGS.batch_size
                end_ind = start_ind + FLAGS.batch_size
                batch_orig_words = dev_orig_words[start_ind:end_ind]
                batch_orig_sents = dev_orig_sents[start_ind:end_ind]
                batch_orig_len = dev_orig_sent_len[start_ind:end_ind]
                # beam search
                batch_generator_outputs = generator.generate(sess, batch_orig_sents, batch_orig_len)
                # most likely sequence
                batch_generator_outputs = np.array(batch_generator_outputs)[:,:,0]
                batch_generator_outputs, batch_outputs_len = data_helpers.cleanGeneratorOutputs(batch_generator_outputs, FLAGS.max_sent_len)
                batch_style_reward, batch_sem_reward, batch_lm_reward, batch_reward = \
                                    rollout.get_sent_reward(sess, FLAGS.batch_size, batch_orig_words, \
                                                            batch_generator_outputs, batch_outputs_len, \
                                                            rnnlm, style_discriminator, semantic_discriminator, False)
                dev_rewards.append(batch_reward)
                dev_style_rewards.append(batch_style_reward)
                dev_sem_rewards.append(batch_sem_reward)
                dev_lm_rewards.append(batch_lm_reward)
            avg_dev_reward = np.mean(dev_rewards)
            avg_dev_style_reward = np.mean(dev_style_rewards)
            avg_dev_sem_reward = np.mean(dev_sem_rewards)
            avg_dev_lm_reward = np.mean(dev_lm_rewards)
            print("batch_count: {}, style_reward: {}, sem_reward: {}, lm_reward: {}, dev reward: {}".format(batch_count, \
                                                                                                            avg_dev_style_reward, avg_dev_sem_reward, \
                                                                                                            avg_dev_lm_reward, avg_dev_reward))
            if avg_dev_style_reward >= 0.82 and avg_dev_sem_reward > best_dev_reward:
                best_dev_reward = avg_dev_sem_reward
                print("best dev sem reward: {}".format(best_dev_reward))
                checkpoint_prefix = model_folder + "best_model"
                saver.save(sess, checkpoint_prefix)

        # train style discriminator
        if (batch_count % 300 == 0): 
            input_x, input_y, input_len = pretrain.generateStyleDiscriminatorSamples(generator_outputs, \
                                                                                     tsf_encoder_sents, tsf_encoder_sent_len, vocab, tsf_vocab_inv, \
                                                                                     FLAGS.max_sent_len)
            feed = {style_discriminator.input_x: input_x, \
                    style_discriminator.input_y: input_y, \
                    style_discriminator.sequence_length: input_len}
            _ = sess.run(style_discriminator.train_op, feed_dict=feed)

        batch_count += 1
        
    checkpoint_prefix = model_folder + "model"
    saver.save(sess, checkpoint_prefix)


if __name__ == "__main__":
    save_folder = "../dump/" + FLAGS.data_type + "/"
    embed_fn = save_folder + "tune_vec.txt"
    tf.app.run()
            
        
        
                
                
            
        
        

    
