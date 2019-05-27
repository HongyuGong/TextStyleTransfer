"""
pretrain modules in style transfer
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
import params
import pickle
import sys


"""
create model
"""

def create_model(sess, save_folder, FLAGS, embed_fn):
    # load vocab & embeddings
    with open(save_folder+"vocab.pkl", "rb") as handle:
        vocab = pickle.load(handle)
    with open(save_folder+"tsf_vocab_inv.pkl", "rb") as handle:
        tsf_vocab_inv = pickle.load(handle)
    with open(save_folder+"init_embed.pkl", "rb") as handle:
        init_embed = pickle.load(handle)
    with open(save_folder+"tsf_init_embed.pkl", "rb") as handle:
        tsf_init_embed = pickle.load(handle)
    vocab_size = len(vocab)
    tsf_vocab_size = len(tsf_vocab_inv)
    print("Vocab size: {}, transfer vocab size: {}".format(vocab_size, tsf_vocab_size))
    
    # generator
    config_list = [(k, FLAGS[k].value) for k in FLAGS]
    generator_config = OrderedDict(sorted(config_list) + [("encoder_vocab_size", vocab_size), ("decoder_vocab_size", tsf_vocab_size)])
    #print("Generator config: {}, cell_type: {}".format(generator_config, "gru"))
    generator = Generator(generator_config, init_embed, tsf_init_embed)

    # language model
    lm_config_list = [(k, FLAGS[k].value) for k in FLAGS if k.startswith("lm_")] + [("batch_size", FLAGS.batch_size)]
    lm_config = OrderedDict(sorted(lm_config_list) + [("lm_vocab_size", vocab_size)])
    rnnlm = RNNLM(lm_config, init_embed)

    # style discriminator
    style_discriminator = StyleDiscriminator(FLAGS.style_num_classes, FLAGS.embedding_dim, \
                                             init_embed, FLAGS.style_hidden_size, \
                                             FLAGS.style_attention_size, FLAGS.max_sent_len, \
                                             FLAGS.style_keep_prob)

    # semantic discriminator
    semantic_discriminator = SemanticDiscriminator(embed_fn)

    # rollout
    rollout = ROLLOUT(vocab, tsf_vocab_inv)
    
    return generator, rnnlm, style_discriminator, semantic_discriminator, rollout, vocab, tsf_vocab_inv



"""
pretrain style discriminator
"""
def generatePretrainStyleSamples(orig_sents, orig_sent_len, tsf_sents, tsf_sent_len):
    sents = np.array(list(orig_sents) + list(tsf_sents))
    sent_len = np.array(list(orig_sent_len) + list(tsf_sent_len))
    labels = np.array([[0]] * len(orig_sents) + [[1]] * len(tsf_sents))
    train_size = len(sents)
    
    shuffled_indices = np.random.permutation(np.arange(train_size))
    shuffled_sents = sents[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]
    shuffled_sent_len = sent_len[shuffled_indices]
    return shuffled_sents, shuffled_labels, shuffled_sent_len


def pretrainStyleDiscriminator(sess, style_discriminator, orig_sents, orig_sent_len, tsf_sents, tsf_sent_len, \
                               style_epochs, batch_size):
    input_x, input_y, input_len = generatePretrainStyleSamples(orig_sents, orig_sent_len, tsf_sents, tsf_sent_len)
    discriminator_batches = data_helpers.batch_iter(list(zip(input_x, input_y, input_len)), batch_size, style_epochs)
    batch_count = 0
    for batch in discriminator_batches:
        batch_x, batch_y, batch_len = zip(*batch)
        feed = {style_discriminator.input_x: batch_x, style_discriminator.input_y: batch_y, \
                style_discriminator.sequence_length: batch_len}
        _ = sess.run(style_discriminator.train_op, feed_dict=feed)
        if (batch_count % 2000 == 0):
            feed = {style_discriminator.input_x: batch_x, style_discriminator.input_y: batch_y, \
                style_discriminator.sequence_length: batch_len}
            scores = sess.run(style_discriminator.scores, feed_dict=feed)
            gold_scores = [s[0] for s in batch_y]
            pred_scores = [s[0] for s in scores]
            fpr, tpr, _ = roc_curve(gold_scores, pred_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            print("Pretrain style discriminaotr - batch: {}, roc_auc: {}".format(batch_count, roc_auc))
        batch_count += 1


"""
pretrain RNNLM
"""
def generatePretrainRNNLMSamples(tsf_sents):
    flat_tsf_sents = np.reshape(tsf_sents, (1, -1))
    flat_outputs = np.copy(flat_tsf_sents)
    flat_outputs[:-1] = flat_tsf_sents[1:]
    flat_outputs[-1] = flat_tsf_sents[0]
    outputs = np.reshape(flat_outputs, np.shape(tsf_sents))
    return tsf_sents, outputs


def pretrainRNNLM(sess, rnnlm, tsf_sents, lm_epochs, lm_learning_rate, lm_decay_rate, lm_batch_size):
    input_x, input_y = generatePretrainRNNLMSamples(tsf_sents)
    for e in range(lm_epochs):
        sess.run(tf.assign(rnnlm.lr, lm_learning_rate * (lm_decay_rate ** e)))
        state =  sess.run(rnnlm.initial_state)
        rnnlm_batches = data_helpers.batch_iter(list(zip(input_x, input_y)), lm_batch_size, 1)
        batch_count = 0
        for batch in rnnlm_batches:
            batch_x, batch_y = zip(*batch)
            feed = {rnnlm.input_data: batch_x, rnnlm.targets: batch_y, \
                    rnnlm.initial_state: state}
            train_loss, state, _, _ = sess.run([rnnlm.cost, rnnlm.final_state, \
                                                rnnlm.train_op, rnnlm.inc_batch_pointer_op], feed)
            if (batch_count  % 600 == 0):
                print("Epoch: {}, batch: {}, LM loss: {}".format(e, batch_count, train_loss))
            batch_count += 1


"""
pretrain generator
"""
def generatePretrainGeneratorSamples(encoder_sents, encoder_sent_len, decoder_sents, decoder_sent_len):
    train_size = len(encoder_sents)
    shuffled_indices = np.random.permutation(np.arange(train_size))
    shuffled_encoder_sents = np.array(encoder_sents)[shuffled_indices]
    shuffled_encoder_sent_len = np.array(encoder_sent_len)[shuffled_indices]
    shuffled_decoder_sents = np.array(decoder_sents)[shuffled_indices]
    shuffled_decoder_sent_len = np.array(decoder_sent_len)[shuffled_indices]
    return shuffled_encoder_sents, shuffled_encoder_sent_len, shuffled_decoder_sents, shuffled_decoder_sent_len           


def pretrainGenerator(sess, generator, tsf_encoder_sents, tsf_encoder_sent_len, tsf_decoder_sents, tsf_decoder_sent_len, \
                      dev_orig_words, dev_orig_sents, dev_orig_sent_len, tsf_vocab_inv, \
                      saver, rnnlm, style_discriminator, semantic_discriminator, rollout, \
                      max_sent_len, epochs, batch_size, model_save_path, verbose=True):
    input_x, input_x_len, input_y, input_y_len = generatePretrainGeneratorSamples(tsf_encoder_sents, tsf_encoder_sent_len, \
                                                                                   tsf_decoder_sents, tsf_decoder_sent_len)
    generator_batches = data_helpers.batch_iter(list(zip(input_x, input_x_len, input_y, input_y_len)), batch_size, epochs)
    best_dev_reward = float("-inf")
    dev_size = len(dev_orig_sents)
    batch_count = 0
    for batch in generator_batches:
        batch_x, batch_x_len, batch_y, batch_y_len = zip(*batch)
        batch_x = np.array(list(batch_x))
        batch_x_len = np.array(list(batch_x_len))
        batch_y = np.array(list(batch_y))
        batch_y_len = np.array(list(batch_y_len))
        feed = {generator.encoder_inputs: batch_x, \
                generator.encoder_inputs_length: batch_x_len, \
                generator.decoder_inputs: batch_y, \
                generator.decoder_inputs_length: batch_y_len}
        _ = sess.run(generator.pretrain_g_updates, feed_dict=feed)
        if (batch_count % 500 == 0):
            dev_rewards = []
            dev_style_rewards = []
            dev_sem_rewards = []
            dev_lm_rewards = []
            for itera in range(int(dev_size/batch_size)):
                start_ind = itera*batch_size
                batch_orig_words = dev_orig_words[start_ind:start_ind+batch_size]
                batch_orig_sents = dev_orig_sents[start_ind:start_ind+batch_size]
                batch_orig_sent_len = dev_orig_sent_len[start_ind:start_ind+batch_size]
                # beam_search outputs
                batch_generator_outputs = generator.generate(sess, batch_orig_sents, batch_orig_sent_len)
                # mostly likely one from beam search
                batch_generator_outputs = np.array(batch_generator_outputs)[:,:,0]
                batch_generator_outputs, batch_outputs_len = data_helpers.cleanGeneratorOutputs(batch_generator_outputs, max_sent_len)
                batch_style_reward, batch_sem_reward, batch_lm_reward, batch_reward = \
                                    rollout.get_sent_reward(sess, batch_size, batch_orig_words, \
                                                            batch_generator_outputs, batch_outputs_len, \
                                                            rnnlm, style_discriminator, semantic_discriminator, False)
                # batch_reward: scalar
                dev_rewards.append(batch_reward)
                dev_style_rewards.append(batch_style_reward)
                dev_sem_rewards.append(batch_sem_reward)
                dev_lm_rewards.append(batch_lm_reward)
            avg_dev_reward = np.mean(dev_rewards)
            avg_dev_style_reward = np.mean(dev_style_rewards)
            avg_dev_sem_reward = np.mean(dev_sem_rewards)
            avg_dev_lm_reward = np.mean(dev_lm_rewards)
            print("dev_size: {}, style_reward: {}, sem_reward: {}, lm_reward: {}, dev reward: {}".format(len(dev_rewards)*batch_size, \
                                                                                                         avg_dev_style_reward, avg_dev_sem_reward, \
                                                                                                         avg_dev_lm_reward, avg_dev_reward))
                                                                                                         
            # save best model
            if (avg_dev_style_reward >= 0.7 and avg_dev_sem_reward > best_dev_reward):
                best_dev_reward = avg_dev_sem_reward
                print("best dev reward: {}".format(best_dev_reward))
                saver.save(sess, model_save_path)               

            if (verbose):
                batch_tsf_words = data_helpers.convertIdxToWords(batch_generator_outputs, tsf_vocab_inv)
                verbose_size = min(2, batch_size)
                print("batch_count:", batch_count)
                for i in range(verbose_size):
                    print("orig sent: "+" ".join(batch_orig_words[i]))
                    print("tsf sent: "+" ".join(batch_tsf_words[i]))
        batch_count += 1



def generateStyleDiscriminatorSamples(neg_samples, tsf_sents, tsf_sent_len, vocab, tsf_vocab_inv, max_sent_len):
    # return both positive and negative sampels for discriminator training
    neg_num = len(neg_samples)
    pos_num = neg_num
    pos_indices = np.random.choice(len(tsf_sents),size=neg_num, replace=False)
    pos_samples = np.array(tsf_sents)[pos_indices]
    pos_len = np.array(tsf_sent_len)[pos_indices]

    # neg_samples: encoder indices
    neg_encoder_samples = [[vocab[tsf_vocab_inv[ind]] for ind in word_inds] for word_inds in neg_samples]
    samples = list(pos_samples) + list(neg_encoder_samples)
    labels = [[1]] * pos_num + [[0]] * neg_num
    sent_len = list(pos_len) + [max_sent_len] * neg_num
    print("samples: {}, labels: {}, sent_len: {}".format(len(samples), len(labels), len(sent_len)))
    return samples, labels, sent_len

