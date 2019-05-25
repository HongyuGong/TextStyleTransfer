"""
roll out for rewards estimation
"""
import tensorflow as tf
import numpy as np
from params import *
import sys

class ROLLOUT(object):
    def __init__(self, vocab, tsf_vocab_inv):
        self.name = "rollout"
        self.vocab = vocab
        self.tsf_vocab_inv = tsf_vocab_inv
        self.gamma = 0.95

    def get_sent_reward(self,  sess, batch_size, orig_words, tsf_dec_sents, tsf_sent_len, \
                        rnnlm, style_discriminator, semantic_discriminator, verbose=False):
        sent_size = len(orig_words)
        # tsf_sents -> raw_tsf_words (with <EOS>)
        raw_tsf_words = [[self.tsf_vocab_inv[ind] for ind in dec_sent] for dec_sent in tsf_dec_sents]
        # tsf_words (without <EOS>)
        tsf_words = [raw_tsf_words[ind][:tsf_sent_len[ind]] for ind in range(sent_size)]
        # raw_tsf_sents -> tsf_enc_sents
        tsf_enc_sents = [[self.vocab[word] for word in words] for words in raw_tsf_words]
        
        # evaluation       
        style_rewards = []
        lm_rewards = []
        semantic_rewards = []
        start_ind = 0
        while (start_ind < sent_size):
            end_ind = start_ind + batch_size
            batch_orig_words = orig_words[start_ind:end_ind]
            batch_tsf_words = tsf_words[start_ind:end_ind]
            batch_tsf_enc_sents = tsf_enc_sents[start_ind:end_ind]
            batch_tsf_len = tsf_sent_len[start_ind:end_ind]
            if (end_ind > sent_size):
                batch_tsf_enc_sents = batch_tsf_enc_sents + tsf_enc_sents[start_ind:start_ind+1] * (end_ind-sent_size)
                batch_tsf_len = batch_tsf_len + tsf_tsf_len[start_ind:start_ind+1]*(end_ind-sent_size)
            batch_style_rewards = style_discriminator.getStyleReward(sess, batch_tsf_enc_sents, batch_tsf_len)
            batch_lm_rewards = rnnlm.getLMReward(sess, batch_tsf_enc_sents)
            batch_semantic_rewards = semantic_discriminator.getSemanticReward(batch_orig_words, batch_tsf_words)
            if (end_ind > sent_size):
                batch_style_rewards = batch_style_rewards[:sent_size-start_ind]
                batch_lm_rewards = batch_lm_rewards[:sent_size-start_ind]
            style_rewards = style_rewards + list(np.reshape(batch_style_rewards, (-1,)))
            lm_rewards = lm_rewards + list(np.reshape(batch_lm_rewards, (-1,)))
            semantic_rewards = semantic_rewards + list(batch_semantic_rewards)
            start_ind += batch_size
        if (verbose):
            print("No-rolling reward: test_size: {}, style_rewards size: {}, lm_rewards size: {}, semantic_rewards size: {}".format(sent_size, \
                                                                                                                 len(style_rewards), len(lm_rewards), len(semantic_rewards)))
            print("avg style: {}, avg lm: {}, avg semantic: {}".format(np.mean(style_rewards), \
                                                                       np.mean(lm_rewards), np.mean(semantic_rewards)))
        mean_style_reward = np.mean(style_rewards)
        mean_sem_reward = np.mean(semantic_rewards)
        mean_lm_reward = np.mean(lm_rewards)
        weighted_reward = style_weight * mean_style_reward + semantic_weight * mean_sem_reward + \
                           lm_weight * mean_lm_reward
        return mean_style_reward, mean_sem_reward, mean_lm_reward, weighted_reward
        

    def get_reward(self, sess, generator, encoder_inputs, encoder_input_words, encoder_inputs_length, \
                   decoder_inputs, rnnlm, style_discriminator, semantic_discriminator, max_sent_len, rollout_num=8, verbose=False):
        # style rewards: max_sent_len, batch_size
        style_rewards = []
        # semantic rewards: max_sent_len, batch_size
        semantic_rewards = []
        # language model rewards: max_sent_len, batch_size
        lm_rewards = []
        for i in range(rollout_num):
            for given_time in range(1, max_sent_len):
                decoder_inputs_length = [max_sent_len] * len(encoder_inputs)
                # rollout_outputs: decoder index
                rollout_decoder_outputs = generator.rollGenerate(sess, encoder_inputs, encoder_inputs_length, \
                                                         decoder_inputs, decoder_inputs_length, given_time)
                max_ind = np.max(rollout_decoder_outputs)
                if (max_ind >= len(self.tsf_vocab_inv)):
                    print("max_ind in rollout_deocoder_outputs: {}, tsf vocab sie: {}".format(max_ind,len(self.tsf_vocab_inv)))
                    sys.exit(0)
                # rollout_wrods
                rollout_words = [[self.tsf_vocab_inv[ind] for ind in decoder_inds] for decoder_inds in rollout_decoder_outputs]
                # rollout_outputs: encoder index (inv: list, vocab: dict)
                rollout_encoder_outputs = [[self.vocab[word] for word in word_seq] for word_seq in rollout_words]
                # style discriminator: batch_size, 1
                rollout_outputs_len = [max_sent_len] * len(rollout_decoder_outputs)
                style_reward = style_discriminator.getStyleReward(sess, rollout_encoder_outputs, rollout_outputs_len)
                # semantic reward: batch_size, 1
                semantic_reward = semantic_discriminator.getSemanticReward(encoder_input_words, rollout_words)
                # lm reward:
                lm_reward = rnnlm.getLMReward(sess, rollout_encoder_outputs)
                # add other discriminator rewards
                if (i==0):
                    style_rewards.append(np.copy(style_reward))
                    semantic_rewards.append(np.copy(semantic_reward))
                    lm_rewards.append(np.copy(lm_reward))
                else:
                    style_rewards[given_time - 1] += style_reward
                    semantic_rewards[given_time - 1] += semantic_reward
                    lm_rewards[given_time - 1] += lm_reward
                    
            # reward of the last token, no need to rollout
            decoder_inputs_len = [len(sent) for sent in decoder_inputs]
            decoder_words = [[self.tsf_vocab_inv[ind] for ind in decoder_inds] for decoder_inds in decoder_inputs]
            # change to encoder indices
            decoder_enc_inputs = [[self.vocab[word] for word in word_seq] for word_seq in decoder_words]
            style_reward = style_discriminator.getStyleReward(sess, decoder_enc_inputs, decoder_inputs_len)
            semantic_reward = semantic_discriminator.getSemanticReward(encoder_input_words, decoder_words)
            lm_reward = rnnlm.getLMReward(sess, decoder_enc_inputs)
            if (i==0):
                style_rewards.append(np.copy(style_reward))
                semantic_rewards.append(np.copy(semantic_reward))
                lm_rewards.append(np.copy(lm_reward))
            else:
                style_rewards[max_sent_len - 1] += style_reward
                semantic_rewards[max_sent_len - 1] += semantic_reward
                lm_rewards[max_sent_len - 1] += lm_reward
        # style rewards: batch_size, max_sent_len
        style_rewards = np.transpose(style_rewards) / float(rollout_num)
        # semantic rewards: batch_size, max_sent_len
        semantic_rewards = np.transpose(semantic_rewards) / float(rollout_num)
        # lm rewards: batch_size, max_sent_len
        lm_rewards = np.transpose(lm_rewards) / float(rollout_num)
        if verbose:
            print("style: {}, semantic: {}, lm: {}".format(np.mean(style_rewards), np.mean(semantic_rewards), np.mean(lm_rewards)))
        # weighted_rewards: batch_size, max_sent_len
        weighted_rewards = style_weight *  style_rewards + semantic_weight * semantic_rewards + \
                           lm_weight * lm_rewards
        # rewards - sentence with EOS token: batch_size, max_sent_len
        weighted_rewards = np.concatenate((weighted_rewards,weighted_rewards[:, [-1]]), axis=1)
        
        discounted_rewards = []
        for row in weighted_rewards:
            # diff_row: \hat{s}[i+1]=s[i+1]-s[i]
            diff_row = []
            for ind in range(len(row)):
                if (ind == 0):
                    diff_row.append(row[ind])
                else:
                    diff_row.append(row[ind]-row[ind-1])
            # sum_row: \tilde{s}[i]=\sum_{j>=i}(\gamma^{j-i}\hat{s}[j])
            sum_row = []
            for i in range(len(row)):
                j = len(row)-1-i
                if (i == 0):
                    sum_row = [diff_row[j]] + sum_row
                else:
                    sum_row = [diff_row[j]+self.gamma*sum_row[0]] + sum_row
            discounted_rewards.append(sum_row[:])
        discounted_rewards = np.array(discounted_rewards)
        return discounted_rewards
        

















