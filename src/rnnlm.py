"""
word based language model
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np
from beam import BeamSearch


class RNNLM():
    def __init__(self, config, init_embed, infer=False):
        self.config = config
        self.vocab_size = config["lm_vocab_size"]
        self.rnn_size = config["lm_rnn_size"]
        self.num_layers = config["lm_num_layers"]
        self.model = config["lm_model"]
        self.batch_size = config["batch_size"]
        self.seq_length = config["lm_seq_length"]
        self.epochs = config["lm_epochs"]
        self.grad_clip = config["lm_grad_clip"]
        self.learning_rate = config["lm_learning_rate"]
        self.decay_rate = config["lm_decay_rate"]
        
        if infer:
            self.batch_size = 1
            self.seq_length = 1
            
        if self.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif self.model == 'gru':
            cell_fn = rnn.GRUCell
        elif self.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(self.model))

        cells = []
        for _ in range(self.num_layers):
            cell = cell_fn(self.rnn_size)
            cells.append(cell)
        self.cell = rnn.MultiRNNCell(cells)

        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)


        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, self.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
            embedding = tf.Variable(init_embed, name="embedding", dtype=tf.float32)
            inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), self.seq_length, 1)
            inputs = [tf.squeeze(input_, 1) for input_ in inputs]

        def loop(prev, _):
            # predict the next symbol, return its embedding
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope="rnnlm")
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.rnn_size])
        self.logits = tf.matmul(output, softmax_w) +  softmax_b
        self.probs = tf.nn.softmax(self.logits)
        # perplexity of sents: (batch_size,)
        self.loss = legacy_seq2seq.sequence_loss_by_example([self.logits], \
                                                       [tf.reshape(self.targets, [-1])], \
                                                       [tf.ones([self.batch_size * self.seq_length])], \
                                                       self.vocab_size)
        self.cost = tf.reduce_sum(self.loss) / self.batch_size / self.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(self.learning_rate, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def getLMReward(self, sess, sents):
        flat_sents = np.reshape(sents, (1, -1))
        flat_targets = np.copy(flat_sents)
        flat_targets[:-1] = flat_sents[1:]
        flat_targets[-1] = flat_sents[0]
        targets = np.reshape(flat_targets, np.shape(sents))
        feed = {self.input_data: sents, self.targets: targets}
        loss = sess.run(self.loss, feed)
        reward = -1.0 * np.mean(np.reshape(loss, np.shape(sents)), axis=1)
        return reward

    def getLMLoss(self, sess, sents):
        flat_sents  = np.reshape(sents, (1, -1))
        flat_targets = np.copy(flat_sents)
        flat_targets[:-1] = flat_sents[1:]
        flat_targets[-1] = flat_sents[0]
        targets = np.reshape(flat_targets, np.shape(sents))
        feed = {self.input_data: sents, self.targets: targets}
        loss = sess.run(self.loss, feed)
        sent_loss = np.sum(loss, axis=1)
        return sent_loss

    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1, pick=0, width=4, quiet=False):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        def beam_search_predict(sample, state):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """

            x = np.zeros((1, 1))
            x[0, 0] = sample[-1]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def beam_search_pick(prime, width):
            """Returns the beam search pick."""
            if not len(prime) or prime == ' ':
                prime = random.choice(list(vocab.keys()))
            prime_labels = [vocab.get(word, 0) for word in prime.split()]
            bs = BeamSearch(beam_search_predict,
                            sess.run(self.cell.zero_state(1, tf.float32)),
                            prime_labels)
            samples, scores = bs.search(None, None, k=width, maxsample=num)
            return samples[np.argmin(scores)]

        ret = ''
        if pick == 1:
            state = sess.run(self.cell.zero_state(1, tf.float32))
            if not len(prime) or prime == ' ':
                prime  = random.choice(list(vocab.keys()))
            if not quiet:
                print(prime)
            for word in prime.split()[:-1]:
                if not quiet:
                    print(word)
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word,0)
                feed = {self.input_data: x, self.initial_state:state}
                [state] = sess.run([self.final_state], feed)

            ret = prime
            word = prime.split()[-1]
            for n in range(num):
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {self.input_data: x, self.initial_state:state}
                [probs, state] = sess.run([self.probs, self.final_state], feed)
                p = probs[0]

                if sampling_type == 0:
                    sample = np.argmax(p)
                elif sampling_type == 2:
                    if word == '\n':
                        sample = weighted_pick(p)
                    else:
                        sample = np.argmax(p)
                else: # sampling_type == 1 default:
                    sample = weighted_pick(p)

                pred = words[sample]
                ret += ' ' + pred
                word = pred
        elif pick == 2:
            pred = beam_search_pick(prime, width)
            for i, label in enumerate(pred):
                ret += ' ' + words[label] if i > 0 else words[label]
        return ret       

        































