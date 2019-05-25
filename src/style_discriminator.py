import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from attention import attention
import numpy as np
import params

class StyleDiscriminator(object):
    def __init__(self, num_classes, embedding_size, init_embed, hidden_size, \
                 attention_size, max_sent_len, keep_prob):
        # word index
        self.input_x = tf.placeholder(tf.int32, [None, max_sent_len], name="input_x")
        # output probability
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.sequence_length = tf.placeholder(tf.int32, [None], name="input_len")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope('style_discriminator'):
            # embedding layer with initialization
            with tf.name_scope("embedding"):
                # trainable embedding
                W = tf.Variable(init_embed, name="W", dtype=tf.float32)
                self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

            # RNN layer + attention
            with tf.name_scope("bi-rnn"):
                rnn_outputs, _ = bi_rnn(GRUCell(hidden_size), GRUCell(hidden_size),\
                                        inputs=self.embedded_chars, sequence_length=self.sequence_length, \
                                        dtype=tf.float32)
                attention_outputs, self.alphas = attention(rnn_outputs, attention_size, return_alphas=True)
                drop_outputs = tf.nn.dropout(attention_outputs, keep_prob)

            # Fully connected layer
            with tf.name_scope("fc-layer"):
                W = tf.Variable(tf.truncated_normal([drop_outputs.get_shape()[1].value, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1,  shape=[num_classes]), name="b")
                self.scores = tf.sigmoid(tf.nn.xw_plus_b(drop_outputs, W, b), name="scores")

            # mean square error
            with tf.name_scope("mse"):
                self.loss = tf.reduce_mean(tf.square(tf.subtract(self.scores, self.input_y)))

        self.params = [param for param in tf.trainable_variables() if 'style_discriminator' in param.name]
        sd_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = sd_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = sd_optimizer.apply_gradients(grads_and_vars)

    def getStyleReward(self, sess, sents, sents_len):
        feed = {self.input_x: sents, self.sequence_length: sents_len}
        rewards = sess.run(self.scores, feed_dict=feed)
        rewards = np.reshape(rewards, (-1,))
        return np.array(rewards)
        
        









