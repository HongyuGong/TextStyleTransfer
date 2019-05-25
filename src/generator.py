"""
generator:
given original sentence, generator generates tsf_sent,
"""

import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.ops import tensor_array_ops
import params

class Generator(object):
    def __init__(self, config, init_embed, decoder_init_embed):
        self.config = config
        self.dtype = tf.float32
        self.cell_type = config['cell_type']
        self.hidden_units = config['hidden_units'] # dimension of hidden cell
        self.depth = config['depth'] # lstm layers
        self.attention_type = config['attention_type']
        self.embedding_dim = config['embedding_dim'] # prev: embedding_size
        self.encoder_vocab_size = config['encoder_vocab_size']
        self.decoder_vocab_size = config['decoder_vocab_size']
        self.max_sent_len = config['max_sent_len']
        self.attn_input_feeding = config['attn_input_feeding'] # indicate whether to use attention
        self.use_dropout = config['use_dropout']
        self.optimizer = config['optimizer']
        self.learning_rate = config['learning_rate']
        self.rl_learning_rate = config['rl_learning_rate']
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.beam_width = config['beam_width']
        self.grad_clip = config['grad_clip']

        self.build_model(init_embed, decoder_init_embed)
        

    def build_model(self, init_embed, decoder_init_embed):
        print("building model...")
        self.init_placeholder()
        self.build_encoder(init_embed)
        self.build_decoder(decoder_init_embed)
        self.summary_op = tf.summary.merge_all()


    def init_placeholder(self):
        # encoder_inputs: word indices [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sent_len), name="encoder_inputs")
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None, ), name="encder_inputs_length")
        """
        generate: encoder_inputs -> tsf_sents
        rollout: encoder_inputs, decoder_inputs, given_time -> sampled_tsf_sents
        train: encoder_inputs, decoder_inputs, rewards -> loss
        """
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        # rollout_mode
        self.given_time = tf.placeholder(dtype=tf.int32, name="given_time")

        # train_mode
        self.rewards = tf.placeholder(dtype=tf.float32, shape=(None, self.max_sent_len+1))

        # rollout_mode or train_mode
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sent_len), name="decoder_inputs")
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None, ), name="decoder_inputs_length")
            

    def _build_single_cell(self):
        cell_type = LSTMCell
        if (self.cell_type.lower() == "gru"):
            cell_type = GRUCell
        cell = cell_type(self.hidden_units)
        return cell


    def _build_encoder_cell(self):
        return MultiRNNCell([self._build_single_cell() for i in range(self.depth)])
        

    def build_encoder(self, init_embed):
        print("building encoder...")
        with tf.variable_scope('encoder'):
            self.encoder_cell = self._build_encoder_cell()
            self.encoder_embeddings = tf.Variable(init_embed, name="encoder_embedding", dtype=self.dtype)
            self.encoder_vocab_size = len(init_embed)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings, \
                                                                  ids=self.encoder_inputs)
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name="input_projection")
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)
            self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(cell=self.encoder_cell, \
                                                                              inputs=self.encoder_inputs_embedded, \
                                                                              sequence_length=self.encoder_inputs_length, \
                                                                              dtype=self.dtype, time_major=False)

    def _build_decoder_cell(self):
        # no beam
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs
            _input_layer = Dense(self.hidden_units, dtype=self.dtype, name="attn_input_feeding")
            return _input_layer(array_ops.concat([inputs, attention], -1))
        
        # attention mechanism 'luong'
        with tf.variable_scope('shared_attention_mechanism'):
            self.attention_mechanism = attention_wrapper.LuongAttention(num_units=self.hidden_units, \
                                                                        memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)        
        # build decoder cell
        self.init_decoder_cell_list = [self._build_single_cell() for i in range(self.depth)]
        decoder_initial_state = encoder_last_state
        
        self.decoder_cell_list = self.init_decoder_cell_list[:-1] + [attention_wrapper.AttentionWrapper(\
            cell = self.init_decoder_cell_list[-1], \
            attention_mechanism=self.attention_mechanism,\
            attention_layer_size=self.hidden_units,\
            cell_input_fn=attn_decoder_input_fn,\
            initial_cell_state=encoder_last_state[-1],\
            alignment_history=False)]
        batch_size = self.batch_size
        initial_state = [state for state in encoder_last_state]
        initial_state[-1] = self.decoder_cell_list[-1].zero_state(batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)
        
        # beam
        beam_encoder_outputs = seq2seq.tile_batch(self.encoder_outputs, multiplier=self.beam_width)
        beam_encoder_last_state = nest.map_structure(lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_last_state)
        beam_encoder_inputs_length = seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_width)

        with tf.variable_scope('shared_attention_mechanism', reuse=True):
            self.beam_attention_mechanism = attention_wrapper.LuongAttention(num_units=self.hidden_units, \
                                                                             memory=beam_encoder_outputs, \
                                                                             memory_sequence_length=beam_encoder_inputs_length)

        beam_decoder_initial_state = beam_encoder_last_state
        self.beam_decoder_cell_list = self.init_decoder_cell_list[:-1] + [attention_wrapper.AttentionWrapper(\
            cell = self.init_decoder_cell_list[-1], \
            attention_mechanism=self.beam_attention_mechanism,\
            attention_layer_size=self.hidden_units,\
            cell_input_fn=attn_decoder_input_fn,\
            initial_cell_state=beam_encoder_last_state[-1],\
            alignment_history=False)]
            
        beam_batch_size = self.batch_size * self.beam_width
        beam_initial_state = [state for state in beam_encoder_last_state]
        beam_initial_state[-1] = self.beam_decoder_cell_list[-1].zero_state(batch_size=beam_batch_size, dtype=self.dtype)
        beam_decoder_initial_state = tuple(beam_initial_state)
        
        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state, \
               MultiRNNCell(self.beam_decoder_cell_list), beam_decoder_initial_state

        
    
    def build_decoder(self, decoder_init_embed):
        print("building attention and decoder...")
        with tf.variable_scope('decoder'):
            self.decoder_cell, self.decoder_initial_state, self.beam_decoder_cell, self.beam_decoder_initial_state \
                               = self._build_decoder_cell()
            # initializer
            self.decoder_embeddings = tf.Variable(decoder_init_embed, name="decoder_embedding", dtype=self.dtype)
            self.decoder_vocab_size = len(decoder_init_embed)
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name="input_projection")
            output_layer = Dense(decoder_init_embed.shape[0], name="output_projection")

            # generate_mode
            decoder_start_tokens = tf.ones(shape=[self.batch_size, ], dtype=tf.int32) * params.start_token
            decoder_end_token =  params.end_token
            def embed_and_input_proj(inputs):
                return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))


            print('greedy decoding...')
            generate_decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=decoder_start_tokens, \
                                                            end_token=decoder_end_token, \
                                                            embedding=embed_and_input_proj)
            generate_inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                         helper=generate_decoding_helper,
                                                         initial_state=self.decoder_initial_state,
                                                         output_layer=output_layer)
            with tf.variable_scope('decode_with_shared_attention'):
                self.gen_outputs, decoder_last_state, gen_outputs_len = (seq2seq.dynamic_decode(
                    decoder=generate_inference_decoder, \
                    output_time_major=False, \
                    maximum_iterations=self.max_sent_len)) # params.max_decoder_len              
            # self.gen_x: batch_size, max_decoder_len
            self.gen_x = self.gen_outputs.sample_id

            
            print("beam decoding...")
            beam_generate_inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.beam_decoder_cell, \
                                                                                    embedding=embed_and_input_proj, \
                                                                                    start_tokens=decoder_start_tokens, \
                                                                                    end_token=decoder_end_token, \
                                                                                    initial_state=self.beam_decoder_initial_state, \
                                                                                    beam_width=self.beam_width, \
                                                                                    output_layer=output_layer)
            with tf.variable_scope('decode_with_shared_attention', reuse=True):
                self.beam_gen_outputs, beam_decoder_last_state, beam_gen_outputs_len = (seq2seq.dynamic_decode(
                    decoder=beam_generate_inference_decoder, \
                    output_time_major=False, \
                    maximum_iterations=self.max_sent_len)) # params.max_decoder_len
            self.beam_gen_x = self.beam_gen_outputs.predicted_ids



            print("decoder for rollout")
            # decoder inputs in train and rollout mode
            self.decoder_inputs_embedded = input_layer(tf.nn.embedding_lookup(params=self.decoder_embeddings, \
                                                                              ids=self.decoder_inputs))
           # rollout mode
            rollout_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=decoder_start_tokens, \
                                                           end_token=decoder_end_token, \
                                                           embedding=embed_and_input_proj)
            rollout_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell, \
                                                    helper=rollout_helper, \
                                                    initial_state=self.decoder_initial_state, \
                                                    output_layer=output_layer)
            # calc samples for each time step (fix sent[:given_time+1], roll out sent[given_time+1:])
            self.rollout_decoder_state = self.decoder_initial_state
            # rollout_outputs shape: max_decoder_len, batch_size

            rollout_outputs = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.max_sent_len, \
                                                                dynamic_size=False, infer_shape=True)
            init_inputs_embedded = embed_and_input_proj(decoder_start_tokens)
            i = tf.constant(0)
            while_condition = lambda i, inputs_embedded, decoder_state, rollout_outputs, given_time: tf.less(i, given_time)
            def feed_body(i, inputs_embedded, decoder_state, rollout_outputs, given_time):
                print("feed body iter:", i)
                next_outputs, decoder_state, next_inputs, decoder_finished = rollout_decoder.step(i, inputs_embedded, decoder_state)
                inputs =  tf.reshape(tf.gather(params=self.decoder_inputs, indices=[i], axis=1), shape=[self.batch_size, ])
                inputs_embedded = embed_and_input_proj(inputs)
                rollout_outputs = rollout_outputs.write(i, inputs)
                return i+1, inputs_embedded, decoder_state, rollout_outputs, given_time               
            i, inputs_embedded, self.rollout_decoder_state, self.rollout_outputs,  _ = tf.while_loop(while_condition, feed_body, \
                                                                                            (0, init_inputs_embedded, \
                                                                                             self.rollout_decoder_state, rollout_outputs, self.given_time))
            # next_outputs shape: (batch_size, decoder_vocab_size)
            inputs =  tf.reshape(tf.gather(params=self.decoder_inputs, indices=[self.given_time], axis=1), shape=[self.batch_size, ])
            inputs_embedded = input_layer(tf.nn.embedding_lookup(params=self.decoder_embeddings, \
                                                             ids=inputs))
            # rollout outputs: sample from output probability
            i = self.given_time
            while_condition = lambda i, inputs_embedded, decoder_state, rollout_outputs, max_len: tf.less(i, self.max_sent_len)
            def pred_body(i, inputs_embedded, decoder_state, rollout_outputs, max_len):
                print("pred body iter", i)
                # record rollout sentences
                next_outputs, decoder_state, next_inputs, decoder_finished = rollout_decoder.step(i, inputs_embedded, \
                                                                           decoder_state)
                inputs = tf.cast(tf.reshape(tf.multinomial(next_outputs.rnn_output, 1), [self.batch_size, ]), tf.int32)
                inputs_embedded =  embed_and_input_proj(inputs)
                rollout_outputs = rollout_outputs.write(i, inputs)
                return i+1, inputs_embedded, decoder_state, rollout_outputs, max_len
            i, inputs_embedded, self.rollout_decoder_state, self.rollout_outputs, _ = tf.while_loop(while_condition, pred_body, (i, inputs_embedded, self.rollout_decoder_state, \
                                                           self.rollout_outputs, self.max_sent_len))
            self.rollout_outputs = self.rollout_outputs.stack()
            self.rollout_outputs = tf.transpose(self.rollout_outputs, perm=[1,0])


            # train mode
            print("decoder for both pre-training and RL training")
            decoder_start_token_train= tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * params.start_token
            decoder_end_token_train= tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * params.end_token
            self.decoder_inputs_train = tf.concat([decoder_start_token_train, self.decoder_inputs], axis=1)
            self.decoder_inputs_length_train= self.decoder_inputs_length + 1
            self.decoder_targets_train = tf.concat([self.decoder_inputs, decoder_end_token_train], axis=1)
            self.decoder_inputs_embedded_train = tf.nn.embedding_lookup(params=self.decoder_embeddings, \
                                                                        ids=self.decoder_inputs_train)
            self.decoder_inputs_embedded_train = input_layer(self.decoder_inputs_embedded_train)

            training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded_train, \
                                                     sequence_length=self.decoder_inputs_length_train, \
                                                     time_major=False,
                                                     name="training_helper")
            training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell, \
                                                    helper=training_helper, \
                                                    initial_state=self.decoder_initial_state, \
                                                    output_layer=output_layer)
            max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)
            self.decoder_outputs_train,  self.decoder_last_state_train, self.decoder_ouputs_len_train \
                                        = seq2seq.dynamic_decode(\
                                            decoder = training_decoder, \
                                            output_time_major = False, \
                                            impute_finished = True, \
                                            maximum_iterations = max_decoder_length)
            # flat-and-pad: rnn_output: batch_size * max_decoder_length * decoder_vocab_size
            self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
            logits_padding = tf.one_hot(indices=tf.ones(shape=[self.batch_size, self.max_sent_len+1-max_decoder_length], dtype=tf.int32) * params.end_token, \
                                        depth=self.decoder_vocab_size, on_value=10.0, off_value=-20.0, axis=-1, dtype=self.dtype)
            # decoder_logits_train_pad: batch_size * (params.max_decoder_len+1 )* decoder_vocab_size
            self.decoder_logits_train_pad = tf.concat([self.decoder_logits_train, logits_padding], axis=1)
            
            # pre-train loss
            masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train, maxlen=self.max_sent_len+1, \
                                     dtype=self.dtype, name="masks")
            self.pretrain_g_loss = seq2seq.sequence_loss(logits=tf.identity(self.decoder_logits_train_pad), \
                                                         targets=self.decoder_targets_train, \
                                                         weights=masks,\
                                                         average_across_timesteps=True,\
                                                         average_across_batch=True)
            # rl loss
            self.gen_prob = tf.nn.softmax(self.decoder_logits_train_pad)
            self.g_loss = -1.0 * tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.decoder_targets_train, [-1])), self.decoder_vocab_size, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(self.gen_prob, [-1, self.decoder_vocab_size]), 1e-20, 1.0)), 1) \
                              * tf.reshape(self.rewards, [-1]))
            self.init_optimizer()


    def generate(self, sess, encoder_inputs, encoder_inputs_length):
        feed = {self.encoder_inputs: encoder_inputs, \
                self.encoder_inputs_length: encoder_inputs_length}
        outputs = sess.run(self.beam_gen_x, feed_dict=feed)
        return outputs

    def rollGenerate(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length, given_time):
        feed = {self.encoder_inputs: encoder_inputs, \
                self.encoder_inputs_length: encoder_inputs_length, \
                self.decoder_inputs: decoder_inputs, \
                self.decoder_inputs_length: decoder_inputs_length, \
                self.given_time: given_time}
        rollout_outputs = sess.run(self.rollout_outputs, feed_dict=feed)
        return rollout_outputs

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)
    

    def init_optimizer(self):
        # trainable parameters
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
        trainable_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")

        # pretrain gradient
        pretrain_g_opt = self.g_optimizer(self.learning_rate)
        self.pretrain_g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_g_loss, trainable_params), self.grad_clip)
        self.pretrain_g_updates = pretrain_g_opt.apply_gradients(zip(self.pretrain_g_grad, trainable_params), global_step=self.global_step)

        # rl gradient
        g_opt = self.g_optimizer(self.rl_learning_rate)
        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, trainable_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, trainable_params), global_step = self.global_step)








        

    
    
