import tensorflow as tf
from dataloader import *
from tensorflow.python.platform import gfile
from utils import chord_idx_mapping


class CNN_Model:
    def __init__(self, batch_size, weight_decay_rate=0.001):
        self.classes = 25
        self.timesteps = 96
        self.batch_size = batch_size
        self.time_axis = 1034
        self.freq_axis = 12
        self.channels = 1
        self.logits = None
        self.cost = 0.0
        self.is_train_mode = None
        self.features = None
        self.chords = None
        self.weight_decay_rate = weight_decay_rate
        self.regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay_rate)

    def build_graph(self, X, Y, is_train_mode=False):
        self.features = X
        self.chords = Y
        self.is_train_mode = is_train_mode
        features = tf.transpose(self.features, perm=[0, 3, 2, 1])
        self.batch_size, self.freq_axis, self.time_axis, _ = features.shape
        conv1 = tf.layers.conv2d(features, filters=30, kernel_size=[12, 4], strides=2,
                                 padding='same', activation=tf.nn.elu)
        mp1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='same')
        conv2 = tf.layers.conv2d(mp1, filters=60, kernel_size=[12, 4], strides=2,
                                 padding='same', activation=tf.nn.elu)
        mp2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=2, padding='same')
        flatten = tf.layers.flatten(mp2)
        dense = tf.layers.dense(flatten, units=self.timesteps*self.classes, activation=tf.nn.relu)  # 25 categories, 96 timesteps
        dropout = tf.layers.dropout(dense, rate=0.7, training=self.is_train_mode)
        self.logits = tf.reshape(dropout, [-1, self.timesteps, self.classes])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.chords, dim=2))

        return self.logits, self.cost


class CRNN_Model:
    def __init__(self, hidden_dim=64, input_len=64, multi=False, batch_size=32, weight_decay_rate=0.001):
        self.classes = 25
        self.timesteps = 96
        self.input_len = input_len
        self.batch_size = batch_size
        self.time_axis = 1034
        self.freq_axis = 12
        self.channels = 1
        self.multi = multi
        self.hidden_dim = hidden_dim
        self.logits = None
        self.pred = None
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.weight_decay_rate = weight_decay_rate
        self.regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay_rate)


        self.is_train_mode = None
        self.features = None
        self.chords = None

    def _decode_lstm(self, output, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.hidden_dim, self.classes], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.classes], initializer=self.const_initializer)
            logits = tf.matmul(output, w_h) + b_h
            return logits

    def build_graph(self, X, Y, is_train_mode=False):
        # 1. Build raw inputs into a (feature_len * 96) sized
        # feature map through a 2-layer conv net
        self.features = X
        self.chords = Y
        self.is_train_mode = is_train_mode
        features = tf.transpose(self.features, perm=[0, 3, 2, 1]) # NCWH -> NHWC
        _, self.freq_axis, self.time_axis, _ = features.shape
        conv1 = tf.layers.conv2d(features, filters=30, kernel_size=[12, 4], strides=2,
                                 padding='same', activation=tf.nn.elu)
        mp1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='same')
        conv2 = tf.layers.conv2d(mp1, filters=60, kernel_size=[12, 4], strides=2,
                                 padding='same', activation=tf.nn.elu)
        mp2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=2, padding='same')
        flatten = tf.layers.flatten(mp2)
        dense = tf.layers.dense(flatten, units=self.timesteps*self.input_len, activation=tf.nn.relu)  # 25 categories, 96 timesteps
        dropout = tf.layers.dropout(dense, rate=0.7, training=self.is_train_mode)
        # Feed cnn outputs as rnn inputs
        rnn_input = tf.reshape(dropout, [-1, self.input_len, self.timesteps])
        loss = 0.0

        if self.multi:
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
            cell2 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
            multi_cell = tf.contrib.rnn.MultiRNNCell([cell, cell2])
            outputs, states = tf.nn.dynamic_rnn(multi_cell, rnn_input, dtype=tf.float32)

        else:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_dim)
            state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            logits = []
            for t in range(self.timesteps):
                with tf.variable_scope('lstm', reuse=(t != 0)):
                    _, (c, h) = lstm_cell(inputs=rnn_input[:,:,t], state=state)
                logit_per_t = self._decode_lstm(h, reuse=(t != 0))
                logits.append(logit_per_t)
                loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.chords[:,t,:], logits=logit_per_t))
            cost = loss / tf.to_float(self.batch_size)
            logits = tf.transpose(tf.stack(logits), (1,0,2))
            return logits, cost
