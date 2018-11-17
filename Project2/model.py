import tensorflow as tf


class Model():
    def __init__(self, inputs, rnn_hidden_dim, mlp_layers, y_size, args):
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.num_genre = 8

        self.is_training = tf.placeholder(tf.bool, shape=None)
        tf.add_to_collection('is_training', self.is_training)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        
        self.base_learning_rate = 1e-4
        self.batch_size_for_learning_rate = 32

        def build_mlp(inputs, layers):
            """"
            Inputs
                - inputs :
                - layers :
                - drop_out : layer_num (int) to apply dropout
            """
            for layer_num, layer_dim in enumerate(layers):
                with tf.variable_scope('dense_{}'.format(layer_num)):
                    if args.batch_norm:
                        inputs = _batch_norm(inputs, name='dense_{}'.format(layer_num))
                    inputs = tf.layers.dense(inputs, layer_dim, activation=tf.nn.relu,
                                             kernel_initializer=self.weight_initializer,
                                             bias_initializer=self.const_initializer)
                    if args.drop_out == layer_num:
                        inputs = tf.layers.dropout(inputs, rate=0.5, training=self.is_training)
                        print('dropout')
                    print('mlp', inputs.shape)
            return inputs
        

        def _batch_norm(x, name=None):
            return tf.contrib.layers.batch_norm(inputs=x,
                                                decay=0.95,
                                                center=True,
                                                scale=True,
                                                is_training= self.is_training,
                                                updates_collections=None,
                                                scope=(name + 'batch_norm'))

        def get_initial_lstm(features):
            with tf.variable_scope('initial_lstm'):
                # one hot encoding of genre idx
                features = tf.one_hot(features, depth=self.num_genre)
                w_h = tf.get_variable('w_h', [self.num_genre, rnn_hidden_dim], initializer=self.weight_initializer)
                b_h = tf.get_variable('b_h', [rnn_hidden_dim], initializer=self.const_initializer)
                h = tf.nn.tanh(tf.matmul(features, w_h) + b_h)

                w_c = tf.get_variable('w_c', [self.num_genre, rnn_hidden_dim],
                                      initializer=self.weight_initializer)
                b_c = tf.get_variable('b_c', [rnn_hidden_dim], initializer=self.const_initializer)
                c = tf.nn.tanh(tf.matmul(features, w_c) + b_c)
            return tf.nn.rnn_cell.LSTMStateTuple(c, h)
        self.id = inputs['track_id']
        self.x = inputs['x']  # track features
        self.x = _batch_norm(self.x, name='x')
        self.x_g = inputs['genre']  # genre -> integer value
        self.y = inputs['y']  # track listens -> integer value

        with tf.variable_scope('rnn_layer'):
            # TODO genre 로부터 hidden state initialize 하는 부분 만들기 -> DONE
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_hidden_dim)
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, output_keep_prob=0.8)
            initial_state = get_initial_lstm(self.x_g)
            rnn_outputs, (last_c, last_h) = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=self.x,
                initial_state=initial_state
            )

        with tf.variable_scope('mlp_out'):
            print('mlp_out')
            mlp_output = build_mlp(last_h, mlp_layers)
            self.logits = tf.layers.dense(mlp_output, y_size, use_bias=False)

        with tf.variable_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

        with tf.variable_scope('learning_rate'):
            self.increment_epoch_op = tf.assign(self.epoch, self.epoch + 1)
            if self.batch_size_for_learning_rate < 64:
                self.learning_rate = self.base_learning_rate
            else:
                self.learning_rate = tf.train.polynomial_decay(
                    self.base_learning_rate,
                    self.epoch,
                    decay_steps=5,
                    end_learning_rate=self.base_learning_rate * (self.batch_size_for_learning_rate / 64))

        with tf.variable_scope('summary'):
            self.prediction = tf.argmax(self.logits, axis=-1)
            tf.add_to_collection('pred', self.prediction)

            self.accuracy, _ = tf.metrics.accuracy(self.y, self.prediction, updates_collections='summary_update')

            summary_trn = list()
            summary_trn.append(tf.summary.scalar('trn_accuracy', self.accuracy))
            summary_trn.append(tf.summary.scalar('learning_rate', self.learning_rate))

            summary_test = list()
            summary_test.append(tf.summary.scalar('test_accuracy', self.accuracy))

            self.summary_trn = tf.summary.merge(summary_trn)
            self.summary_test = tf.summary.merge(summary_test)

            trn_loss_summary = [tf.summary.scalar('trn_xent_loss', self.loss)]
            test_loss_summary = [tf.summary.scalar('test_xent_loss', self.loss)]

            self.trn_loss_summary = tf.summary.merge(trn_loss_summary)
            self.test_loss_summary = tf.summary.merge(test_loss_summary)

        with tf.variable_scope('train'):
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.summary_update_ops = tf.get_collection('summary_update')

            with tf.control_dependencies(self.update_ops):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.loss, global_step=self.global_step
                )


class Stacked_LSTM():
    def __init__(self, inputs, rnn_hidden_dim, mlp_layers, y_size, args):
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.num_genre = 8

        self.is_training = tf.placeholder(tf.bool, shape=None)
        tf.add_to_collection('is_training', self.is_training)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch = tf.Variable(0, trainable=False, name='epoch')
        
        self.base_learning_rate = 1e-4
        self.batch_size_for_learning_rate = 32
        self.num_layers = 2

        def build_mlp(inputs, layers, drop_out=None, batch_norm=False):
            """"
            Inputs
                - inputs :
                - layers :
                - drop_out : layer_num (int) to apply dropout
            """
            for layer_num, layer_dim in enumerate(layers):
                with tf.variable_scope('dense_{}'.format(layer_num)):
                    if batch_norm:
                        inputs = _batch_norm(inputs, name='dense_{}'.format(layer_num))
                    inputs = tf.layers.dense(inputs, layer_dim, activation=tf.nn.relu,
                                             kernel_initializer=self.weight_initializer,
                                             bias_initializer=self.const_initializer)
                    if drop_out == layer_num:
                        inputs = tf.layers.dropout(inputs, rate=0.5, training=self.is_training)
                        print('dropout')
                    print('mlp', inputs.shape)
            return inputs
        

        def _batch_norm(x, name=None):
            return tf.contrib.layers.batch_norm(inputs=x,
                                                decay=0.95,
                                                center=True,
                                                scale=True,
                                                is_training= self.is_training,
                                                updates_collections=None,
                                                scope=(name + 'batch_norm'))

        def get_initial_lstm(features):
            with tf.variable_scope('initial_lstm'):
                # one hot encoding of genre idx
                features = tf.one_hot(features, depth=self.num_genre)
                w_h = tf.get_variable('w_h', [self.num_genre, rnn_hidden_dim], initializer=self.weight_initializer)
                b_h = tf.get_variable('b_h', [rnn_hidden_dim], initializer=self.const_initializer)
                h = tf.nn.tanh(tf.matmul(features, w_h) + b_h)

                w_c = tf.get_variable('w_c', [self.num_genre, rnn_hidden_dim],
                                      initializer=self.weight_initializer)
                b_c = tf.get_variable('b_c', [rnn_hidden_dim], initializer=self.const_initializer)
                c = tf.nn.tanh(tf.matmul(features, w_c) + b_c)
            return tf.nn.rnn_cell.LSTMStateTuple(c, h)

        x = inputs['x']  # track features
        x = _batch_norm(x, name='x')
        x_g = inputs['genre']  # genre -> integer value
        y = inputs['y']  # track listens -> integer value

        with tf.variable_scope('rnn_layer'):
            # TODO genre 로부터 hidden state initialize 하는 부분 만들기 -> DONE
            x = tf.unstack(x, x.shape[0], 1)
            stack_rnn = []
            for i in range(self.num_layers):
                rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_hidden_dim)
                rnn_cell = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, output_keep_prob=0.8)
                stack_rnn.append(rnn_cell)
            stacked_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_cell, state_is_tuple=True)
            
            rnn_outputs, (_, last_h) = rnn.static_rnn(stacked_cell, x, dtype=tf.float)

        with tf.variable_scope('mlp_out'):
            print('mlp_out')
            mlp_output = build_mlp(last_h, mlp_layers)
            logits = tf.layers.dense(mlp_output, y_size, use_bias=False)

        with tf.variable_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            self.loss = tf.reduce_mean(loss)

        with tf.variable_scope('learning_rate'):
            self.increment_epoch_op = tf.assign(self.epoch, self.epoch + 1)
            if self.batch_size_for_learning_rate < 64:
                self.learning_rate = self.base_learning_rate
            else:
                self.learning_rate = tf.train.polynomial_decay(
                    self.base_learning_rate,
                    self.epoch,
                    decay_steps=5,
                    end_learning_rate=self.base_learning_rate * (self.batch_size_for_learning_rate / 64))

        with tf.variable_scope('summary'):
            self.prediction = tf.argmax(logits, axis=-1)
            tf.add_to_collection('pred', self.prediction)

            self.accuracy, _ = tf.metrics.accuracy(y, self.prediction, updates_collections='summary_update')

            summary_trn = list()
            summary_trn.append(tf.summary.scalar('trn_accuracy', self.accuracy))
            summary_trn.append(tf.summary.scalar('learning_rate', self.learning_rate))

            summary_test = list()
            summary_test.append(tf.summary.scalar('test_accuracy', self.accuracy))

            self.summary_trn = tf.summary.merge(summary_trn)
            self.summary_test = tf.summary.merge(summary_test)

            trn_loss_summary = [tf.summary.scalar('trn_xent_loss', self.loss)]
            test_loss_summary = [tf.summary.scalar('test_xent_loss', self.loss)]

            self.trn_loss_summary = tf.summary.merge(trn_loss_summary)
            self.test_loss_summary = tf.summary.merge(test_loss_summary)

        with tf.variable_scope('train'):
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.summary_update_ops = tf.get_collection('summary_update')

            with tf.control_dependencies(self.update_ops):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.loss, global_step=self.global_step
                )
