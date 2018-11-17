import tensorflow as tf
import numpy as np
import pickle
import utils
from model import Model
import os
import time
import re
import easydict


args = easydict.EasyDict({
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "rnn_hidden_dim" : 256,
        "mlp_layers" : [128, 64],
        "option" : [200, 20],
        "restore" : False,
        "save_interval" : 3,
        "batch_norm": True,
        "drop_out": True
    })
data_type = 'stft'

def layer_config_to_str(layer_config):
    return '_'.join([str(x) for x in layer_config])

dir_format = 'model/' + data_type +'/bs-{}_rnn-{}-mlp_layers-{}'
model_dir = dir_format.format(
    args.batch_size,
    args.rnn_hidden_dim,
    layer_config_to_str(args.mlp_layers)
)

if args.option:
    model_dir = '{}/_option-{}'.format(model_dir, '_'.join([*map(str, args.option)]))
else:
    model_dir = '{}/_option-{}'.format(model_dir, '')


data_type = 'stft'

def layer_config_to_str(layer_config):
    return '_'.join([str(x) for x in layer_config])

dir_format = 'model/' + data_type +'/bs-{}_rnn-{}-mlp_layers-{}'
model_dir = dir_format.format(
    args.batch_size,
    args.rnn_hidden_dim,
    layer_config_to_str(args.mlp_layers)
)

if args.option:
    model_dir = '{}/_option-{}'.format(model_dir, '_'.join([*map(str, args.option)]))
else:
    model_dir = '{}/_option-{}'.format(model_dir, '')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def train(which = 'stft', split_option = args.option):

    with tf.Graph().as_default():
        with tf.variable_scope('inputs'):
            next_batch, trn_init_op, test_init_op = utils.inputs(args.batch_size, which=data_type, split_option = args.option)
            tf.add_to_collection('test_init_op', test_init_op)
            tf.add_to_collection('train_init_op', trn_init_op)

        with tf.variable_scope('model'):
            model = Model(next_batch, args.rnn_hidden_dim, args.mlp_layers, 5, args)
            

        # gpu options
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(max_to_keep=10)
            summary_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)
            global_step = 1
            if args.restore:
                latest_model = tf.train.latest_checkpoint(model_dir)
                print('restored model from ', latest_model)
                #epoch_num = int(re.search('model.ckpt-(\d+)', latest_model).group(1))
                #sess.run(tf.assign(model.epoch, epoch_num))
                saver.restore(sess, latest_model)
            else:
                sess.run(tf.global_variables_initializer())
                epoch_num = sess.run(model.epoch)

            for _ in range(args.num_epochs):

                print('epoch num', epoch_num, 'batch iteration', global_step)
                prev = time.time()
                sess.run(trn_init_op)
                sess.run(tf.local_variables_initializer())

                trn_feed = {model.is_training: True}

                try:
                    while True:
                        if global_step % args.save_interval == 0:
                            _, global_step, trn_loss_summary, _ = sess.run([model.train_op,
                                                                            model.global_step,
                                                                            model.trn_loss_summary,
                                                                            model.summary_update_ops
                                                                            ],
                                                                           trn_feed
                                                                           )

                            summary_writer.add_summary(trn_loss_summary, epoch_num)
                        else:
                            _, global_step, loss, _ = sess.run([model.train_op,
                                                                model.global_step,
                                                                model.loss,
                                                                model.summary_update_ops
                                                                ],
                                                               trn_feed
                                                               )

                except tf.errors.OutOfRangeError:
                    sess.run(model.increment_epoch_op)
                    epoch_num = sess.run(model.epoch)
                    print('out of range', 'epoch', epoch_num, 'iter', global_step)
                    now = time.time()
                    summary_value, trn_acc = sess.run([model.summary_trn,
                                                       model.accuracy],
                                                      {model.is_training: False})
                    summary_writer.add_summary(summary_value, global_step=epoch_num)

                    sess.run(test_init_op)
                    sess.run(tf.local_variables_initializer())  # metrics value init to 0

                    try:
                        print('test_start')
                        tmp_step = 0

                        while True:
                            if tmp_step % args.save_interval == 0:
                                _, test_loss_summary = sess.run([model.summary_update_ops,
                                                                 model.test_loss_summary],
                                                                {model.is_training: False})
                                summary_writer.add_summary(test_loss_summary,
                                                           global_step=epoch_num)
                            else:
                                sess.run(model.summary_update_ops, {model.is_training: False})

                            tmp_step += 1

                    except tf.errors.OutOfRangeError:
                        print('test_start end')
                        summary_value, test_acc = sess.run([model.summary_test,
                                                            model.accuracy],
                                                           {model.is_training: False})
                        summary_writer.add_summary(summary_value, global_step=epoch_num)

                    minutes = (now - prev) / 60
                    result = 'num iter: {} | trn_acc : {} test acc : {}'.format(
                        global_step, trn_acc, test_acc)

                    message = 'took {} min'.format(minutes)
                    print(model_dir)
                    print(result)
                    print(message)

                    saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=epoch_num)

def inference(model_dir, args, mode='val'):
    """
    Inputs
        model_dir : string that indicated the directory the model is stored.
        mode : 'val' or 'test'
    """
    which = model_dir.split('model')[1].split('/')[1]
    split_list = re.findall("[0-9]+", model_dir.split('/')[-2].split('-')[-1])
    if len(split_list):
        args.option = [int(i) for i in split_list]
    else: 
        args.option = None
    args.rnn_hidden_dim = int(model_dir.split('rnn-')[1][:3])
    
    def most_common(lst):
        return max(set(lst), key=lst.count)
    
    if mode == 'val':
        with tf.Graph().as_default():
            with tf.variable_scope('inputs'):
                next_batch, trn_init_op, test_init_op = utils.inputs(100, which, split_option=args.option, test=False)
                tf.add_to_collection('test_init_op', test_init_op)
            with tf.variable_scope('model'):
                model = Model(next_batch, args.rnn_hidden_dim, args.mlp_layers, 5, args)

            # gpu options
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(test_init_op)
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()
                global_step = 1
                print('restored model from ', model_dir)
                saver.restore(sess, model_dir)
                sess.run(tf.local_variables_initializer())

                final_pred = {}
                true = {}
                try:
                    while True:
                        val_id, val_y, val_pred = sess.run([model.id, model.y, model.prediction],
                                                           {model.is_training: False})
                        for idx, track_id in enumerate(val_id):
                            if not track_id in final_pred.keys():
                                final_pred[track_id] = [val_pred[idx]]
                            else:
                                final_pred[track_id].append(val_pred[idx])

                            if not track_id in true.keys():
                                true[track_id] = val_y[idx]
                            else:
                                pass
                except tf.errors.OutOfRangeError:
                    corr_pred = 0
                    data_size = len(true)
                    for key, val in final_pred.items():
                        final_pred[key] = most_common(final_pred[key])
                        if true[key] == final_pred[key]:
                            corr_pred += 1
                        else:
                            pass
                    accuracy = corr_pred / data_size * 100.0
                    print("Validation accuracy is %.2f %%" % (accuracy))
                    #return final_pred, true

    elif mode == 'test':
        with tf.Graph().as_default():
            with tf.variable_scope('inputs'):
                next_batch, test_init_op = utils.inputs(1, which, split_option=args.option, test=True)
                tf.add_to_collection('test_init_op', test_init_op)

            # load model
            args.restore = True
            with tf.variable_scope('model'):
                model = Model(next_batch, args.rnn_hidden_dim, args.mlp_layers, 5, args)

                # gpu options
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                with tf.Session(config=config) as sess:
                    sess.run(test_init_op)
                    sess.run(tf.local_variables_initializer())
                    saver = tf.train.Saver()
                    global_step = 1
                    saver.restore(sess, model_dir)
                    sess.run(tf.local_variables_initializer())

                    final_pred = {}
                    true = {}
                    try:
                        while True:
                            val_id, val_y, val_pred = sess.run([model.id, model.y, model.prediction],
                                                               {model.is_training: False})

                            for idx, track_id in enumerate(val_id):
                                if not track_id in final_pred.keys():
                                    final_pred[track_id] = [val_pred[idx]]
                                else:
                                    final_pred[track_id].append(val_pred[idx])

                                if not track_id in true.keys():
                                    true[track_id] = val_y[idx]
                                else:
                                    pass
                    except tf.errors.OutOfRangeError:
                        corr_pred = 0
                        data_size = len(true)
                        for key, val in final_pred.items():
                            if true[key] == most_common(final_pred[key]):
                                corr_pred += 1
                            else:
                                pass
                        accuracy = corr_pred / data_size * 100.0
                        print("Test accuracy is %.2f %%" % (accuracy))
                        #return (final_pred, true)



if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # set default hyperparameters
    train = False
    if train:
        train()
    else:
        args.restore = True
        model_dirs = './model/stft/bs-32_rnn-512-mlp_layers-128_64/_option-123_10/model.ckpt-16'
        # change mode to 'test' for test validation
        inference(model_dirs, args = args, mode='val')