import tensorflow as tf
import pandas as pd
import numpy as np
from dataloader import DataLoader
from tensorflow.python.platform import gfile
from time import strftime, localtime, time
import matplotlib.pyplot as plt
import pickle
import math
from preprocess import raw_to_pickle, load_pickle
from config import H, W, filter_size, n_filter, pool_size, fc_out_dim, weight_decay_rate
from model import model
import glob
import os.path

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# hyperparameters
# TODO : declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
batch_size = 5
epoch_num = 10

# fixed
metadata_path = 'track_metadata.csv'
# True if you want to train, False if you already trained your model

### TODO : IMPORTANT !!! Please change it to False when you submit your code
is_train_mode = False
### TODO : IMPORTANT !!! Please specify the path where your best model is saved


### example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint'
# 'track_genre_top' for project 1, 'listens' for project 2
label_column_name = 'track_genre_top'

# build model
# TODO : build your model here

n_input = 20*1231
n_classes = 8
image_height = 20
image_width = 1231

conv_size =(3,3)
n_filter1 = 64
n_filter2 = 160
n_filter3 = 128
pool_size = 2
fc_out_dim = 64

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, image_height, image_width, 1])
y = tf.placeholder(tf.int64, [None, n_classes])


logits, y_out = model(X,y,is_training=True)

# train and evaluate
def train(train_x, train_y, is_train_mode = True, use_pkl = False,
          batch_size = batch_size, epoch_num = epoch_num, 
          print_every = 100, save_every = 5, 
          learning_rate = 1e4, lr_decay = True,
          restore = False, checkpoint_path = './checkpoint'):
    
    if is_train_mode:
        # Clear old variables.        
        tf.reset_default_graph()

        # Initialize placeholder variables.
        X = tf.placeholder(tf.float32, [None, H, W, 1])
        y = tf.placeholder(tf.int64, [None, n_classes])
        global_step = tf.Variable(0, trainable=False)

        logits, y_pred = model(X, y, is_training = is_train_mode)
        avg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits=logits))
        
        if lr_decay:
            starter_learning_rate = learning_rate
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                    5000, 0.97, staircase=True)
        optimizer = tf.train.RMSPropOptimizer(learning_rate) 
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = optimizer.minimize(avg_loss, global_step=global_step)
        is_correct = tf.equal(tf.argmax(y_pred,-1), tf.argmax(train_y,-1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        
        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if restore:
                saver.restore(sess, checkpoint_path)			
            variables = [avg_loss, is_correct, train_step]
            train_indicies = np.arange(train_x.shape[0])
            np.random.shuffle(train_indicies)
            iter_cnt = 0
            for e in range(epoch_num):
                # Compute training loss
                n_correct = 0
                losses = []
                n_iter = int(np.ceil(train_x.shape[0]/batch_size))
                for i in range(n_iter):
                    start_idx = (i*batch_size)%train_x.shape[0]
                    idx = train_indicies[start_idx:start_idx+batch_size]
                    feed_dict = {X: train_x[idx,:], y: train_y[idx]}
                    actual_batch_size = train_y[idx].shape[0]
                    loss, correct, _ = sess.run(variables, feed_dict=feed_dict)
                    losses.append(loss * actual_batch_size)
                    n_correct += np.sum(correct)
                    iter_cnt += 1
                    if (iter_cnt % print_every) == 0:
                        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                                .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                total_correct = correct/train_x.shape[0]
                total_loss = np.sum(losses)/train_x.shape[0]
                print("Epoch {2}, Overall loss = {0:.3f} and accuracy of {1:.3f}"\
                .format(total_loss,total_correct,e))
            print('Training finished !')
            output_dir = checkpoint_path + '/run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5]
            if not gfile.Exists(output_dir):
                gfile.MakeDirs(output_dir)
            saver.save(sess, output_dir)
            print('Model saved in file : %s' % output_dir)


def evaluate(mode = "valid"):
    if mode == "valid":
        if not os.path.isfile('val_data.pkl'):
            raw_to_pickle(metadata_path, mode = "valid")
        val_x, val_y = load_pickle('val_data.pkl')
        
    if mode == "test":
        if not os.path.isfile('test_data.pkl'):
            raw_to_pickle(metadata_path, mode = "test")
        val_x, val_y = load_pickle('test_data.pkl')

    _, H, W, _ = val_x.shape
    n_classes = val_y.shape[-1]
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, H, W, 1])
    y = tf.placeholder(tf.int64, [None, n_classes])
    
    logits, y_pred = model(X, y, is_training = False)
    avg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits=logits))
                
    total_pred = np.zeros_like(val_y)
    best_models = ['./checkpoint/run-1029-0518', './checkpoint/run-1029-0412', './checkpoint/run-1029-0408', './checkpoint/run-1029-0553', './checkpoint/run-1029-1857']
    #glob.glob(checkpoint_path + '/run-*')
    #best_models = [m for m in best_models if len(m) == len('./checkpoint/run-0000-0000')]

    #average_val_cost = 0.0
    with tf.Session(config = config) as sess:
        for checkpoint_path in best_models:
            curr_pred = np.zeros_like(val_y)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            batch_size = 100
            n_iter = int(val_x.shape[0] / batch_size)
            
            for step in range(n_iter):
                offset = (step*batch_size)
                batch_x = val_x[offset: offset+batch_size, :]
                batch_y = val_y[offset: offset+batch_size]
                feed_dict_val = {X: batch_x, y: batch_y}
                pred_y = sess.run(y_pred, feed_dict=feed_dict_val)
                curr_pred[offset: offset+batch_size, :] = pred_y
                #average_val_cost += curr_loss * batch_size
            total_pred += curr_pred
        vote = np.argmax(total_pred, axis = -1)
        accuracy = np.mean(np.equal(vote, np.argmax(val_y, axis = -1)))
        print("Final validation accuracy with %d models: %.3f"%(len(best_models),accuracy))
        #print('Validation loss : %f' % average_val_cost)

if __name__ == "__main__":
    # change mode to "test" for test set evaluation
    is_train_mode = False
    if is_train_mode:
        if not os.path.isfile('train_data.pkl'):
            raw_to_pickle(metadata_path, mode = "train")
        train_x, train_y = load_pickle('train_data.pkl')
        train(train_x, train_y, is_train_mode = True, batch_size = 5, epoch_num = 5, print_every = 100, save_every = 5,
            restore = False, checkpoint_path = './checkpoint')
    evaluate(mode = "valid")