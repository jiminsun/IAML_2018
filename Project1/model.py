import tensorflow as tf
import numpy as np
from dataloader import DataLoader
from tensorflow.python.platform import gfile
from time import strftime, localtime, time
from preprocess import raw_to_pickle, load_pickle
from config import H, W, filter_size, n_filter, pool_size, fc_out_dim, weight_decay_rate
from ensemble import _ensemble

def model(X, y, is_training=False):
    n_classes = y.shape[-1]
    n_data, image_height, img_width, _ = X.shape

    n_filter1, n_filter2, n_filter3 = n_filter
    
    activation = tf.nn.relu
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay_rate)
    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(0.0)

    w_out = tf.get_variable("W_out", shape=[fc_out_dim, n_classes], initializer = w_init)
    b_out = tf.get_variable("b_out", shape=[n_classes], initializer=b_init)
    
    conv1_1 = tf.layers.conv2d(X, n_filter1, filter_size, 1, "same", activation = activation, kernel_initializer = w_init)
    conv1_2 = tf.layers.conv2d(conv1_1, n_filter1, filter_size, 1, "same", activation = activation, kernel_initializer = w_init)
    pool1 = tf.layers.max_pooling2d(conv1_2, pool_size, 2,  "same")
    
    conv2_1 = tf.layers.conv2d(pool1, n_filter1, filter_size, 1, "same", activation = activation, kernel_initializer = w_init, kernel_regularizer = regularizer)
    conv2_2 = tf.layers.conv2d(conv2_1, n_filter1, filter_size, 1, "same", activation = activation, kernel_initializer = w_init, kernel_regularizer = regularizer)
    pool2 = tf.layers.max_pooling2d(conv2_2, pool_size, 2,  "same")
    
    conv3_1 = tf.layers.conv2d(pool2, n_filter2, filter_size, 1, "same", activation = activation, kernel_initializer = w_init)
    conv3_2 = tf.layers.conv2d(conv3_1, n_filter2, filter_size, 1, "same", activation = activation, kernel_initializer = w_init)
    pool3 = tf.layers.max_pooling2d(conv3_2, pool_size, 2,  "same")
    
    conv4_1 = tf.layers.conv2d(pool3, n_filter2, filter_size, 1, "same", activation = activation, kernel_initializer = w_init, kernel_regularizer = regularizer)
    conv4_2 = tf.layers.conv2d(conv4_1, n_filter2, filter_size, 1, "same", activation = activation, kernel_initializer = w_init, kernel_regularizer = regularizer)
    pool4 = tf.layers.max_pooling2d(conv4_2, pool_size, 2, "same")
    
    conv5_1 = tf.layers.conv2d(pool4, n_filter3, filter_size, 1, "same", activation = activation, kernel_initializer = w_init)
    conv5_2 = tf.layers.conv2d(conv5_1, n_filter3, filter_size, 1, "same", activation = activation, kernel_initializer = w_init)
    pool5 = tf.layers.max_pooling2d(conv5_2, pool_size, 2, "same")
    
    conv6_1 = tf.layers.conv2d(pool5, n_filter3, filter_size, 1, "same", activation = activation, kernel_initializer = w_init)
    conv6_2 = tf.layers.conv2d(conv6_1, n_filter3, filter_size, 1, "same", activation = activation, kernel_initializer = w_init)
    pool6 = tf.layers.max_pooling2d(conv6_2, pool_size, 2, "same")
    flat = tf.contrib.layers.flatten(pool6)
    
    fc1 = tf.layers.dense(flat, units = fc_out_dim, activation = activation)
    logits = tf.matmul(fc1, w_out) + b_out
    y_pred = tf.nn.softmax(logits, axis = -1)
    return logits, y_pred