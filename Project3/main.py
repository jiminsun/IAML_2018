import tensorflow as tf
import numpy as np
from dataloader import *
from tensorflow.python.platform import gfile
from time import strftime, localtime, time

# hyperparameters
# TODO : declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
batch_size = 32
epoch_num = 5
learning_rate = 0.001


# fixed
data_path = 'data'
feature_path = data_path + '/features.pkl'
chord_path = data_path + '/chords.pkl'
# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
is_train_mode = True
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint'


# make xs and ys
wav_list = get_wav_list(data_path, chord_path)

# if there is no feature file, make it
if gfile.Exists(feature_path):
    print('Load feature file')
    with open(feature_path, 'rb') as f:
        valid_wav_list, features_list = pickle.load(f)
else:
    print('Make feature file')
    valid_wav_list, features_list = wav_list_to_feature_list(wav_list)
    with open(feature_path, 'wb') as f:
        pickle.dump((valid_wav_list, features_list), f)

# onehot encoding (total 25 chords)
label_list = get_label_list(chord_path, valid_wav_list)
encoded_label_list = [encode_label(labels) for labels in label_list]
encoded_label_list = np.array(encoded_label_list)

# split train and test
split = int(len(wav_list)*0.8)

# build dataset
dataset = tf.data.Dataset.from_tensor_slices((features_list, encoded_label_list))
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=len(wav_list))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(batch_size)

# train/test data is different each time you run main.py
test_dataset = dataset.take(split)
train_dataset = dataset.skip(split)

iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
train_init = iter.make_initializer(train_dataset)
test_init = iter.make_initializer(test_dataset)

# batch of features, batch of labels
X, Y = iter.get_next()


# TODO : build your model here
global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.reshape(X, [-1, 1034, 20, 1])

conv = tf.layers.conv2d(X, filters=5, kernel_size=[84, 1], strides=10,
                       padding='VALID', activation=tf.nn.relu)
conv = tf.reshape(conv, [-1, 96*2*5])
dense = tf.layers.dense(conv, units=96*25, activation=tf.nn.relu)
dropout = tf.layers.dropout(dense, rate=0.5, training=is_train_mode)
logits = tf.reshape(dropout, [-1, 96, 25])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y, dim=2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

infer = tf.argmax(logits, axis=2)
answer = tf.argmax(Y, axis=2)

# calculate accuracy
correct_prediction = tf.equal(infer, answer)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train and evaluate
with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Load model from : %s' % checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    if is_train_mode:
        print('Start training')
        train_total_batch = int(split / batch_size)
        total_epoch = 0
        sess.run(train_init)
        for epoch in range(epoch_num):
            # TODO: do some train step code here
            print('------------------- epoch:', epoch, ' -------------------')
            for _ in range(train_total_batch):
                c, _, acc = sess.run([cost, optimizer, accuracy])
                print('Step: %5d, ' % sess.run(global_step), ' Cost: %.4f ' % c,
                      ' Accuracy: %.4f ' % acc)

        print('Training finished!')

        # TODO : do accuracy test
        test_total_batch = int((len(wav_list) - split) / batch_size)
        sess.run(test_init)
        acc = 0.0
        for _ in range(test_total_batch):
            acc += accuracy.eval()

        print('Test accuracy: %.4f' % (acc/test_total_batch))

        # save checkpoint
        final_path = checkpoint_path + '/run-%02d%02d-%02d%02d/' % tuple(localtime(time()))[1:5]
        if not gfile.Exists(final_path):
            gfile.MakeDirs(final_path)
        saver.save(sess, final_path, global_step=global_step)
        print('Model saved in file : %s' % final_path)

    else:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # ...
