import tensorflow as tf
from dataloader import *
from tensorflow.python.platform import gfile
from time import strftime, localtime, time
from model import *
from utils import *
import argparse

# hyperparameters
# TODO : declare additional Hyperparameters.
parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-learning_rate', type=float, default=0.001)
parser.add_argument('-lr_decay', action='store_true', default=True)
parser.add_argument('-optimizer', type=str, default='adam')
parser.add_argument('-epoch_num', type=int, default=100)
parser.add_argument('-train_test_split_ratio', type=float, default=0.8)
parser.add_argument('-model', type=str, default='crnn')
parser.add_argument('-rnn_hidden_dim', type=int, default=64)
parser.add_argument('-input_dim', type=int, default=64)
parser.add_argument('-multi_rnn', action='store_true', default=False)
parser.add_argument('-restore', action='store_true', default=False)
parser.add_argument('-n_epoch_no_imprv', type=int, default=10)

args = parser.parse_args()

global_step = tf.Variable(0, trainable=False, name='global_step')
if args.lr_decay:
    #print("Learning rate decay applied")
    starter_learning_rate = args.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step, 500, 0.97, staircase=True)
else:
    learning_rate = args.learning_rate


# fixed
data_path = 'data'
feature_path = data_path + '/features.pkl'
chord_path = data_path + '/chords.pkl'
# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
is_train_mode = False
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint/crnn_64_run-1126-2009'
if is_train_mode:
    checkpoint_path = 'checkpoint'

# make xs and ys

# if there is no feature file, make it
if gfile.Exists(feature_path):
    print('Load feature file')
    with open(feature_path, 'rb') as f:
        valid_wav_list, features_list = pickle.load(f)
else:
    print('Make feature file')
    wav_list = get_wav_list(data_path, chord_path)
    valid_wav_list, features_list = wav_list_to_feature_list(wav_list)
    with open(feature_path, 'wb') as f:
        pickle.dump((valid_wav_list, features_list), f)

wav_list = valid_wav_list
# onehot encoding (total 25 chords)
label_list = get_label_list(chord_path, valid_wav_list)
encoded_label_list = [encode_label(labels) for labels in label_list]
encoded_label_list = np.array(encoded_label_list)

# split train and test
split = int(len(wav_list)*args.train_test_split_ratio)

# TODO : Build model
# build dataset
dataset = tf.data.Dataset.from_tensor_slices((features_list, encoded_label_list))
train_dataset = dataset.take(split)
test_dataset = dataset.skip(split)

train_dataset = train_dataset.shuffle(buffer_size=len(wav_list) - split)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.prefetch(args.batch_size)
train_dataset = train_dataset.repeat(args.epoch_num)

test_dataset = test_dataset.shuffle(buffer_size=split)
test_dataset = test_dataset.batch(args.batch_size)
test_dataset = test_dataset.prefetch(args.batch_size)
test_dataset = test_dataset.repeat(1)

iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init = iter.make_initializer(train_dataset)
test_init = iter.make_initializer(test_dataset)

# batch of features, batch of labels
X, Y = iter.get_next()
# print(X.shape, Y.shape)

#model = CNN_Model(batch_size=args.batch_size)
#model_name = args.model

model = CRNN_Model(hidden_dim=args.rnn_hidden_dim, input_len=args.input_dim,
                   batch_size=args.batch_size)
model_name = args.model + '_' + str(args.rnn_hidden_dim) + '_'

logits, cost = model.build_graph(X, Y, is_train_mode)

optimizer = get_optim(args.optimizer)(learning_rate=learning_rate)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(cost, global_step=global_step)
infer = tf.argmax(logits, axis=-1)
answer = tf.argmax(Y, axis=-1)

# calculate accuracy
correct_prediction = tf.equal(infer, answer)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train and evaluate
with tf.Session() as sess:
    if is_train_mode:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Load model from : %s' % checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        print('Start training')
        train_total_batch = int(split / args.batch_size)
        #sess.run(train_init)
        best_score = 0.0
        n_epoch_no_imprv = 0 # for early stopping

        for epoch in range(args.epoch_num):
            # TODO: do some train step code here
            print('------------------- epoch:', epoch, ' -------------------')
            sess.run(train_init)
            is_train_mode = True
            for _ in range(train_total_batch):
                c, _, acc = sess.run([cost, train_step, accuracy])
                print('Step: %5d, ' % sess.run(global_step), ' Cost: %.4f ' % c,
                      ' Accuracy: %.4f ' % acc)

             # TODO : do accuracy test
             # TODO : implement early-stopping

            test_total_batch = int((len(wav_list) - split) / args.batch_size)
            sess.run(test_init)
            val_acc = 0.0
            is_train_mode = False
            for _ in range(test_total_batch):
                val_acc += sess.run(accuracy)
            print('Test accuracy: %.4f' % (val_acc / test_total_batch))

            if val_acc >= best_score:
                n_epoch_no_imprv = 0
                if val_acc / test_total_batch > 0.85:
                    final_path = checkpoint_path + '/' + model_name + 'run-%02d%02d-%02d%02d/' % tuple(localtime(time()))[1:5]
                    if not gfile.Exists(final_path):
                        gfile.MakeDirs(final_path)
                    saver.save(sess, final_path, global_step=global_step)
                    print('Model saved in file : %s' % final_path)
                best_score = val_acc
            else:
                n_epoch_no_imprv += 1
                if n_epoch_no_imprv == args.n_epoch_no_imprv:
                    print("- early stopping {} epochs without improvement".format(n_epoch_no_imprv))
                    break
        print('Training finished!')

    else:
        print("Start evaluating..")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        test_total_batch = int((len(wav_list) - split) / args.batch_size)
        sess.run(test_init)
        acc = 0.0
        for _ in range(test_total_batch):
            acc += sess.run(accuracy)
        print('Test accuracy: %.4f' % (acc / test_total_batch))
