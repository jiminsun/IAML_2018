import pickle
import os
import numpy as np
import tensorflow as tf
from preprocess import raw_to_pickle


def make_tf_record_file(data, data_type, which, split=None):
    """
    Inputs
        param data :
        param data_type : 'train' or 'val'
        split : None to use original data. 
                List of [n, window_size] otherwise, 
                in order to split data into n pieces
                each having a size of (window_size, 20)
                These two values will automatically determine stride, 
                as in convolutional filters in CNN.
    
    Returns
        tf.train.Example instance
    """
    def make_example(track_id, timeseries_data, genre, y):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        #track_id = timeseries_data.astype(np.int64) # check if datatype is string in pkl file
        timeseries_data = timeseries_data.astype(np.float32)
            
        timeseries_data_bytes = tf.compat.as_bytes(timeseries_data.tostring())
        feature = {'id': _int64_feature([track_id]),
                   'x': _bytes_feature([timeseries_data_bytes]),
                   'genre': _int64_feature([genre]),
                   'y': _int64_feature([y])}
        #if data_type != 'train':
        #    feature['track_id'] = _bytes_feature(track_id) # str 이렇게 처리하는게 맞는지.

        features = tf.train.Features(feature = feature)
        example = tf.train.Example(features = features)
        return example

    writer = tf.python_io.TFRecordWriter(
        'data/'+which+'/tfrecord_data/{}_{}.tfrecord'.format(data_type, str(split))
    )
    
    for key, val in data.items():
        track_id = np.int(key[1:-1])
        #print(track_id)
        timeseries_data, genre, track_listens = val
        timeseries_data = np.transpose(timeseries_data)
        
        # This is where one timeseries data is split into n slices.
        if split == None:
            ex = make_example(track_id, timeseries_data, genre, track_listens)
            writer.write(ex.SerializeToString())
        else:
            n = split[0]
            w = split[1]
            start_points = {*range(0, (timeseries_data.shape[0] - w), 
                                   (timeseries_data.shape[0] - w)// n)}
            start_points.add(timeseries_data.shape[0] - w)
            for i in start_points:
                timeseries_slice = timeseries_data[i: i + w, :]
                
                ex = make_example(track_id, timeseries_slice, genre, track_listens)
                writer.write(ex.SerializeToString())
    writer.close()
    print('tfrecord {} made'.format(data_type))

def make_tfrecord_data(pkl_path, data_type, split_option, which):
    """
    :param pkl_path: path of pkl file
    :param data_type: one of train / val / test
    :return: None
    """
    # load train pkl file
    
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    make_tf_record_file(dataset, data_type, which, split = split_option)

def inputs(batch_size, which='stft', split_option = None, test = False, num_parallel_calls=10):
    def make_dataset(file_list):
        dataset = tf.data.TFRecordDataset(file_list)
        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 10)
        return dataset
    
    def decode(serialized_example):
        """Parses an image and label from the given `serialized_example`"""
        parsed = tf.parse_single_example(
            serialized_example,
            features = {
                'id': tf.FixedLenFeature([], tf.int64),
                'x': tf.FixedLenFeature([], tf.string),
                'genre': tf.FixedLenFeature([], tf.int64),
                'y': tf.FixedLenFeature([], tf.int64)
            }
        )
        track_id = tf.cast(parsed['id'], tf.int32)

        x = tf.decode_raw(parsed['x'], tf.float32)
        if which == 'mfcc':
            x = tf.reshape(x, [-1, 20]) # TODO: enter dimension! -> DONE
        elif which == 'cqt':
            x = tf.reshape(x, [-1, 12])
        elif which == 'stft':
            x = tf.reshape(x, [-1, 12])
        y = tf.cast(parsed['y'], tf.int32)

        genre = tf.cast(parsed['genre'], tf.int32)

        return {'x': x, 'y': y, 'genre': genre, 'track_id' : track_id}
    
    
    trn_pkl_path = './train_data_sr22050_chroma_stft.pkl'
    val_pkl_path = './val_data_sr22050_chroma_stft.pkl'
    test_pkl_path = './test_data_sr22050_chroma_stft.pkl'
    

    if not test:
        if not os.path.isfile(trn_pkl_path):
            raw_to_pickle('./track_metadata.csv', mode='train')
        if not os.path.isfile(val_pkl_path):
            raw_to_pickle('./track_metadata.csv', mode='val')

        tfrecord_data_dir = './data/' + which + '/tfrecord_data'
        if not os.path.exists(tfrecord_data_dir):
            os.makedirs(tfrecord_data_dir)
        print('{0}/train_{1}.tfrecord'.format(tfrecord_data_dir, str(split_option)))
        if not os.path.isfile('{0}/train_{1}.tfrecord'.format(tfrecord_data_dir, str(split_option))):
            make_tfrecord_data(trn_pkl_path, 'train', split_option, which)
        if not os.path.isfile('{0}/val_{1}.tfrecord'.format(tfrecord_data_dir, str(split_option))):
            make_tfrecord_data(val_pkl_path, 'val', split_option, which)

        trn_dataset = make_dataset('{0}/train_{1}.tfrecord'.format(tfrecord_data_dir, str(split_option)))
        test_dataset = make_dataset('{0}/val_{1}.tfrecord'.format(tfrecord_data_dir, str(split_option)))
        iterator = tf.data.Iterator.from_structure(trn_dataset.output_types, trn_dataset.output_shapes)

        next_batch = iterator.get_next()

        trn_init_op = iterator.make_initializer(trn_dataset)
        test_init_op = iterator.make_initializer(test_dataset)

        return next_batch, trn_init_op, test_init_op

    if test:
        if not os.path.isfile(test_pkl_path):
            raw_to_pickle('./track_metadata.csv', mode='test')
        tfrecord_data_dir = './data/' + which + '/tfrecord_data'
        if not os.path.exists(tfrecord_data_dir):
            os.makedirs(tfrecord_data_dir)

        if not os.path.isfile('{0}/test_{1}.tfrecord'.format(tfrecord_data_dir, str(split_option))):
            make_tfrecord_data(test_pkl_path, 'test', split_option, which)

        test_dataset = make_dataset('{0}/test_{1}.tfrecord'.format(tfrecord_data_dir, str(split_option)))
        iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
        next_batch = iterator.get_next()
        test_init_op = iterator.make_initializer(test_dataset)
        return next_batch, test_init_op

def test():
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        next_batch, trn_init_op, test_init_op = inputs(32) # batch_size
        sess.run(trn_init_op)
        a = sess.run(next_batch)
        print(a['x'].shape)

if __name__ == '__main__':
    test()