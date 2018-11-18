import numpy as np
import features
import os
import pickle

roots = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
chords_type = ['', 'm']
chords = [x+y for x in roots for y in chords_type]
chords.append('-')
chords_dict = {k: v for v, k in enumerate(sorted(set(chords)))}     # total 25 chords


def wav_list_to_feature_list(wav_list):
    '''
    :param wav_list: lsit of the wav files
    :return: list of valid wav files and features
    '''
    valid_wav_list = []
    features_list = []
    for wav_path in wav_list:
        feature = features.compute_mfcc_example(wav_path)
        if feature is not None:
            valid_wav_list.append(wav_path)
            features_list.append(feature)
        else:
            print(wav_path)

    return valid_wav_list, features_list


def get_wav_list(data_path, chord_path):
    '''
    :param data_path: data directory
    :param data_path: chord dictionary path
    :return: list of wav file paths
    '''
    paths = []
    with open(chord_path, 'rb') as f:
        chords = pickle.load(f)
    for path, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav') and len(chords[file]) == 96:
                paths.append(path + '/' + file)

    return paths


def get_label_list(chord_path, wav_paths):
    '''
    :param chord_path: path of the chord info file
    :param wav_paths: paths of wav files
    :return: list of labels of wav files (each wav file has 96 chords)
    '''
    label_list = []
    with open(chord_path, 'rb') as f:
        chord_dict = pickle.load(f)

    for path in wav_paths:
        wav_name = path[path.rfind('/')+1:]   # Maybe need to change '/' to another character

        label_list.append(chord_dict[wav_name])

    return label_list


def encode_label(labels):
    '''
    :param labels: labels from the same wav file
    :return: numpy array with (96, number of labels) shape, one-hot encoded
    '''

    # create one-hot encoded array
    label_array = np.zeros((len(labels), len(chords_dict)), dtype=np.int32)
    for i, label in enumerate(labels):
        label_pos = chords_dict[label]
        label_array[i, label_pos] = 1
    return label_array




