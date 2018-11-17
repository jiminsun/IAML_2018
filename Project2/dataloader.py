import numpy as np
import pandas as pd
import features


class DataLoader():
    def __init__(self, file_path, batch_size, is_training = True):
        '''
        :param file_path: file path for track_metadata.csv
        :param batch_size: batch size
        :param label_column_name: column name of label (project 1: track_genre, project 2: listens)
        :param is_training: training / validation mode
        '''

        self.batch_size = batch_size
        self.token_stream = []
        self.file_path = file_path
        self.is_training = is_training
        # self.label_column_name = label_column_name
        # self.track_genre = 'track_genre'
        # self.track_listens = 'track_listens'
        self.create_batches()

    def create_batches(self):
        '''
        :return: no return
        '''

        self.metadata_df = pd.read_csv(self.file_path)
        if self.is_training:
            self.metadata_df = self.metadata_df[self.metadata_df['split'] == 'training']
        else:
            self.metadata_df = self.metadata_df[self.metadata_df['split'] == 'validation']

        self.num_batch = int(len(self.metadata_df) / self.batch_size)
        self.pointer = 0
        # self.label_dict = {k: v for v, k in enumerate(sorted(set(self.metadata_df[self.label_column_name].values)))}
        # self.label_dict_genre = {k: v for v, k in enumerate(sorted(set(self.metadata_df[self.track_genre].values)))}
        # self.label_dict_listens = {k: v for v, k in enumerate(sorted(set(self.metadata_df[self.track_listens].values)))}

    def next_batch(self):
        '''
        Possible 'which_feature' index
            0 : stft
            1 : melspectrogram
            2 : mfcc 20
            3 : mfcc 50
            4 : cqt


        :return: feature array, label array (one-hot encoded)
        '''

        self.pointer = (self.pointer + 1) % self.num_batch
        start_pos = self.pointer * self.batch_size
        meta_df = self.metadata_df.iloc[start_pos:(start_pos+self.batch_size)]
        # TODO: load features
        track_ids = meta_df['track_id'].values
        # 여기서 feature가져오는 함수 수정
        valid_ids, valid_features = features.compute_chroma_stft(track_ids)

        valid_df = meta_df[meta_df['track_id'].isin(valid_ids)]
        labels_genre, labels_listens = self.convert_labels(valid_df)

        return valid_ids, valid_features, labels_genre, labels_listens


    def reset_pointer(self):
        self.pointer = 0

    def convert_labels(self, meta_df):
        '''
        :param meta_df: metadata (as pandas DataFrame)
        :return: numpy array with (batch_size, number of labels) shape. one-hot encoded
        '''

        label_cols = ['track_genre', 'track_listens']
        label_arrays_list = []
        # label_dict_genre = {k: v for v, k in enumerate(sorted(set(self.metadata_df[self.track_genre].values)))}
        # label_dict_listens = {k: v for v, k in enumerate(sorted(set(self.metadata_df[self.track_listens].values)))}
        
        for col_name in label_cols:
            label_dict = {k: v for v, k in enumerate(sorted(set(self.metadata_df[col_name].values)))}

            label_array = np.zeros((len(meta_df), len(label_dict)))

            labels = meta_df[col_name].values

            for i, label in enumerate(labels):
                label_pos = label_dict.get(label)
                label_array[i, label_pos] = 1

            label_arrays_list.append(label_array)


        # create one-hot encoded array
        # label_array = np.zeros((len(meta_df), len(self.label_dict)))
        # label_array_genre = np.zeros((len(meta_df), len(self.label_dict_genre)))
        # label_array_listens = np.zeros((len(meta_df), len(self.label_dict_listens)))

        # labels = meta_df[self.label_column_name].values
        # labels = meta_df[self.track_genre].values
        # labels = meta_df[self.track_listens].values

        # for i, label in enumerate(labels):
        #     label_pos = self.label_dict.get(label)
        #     label_array[i, label_pos] = 1
        return label_arrays_list[0], label_arrays_list[1] 
        # [0]is genre and [1] is listens


if __name__ == "__main__":
    # test the process
    # training_loader = DataLoader('track_metadata.csv', 32, 'track_genre', is_training=True)
    # validation_loader = DataLoader('track_metadata.csv', 32, 'track_genre', is_training=False)
    training_loader = DataLoader('track_metadata.csv', 32, is_training=True)
    validation_loader = DataLoader('track_metadata.csv', 32, is_training=False) 
    which_feature = ''

    for _ in range(training_loader.num_batch):
        valid_ids, valid_features, labels_genre, labels_listens = training_loader.next_batch(0)
        print('Training loader working..')
        print('valid training ids:', valid_ids)
        print('valid_features: ', np.array(valid_features).shape)
        print('labels_genre: ', labels_genre[:10])
        print('labels_listens:', labels_listens[:10])
        break

    for _ in range(validation_loader.num_batch):
        valid_ids, valid_features, labels_genre, labels_listens = validation_loader.next_batch(0)
        print('Validation loader working..')
        print('valid validation ids:', valid_ids)
        print('valid_features: ', np.array(valid_features).shape)
        print('labels_genre: ', labels_genre[:10])
        print('labels_listens:', labels_listens[:10])
        break