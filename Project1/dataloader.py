import numpy as np
import pandas as pd
import features


class DataLoader():
    def __init__(self, file_path, batch_size, label_column_name, is_training = True):
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
        self.label_column_name = label_column_name
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
        self.label_dict = {k: v for v, k in enumerate(sorted(set(self.metadata_df[self.label_column_name].values)))}

    def next_batch(self):
        '''
        :return: feature array, label array (one-hot encoded)
        '''

        self.pointer = (self.pointer + 1) % self.num_batch
        start_pos = self.pointer * self.batch_size
        meta_df = self.metadata_df.iloc[start_pos:(start_pos+self.batch_size)]
        # TODO: load features
        track_ids = meta_df['track_id'].values
        valid_ids, valid_features = features.compute_mfcc_example(track_ids)
        valid_df = meta_df[meta_df['track_id'].isin(valid_ids)]
        return valid_features, self.convert_labels(valid_df)

    def reset_pointer(self):
        self.pointer = 0

    def convert_labels(self, meta_df):
        '''
        :param meta_df: metadata (as pandas DataFrame)
        :return: numpy array with (batch_size, number of labels) shape. one-hot encoded
        '''

        # create one-hot encoded array
        label_array = np.zeros((len(meta_df), len(self.label_dict)))
        labels = meta_df[self.label_column_name].values
        for i, label in enumerate(labels):
            label_pos = self.label_dict.get(label)
            label_array[i, label_pos] = 1
        return label_array


if __name__ == "__main__":
    # test the process
    training_loader = DataLoader('track_metadata.csv', 32, 'track_genre', is_training=True)
    validation_loader = DataLoader('track_metadata.csv', 32, 'track_genre', is_training=False)

    for _ in range(training_loader.num_batch):
        track_features, label_onehot = training_loader.next_batch()
        print(np.array(track_features).shape)
        print(label_onehot[:10])
        break

    for _ in range(validation_loader.num_batch):
        track_features, label_onehot = validation_loader.next_batch()
        print(np.array(track_features).shape)
        print(label_onehot[:10])
        break
