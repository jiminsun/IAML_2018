import pickle
import pandas as pd
import numpy as np
from dataloader import DataLoader



def raw_to_pickle(metadata_path, mode = 'train'):
    metadata = pd.read_csv(metadata_path, encoding = 'utf-8')
    train_size = sum(metadata['split'] == 'training')
    val_size =  sum(metadata['split'] != 'training')
    n_classes = 8 
    label_column_name = 'track_genre'
    train_batchsize = 1
    val_batchsize = 1
    if mode == 'train':
        train_dataloader = DataLoader(file_path=metadata_path, batch_size=train_batchsize,
                                      label_column_name=label_column_name, is_training=True)
        train_dataset = []
        train_label = []
        for i in range(train_size):
            train_x, train_y = train_dataloader.next_batch()
            try:
                if len(train_x):
                    train_dataset.append(np.array(train_x))
                    train_label.append(np.array(train_y))
            except Exception as e:
                print("####### Error at iteration %d"%i)
            if i % 100 == 0:
                print("Training data %d: finished!"%i)
        dataset = {'x': np.expand_dims(np.squeeze(np.array(train_dataset)), -1),
                   'y': np.squeeze(np.array(train_label))}
    else:
        val_dataset = []
        val_label = []

        validation_loader = DataLoader(file_path=metadata_path, batch_size=val_batchsize,
                            label_column_name=label_column_name, is_training=False)
        for i in range(val_size):
            val_x, val_y = validation_loader.next_batch()
            try:
                if len(val_x):
                    val_dataset.append(np.array(val_x))
                    val_label.append(np.array(val_y))
            except Exception as e:
                print("####### Error at iteration %d"%i)
            if i % 100 == 0:
                print("Validation data %d: finished!"%i)
        dataset = {'x': np.expand_dims(np.squeeze(np.array(val_dataset)), -1),
                   'y': np.squeeze(np.array(val_label))}
    print("Preprocessing Done!")
                   
    with open(mode+'_data.pkl', 'wb') as file:
        pickle.dump(dataset, file)

def load_pickle(pickle_dir, sample_n = None):
    with open(pickle_dir, 'rb') as file:
        data = pickle.load(file)
    if sample_n:
        print("Loading data of size %d .."%(sample_n))
        return data['x'][:sample_n, :], data['y'][:sample_n]
    print("Loading data of size %d .."%(len(data['y'])))
    return data['x'], data['y']