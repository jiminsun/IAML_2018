from os import path
import pickle
import pandas as pd
import numpy as np
from dataloader import DataLoader


def raw_to_pickle(metadata_path, mode = 'train'):
    '''
        inputs:
            - metadata_path
            - mode : 
                - train
                - test/val
            
        output : pickle file containing a dictionary in the form of
            { 'track id' : (feature (np array), genre(int), track_listens(int), ) }
            {스트링 :  튜플}
            튜플 애들 (어레이, 인트, 인트)
            {}
    '''
    metadata = pd.read_csv(metadata_path, encoding = 'utf-8')
    train_size = sum(metadata['split'] == 'training')
    val_size =  sum(metadata['split'] == 'validation')
    test_size = sum(metadata['split'] == 'test')

    train_batchsize = 1
    val_batchsize = 1
    test_batchsize = 1

    
    if mode == 'train':
        dataset = {}
        train_dataloader = DataLoader(file_path=metadata_path, batch_size=train_batchsize, is_training=mode)
        for i in range(train_size):
            train_id, train_feature, train_genre, train_listens = train_dataloader.next_batch(which_feature)
            try : 
                if len(train_feature): # length가 0이면 false
                    # train_tuple = (train_feature[0], train_genre[0], train_listens[0])
                    int_genre = np.argmax(train_genre)
                    int_listens = np.argmax(train_listens)

                    train_tuple = (train_feature[0], int_genre, int_listens)
                    idtostr = str(train_id)
                    dataset[idtostr] = train_tuple # add each music item to dataset dictionary {'id1' = (feature1, genre1, listen1), 'id2' = (feature2, genre2, listen2)}
            except Exception as e:
                # print("####### Error at iteration %d"%i)
                print("####### Error at iteration %d"%i, e, "/n", 'train_genre:', train_genre, ',train_listens:', train_listens)
            if i % 100 == 0:
                print("Training data %d: finished!"%i)
                print("current dataset(dictionary) length:",len(dataset))

    

    elif mode == 'val':

        dataset = {}

        validation_loader = DataLoader(file_path=metadata_path, batch_size=val_batchsize, is_training=mode)
        for i in range(val_size):
            val_id, val_feature, val_genre, val_listens = validation_loader.next_batch(which_feature)
            try : 
                # print('length val_feature:', len(val_feature))
                # print('val_feature[0]', val_feature[0])
                if len(val_feature): # length가 0이면 false
                    int_genre = np.argmax(val_genre)
                    int_listens = np.argmax(val_listens)
                    # print(len(val_feature))
                    # val_tuple = (val_feature[0], val_genre[0], val_listens[0]) # list가 [[우리원하는데이터]] 이렇게 한번 더 감싸서 가져오길래 괄호하나 빼게함
                    val_tuple = (val_feature[0], int_genre, int_listens) # list가 [[우리원하는데이터]] 이렇게 한번 더 감싸서 가져오길래 괄호하나 빼게함
                    idtostr = str(val_id)
                    dataset[idtostr] = val_tuple # add each music item to dataset dictionary {'id1' = (feature1, genre1, listen1), 'id2' = (feature2, genre2, listen2)}
                                        
            except Exception as e:
                print("####### Error at iteration %d"%i, e, "/n", 'train_genre:', val_genre, ',train_listens:', val_listens)
            if i % 100 == 0:
                print("Validation data %d: finished!"%i)
                print('current dataset(dictionary) length:',len(dataset))
    else:
        dataset = {}
        test_loader = DataLoader(file_path=metadata_path, batch_size=test_batchsize, is_training='test')
        for i in range(test_size):
            test_id, test_feature, test_genre, test_listens = test_loader.next_batch()
            try:
                # print('length val_feature:', len(val_feature))
                # print('val_feature[0]', val_feature[0])
                if len(test_feature):  # length가 0이면 false
                    int_genre = np.argmax(test_genre)
                    int_listens = np.argmax(test_listens)
                    test_tuple = (test_feature[0], int_genre, int_listens)  # list가 [[우리원하는데이터]] 이렇게 한번 더 감싸서 가져오길래 괄호하나 빼게함
                    idtostr = str(test_id)
                    dataset[idtostr] = test_tuple  # add each music item to dataset dictionary {'id1' = (feature1, genre1, listen1), 'id2' = (feature2, genre2, listen2)}

            except Exception as e:
                print("####### Error at iteration %d" % i, e, "/n", 'genre:', test_genre, ',track_listens:',
                      test_listens)
            if i % 100 == 0:
                print("Test data %d: finished!" % i)
                print('current dataset(dictionary) length:', len(dataset))


    print("Preprocessing Done!")

    with open(mode + '_data' + '_sr22050' + f'_' + 'chroma_stft'+'.pkl' , 'wb') as file:
        pickle.dump(dataset, file)



if __name__ == "__main__":
    metadata_path = './track_metadata.csv'
