import os
import numpy as np
import pandas as pd
import librosa

'''
Possible which_feature values : stft
                                melspec
                                mfcc
                                cqt
                                spectralcontrast
    feature 마다 함수화 
'''
def compute_chroma_stft(track_ids):
    threshold = 630000

    successful_track_ids = []
    successful_features = []

    for tid in track_ids:
        try:
            filepath = get_audio_path('./music_dataset', tid) # 나중에 경로 다시 변경 같은폴더에 있도록
            x, sr = librosa.load(filepath, sr = 22050, mono=True, duration=29.0)
            x = x.tolist()

            if len(x) < threshold:
                raise ValueError('song length is shorter than threshold')
            else:
                x = x[:threshold]
            x = np.array(x)

            # stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
            # mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
            # del stft
            # f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=50)
            stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
            assert stft.shape[0] == 1 + 2048 // 2
            assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
            del x

            # chroma_stft
            # returns (n_chroma, t)
            f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)


            # print(tid, "is successful")



            successful_track_ids.append(tid)
            successful_features.append(f.tolist())

        except Exception as e:
            print('{}: {}'.format(tid, repr(e)))

    # print('feature shape:', np.shape(successful_features[0]))

    return successful_track_ids, successful_features










##########################################################################

def feature_examples(tid):
    # example of various librosa features
    # please check [https://librosa.github.io/librosa/feature.html]
    threshold = 630000
    try:
        filepath = get_audio_path('music_dataset', tid)
        x, sr = librosa.load(filepath, sr=None, mono=True, duration=29.0)
        x = x.tolist()
        if len(x) < threshold:
            raise ValueError('song length is shorter than threshold')
        else:
            x = x[:threshold]
        x = np.array(x)

        # zero_crossing_rate
        # returns (1,t)
        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7 * 12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1

        # chroma_cqt
        # returns (n_chroma, t)
        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)

        # chroma_cqt
        # returns (n_chroma, t)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        del cqt

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
        del x

        # chroma_stft
        # returns (n_chroma, t)
        f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)

        # rmse
        # returns (1,t)
        f = librosa.feature.rmse(S=stft)

        # spectral_centroid
        # returns (1,t)
        f = librosa.feature.spectral_centroid(S=stft)

        # spectral_bandwidth
        # returns (1,t)
        f = librosa.feature.spectral_bandwidth(S=stft)

        # spectral_contrast
        # returns (n_bands+1, t)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)

        # spectral_rolloff
        # returns (1,t)
        f = librosa.feature.spectral_rolloff(S=stft)

        # mfcc
        # returns (n_mfcc, t)
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        del stft

        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')