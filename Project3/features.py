import os
import numpy as np
import pandas as pd
import librosa

def compute_chroma_stft(path):
    threshold = 630000
    features = []
    try:
        x, sr = librosa.load(path, sr=44100, mono=True)
        x = x.tolist()

        #if len(x) < threshold:
        #    raise ValueError('song length is shorter than threshold')
        #else:
        #    x = x[:threshold]
        x = np.array(x)

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
        del x

            # chroma_stft
            # returns (n_chroma, t)
        f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
        features.append(np.transpose(f).tolist())

    except Exception as e:
        print('{}: {}'.format(path, repr(e)))
        return None

    return features


def compute_melspec(path):
    features = []
    try:
        x, sr = librosa.load(path, sr=44100, mono=True)
        x = x.tolist()
        x = np.array(x)
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        del x
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        features.append(np.transpose(mel).tolist())

    except Exception as e:
        print('{}: {}'.format(path, repr(e)))
        return None

    return features


def compute_mfcc_example(path):
    features = []
    try:
        x, sr = librosa.load(path, sr=44100, mono=True)

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        features.append(np.transpose(f).tolist())

    except Exception as e:
        print('{}: {}'.format(path, repr(e)))
        return None

    return features


def feature_examples(filepath):
    # example of various librosa features
    # please check [https://librosa.github.io/librosa/feature.html]
    threshold = 630000
    try:
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
        print('{}: {}'.format(filepath, repr(e)))


