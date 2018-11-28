import tensorflow as tf


def get_optim(optimizer_type):
    if optimizer_type == 'rms':
        return tf.train.RMSPropOptimizer
    else:
        return tf.train.AdamOptimizer

def decode_chords(chords, idx_to_chord):
    if chords.ndim == 1:
        T = chords.shape[0]
        N = 1
        assert T == 96
    else:
        N, T = chords.shape

    decoded = []
    for i in range(N):
        chord_per_song = []
        for t in range(T):
            if chords.ndim == 1:
                chord = idx_to_chord[chords[t]]
            else:
                chord = idx_to_chord[chords[i, t]]
            chord_per_song.append(chord)
        decoded.append(' '.join(chord_per_song))
    return decoded

def chord_idx_mapping():
    roots = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    chords_type = ['', 'm']
    chords = [x+y for x in roots for y in chords_type]
    chords.append('-')
    chord2idx = {k: v for v, k in enumerate(sorted(set(chords)))}     # total 25 chords
    idx2chord = {v: k for k, v in chord2idx.items()}
    return chord2idx, idx2chord