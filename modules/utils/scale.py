import numpy as np
from constant import SCALES, KYE_DICT_ON_PIANO


def get_key_signiture_from_spectrogram(spectrogram):
    '''return key signiture of the spectrogram'''
    n_times_in_audio = (spectrogram+80).sum(axis=1)
    
    var_of_keys_dB = {
        'C': n_times_in_audio[KYE_DICT_ON_PIANO['C']].var(),
        'C#': n_times_in_audio[KYE_DICT_ON_PIANO['C#']].var(),
        'D': n_times_in_audio[KYE_DICT_ON_PIANO['D']].var(),
        'D#': n_times_in_audio[KYE_DICT_ON_PIANO['D#']].var(),
        'E': n_times_in_audio[KYE_DICT_ON_PIANO['E']].var(),
        'F': n_times_in_audio[KYE_DICT_ON_PIANO['F']].var(),
        'F#': n_times_in_audio[KYE_DICT_ON_PIANO['F#']].var(),
        'G': n_times_in_audio[KYE_DICT_ON_PIANO['G']].var(),
        'G#': n_times_in_audio[KYE_DICT_ON_PIANO['G#']].var(),
        'A': n_times_in_audio[KYE_DICT_ON_PIANO['A']].var(),
        'A#': n_times_in_audio[KYE_DICT_ON_PIANO['A#']].var(),
        'B': n_times_in_audio[KYE_DICT_ON_PIANO['B']].var(),
    }
    
    var_of_dB_scales = {
        key_signiture: np.mean([var_of_keys_dB[key] for key in SCALES[key_signiture]])
        for key_signiture in ([
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'
        ])
    }
    
    return max(var_of_dB_scales, key=var_of_dB_scales.get)


def get_scale_from_spectrogram(spectrogram):
    '''return the scale of the spectrogram'''
    key_signiture = get_key_signiture_from_spectrogram(spectrogram)
    return SCALES[key_signiture]


def get_scale(key_signiture):
    '''return the scale of the key signiture'''
    return SCALES[key_signiture]
