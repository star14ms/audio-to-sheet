import numpy as np
from audio2midi.constants import SCALES, KYE_DICT_ON_PIANO


def get_key_signiture_from_spectrogram(spectrogram):
    '''return key signiture of the spectrogram'''
    var_of_88_notes = (spectrogram+80).var(axis=1)
    
    var_of_keys_dB = {
        'C': var_of_88_notes[KYE_DICT_ON_PIANO['C']].mean(),
        'C#': var_of_88_notes[KYE_DICT_ON_PIANO['C#']].mean(),
        'D': var_of_88_notes[KYE_DICT_ON_PIANO['D']].mean(),
        'D#': var_of_88_notes[KYE_DICT_ON_PIANO['D#']].mean(),
        'E': var_of_88_notes[KYE_DICT_ON_PIANO['E']].mean(),
        'F': var_of_88_notes[KYE_DICT_ON_PIANO['F']].mean(),
        'F#': var_of_88_notes[KYE_DICT_ON_PIANO['F#']].mean(),
        'G': var_of_88_notes[KYE_DICT_ON_PIANO['G']].mean(),
        'G#': var_of_88_notes[KYE_DICT_ON_PIANO['G#']].mean(),
        'A': var_of_88_notes[KYE_DICT_ON_PIANO['A']].mean(),
        'A#': var_of_88_notes[KYE_DICT_ON_PIANO['A#']].mean(),
        'B': var_of_88_notes[KYE_DICT_ON_PIANO['B']].mean(),
    }
    
    var_of_dB_scales = {
        key_signiture: np.mean([var_of_keys_dB[key] for key in SCALES[key_signiture]])
        for key_signiture in ([
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'
        ])
    }
    
    key_signitures = [key for key in var_of_dB_scales if var_of_dB_scales[key] == max(var_of_dB_scales.values())]
    key_signiture = max(map(lambda key: key.replace('m', ''), key_signitures), key=var_of_keys_dB.get)
    return key_signiture


def get_scale_from_spectrogram(spectrogram):
    '''return the scale of the spectrogram'''
    key_signiture = get_key_signiture_from_spectrogram(spectrogram)
    return SCALES[key_signiture]


def get_scale(key_signiture):
    '''return the scale of the key signiture'''
    return SCALES[key_signiture]
