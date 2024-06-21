import numpy as np
import time
import librosa
from rich.progress import track
from rich import print

from modules.test import synth
from modules.preprocess import get_simplified_frequency_spectrogram
from modules.utils.midi import write_notes_to_midi
from modules.utils import select_file
from modules.utils.scale import get_scale_from_spectrogram
from modules.constants import NOTE_TO_KEY, NOTES


def print_notes_info_to_press(amplitudes, notes_to_press, idxs_to_press, open_threshold, scale, times, i, amplitudes_prev):
    keys_dict = {note: (amplitudes_prev[idx], amplitudes[idx]) for note, idx in zip(notes_to_press, idxs_to_press)}
    keys_str = ' | '.join(map(
        lambda item: \
            ('[green]' if item[0][:-1] in scale else '[white]') + \
            f'{item[0]:>3} {round(item[1][0]-open_threshold)}>{round(item[1][1]-open_threshold):>2}' + \
            ('[/green]' if item[0][:-1] in scale else '[/white]'), 
        keys_dict.items()
    ))
    # print(f'{round(times[i], 2):.2f} s: {keys_str}')
    print(f'{keys_str}')


@synth
def extract_notes(fs, spectrogram, sr, hop_length, open_threshold_weight, listen=False, speed=1.0):
    notes = []
    times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr, hop_length=hop_length)
    time_interval = hop_length / sr

    if listen:
        scale = get_scale_from_spectrogram(spectrogram)
        
    amplitudes_prev = np.full(spectrogram.shape[0], -80)[None]
    indexes_prev = set()
    time_last_updated = 0
    pressed = set()
    
    # for i, amplitudes in enumerate(spectrogram.T):
    for i, amplitudes in track(enumerate(spectrogram.T), total=spectrogram.shape[1]):
        # thresholds changed dynamically
        open_threshold = np.mean(amplitudes) + np.std(amplitudes) * open_threshold_weight
        close_threshold = np.mean(amplitudes)

        # determine which notes to press and unpress
        indexes_to_press_initial = set(np.where(amplitudes - amplitudes_prev > 10)[0]) & \
                                    set(np.where(amplitudes > open_threshold)[0])
            
        idxs_to_press = list(indexes_to_press_initial - indexes_prev)

        # check if the note is continuous
        for j in range(len(idxs_to_press)-1, -1, -1):
            if np.mean(spectrogram[idxs_to_press[j], i+1:i+round(1/time_interval)]) < close_threshold:
                print(f'Note {NOTES[idxs_to_press[j]]} is not continuous')
                idxs_to_press = np.delete(idxs_to_press, j)

        pressed = pressed.union(idxs_to_press)
        idxs_to_unpress = [idx for idx in pressed if spectrogram[idx, i] < close_threshold]
        keys_to_unpress = [NOTE_TO_KEY[NOTES[idx]] for idx in idxs_to_unpress]
        pressed.difference_update(idxs_to_unpress)
        
        notes_to_press = set([NOTES[idx] for idx in idxs_to_press])
        keys = set([NOTE_TO_KEY[note] for note in notes_to_press])

        # extract notes
        for key, idx in zip(keys, idxs_to_press):
            notes.append(('on', key, times[i]-time_last_updated))
            time_last_updated = times[i]
        for key in keys_to_unpress:
            notes.append(('off', key, times[i]-time_last_updated))
            time_last_updated = times[i]

        if listen:
            if len(idxs_to_press) > 0:
                print_notes_info_to_press(amplitudes, notes_to_press, idxs_to_press, open_threshold, scale, times, i, amplitudes_prev)
        
            for key, idx in zip(keys, idxs_to_press):
                fs.noteon(0, key, round(120+amplitudes[idx]))

            time.sleep(time_interval/speed)

            for key in keys_to_unpress:
                fs.noteoff(0, key)

        indexes_prev = set(indexes_to_press_initial)
        amplitudes_prev = amplitudes.copy()
        # print([NOTE_TO_KEY[NOTES[idx]] for idx in pressed])
        
    return notes


if __name__ == '__main__':
    audio_file = select_file('./data/audio')

    n_fft = 2048
    win_length = n_fft
    hop_length = 512
    open_threshold_weight = 1.2

    start_frame = 0
    speed = 1.2
    listen = True

    spectrogram, sr = get_simplified_frequency_spectrogram(audio_file, n_fft, win_length, hop_length, optimize=True, plot=False)
    notes = extract_notes(spectrogram[:, start_frame:], sr, hop_length, open_threshold_weight, listen, speed)
    write_notes_to_midi(notes, 'output/main_simplify_spec.mid')
