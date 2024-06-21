import numpy as np
import time
import librosa
from rich.progress import track
from rich import print

from modules.test import synth
from modules.preprocess import get_audio_frequency_spectrogram
from modules.utils import select_file
from modules.utils.midi import write_notes_to_midi
from modules.utils.scale import get_scale_from_spectrogram

from constant import NOTE_TO_KEY, NOTES


def print_notes_info_to_press(amplitudes, notes_to_press, indexes, open_threshold, scale):
    keys_dict = {note: round(amplitudes[idx]) for note, idx in zip(notes_to_press, indexes)}
    keys_str = ' | '.join(map(
        lambda item: \
            ('[green]' if item[0][:-1] in scale else '[white]') + \
            f'{item[0]:>3} {round(item[1]-open_threshold):>2}' + \
            ('[/green]' if item[0][:-1] in scale else '[/white]'), 
        keys_dict.items()
    ))
    print(f'{keys_str}') # t: {round(times[frame_start+i], 2):.2f} 


@synth
def extract_notes(fs, spectrogram, sr, hop_length, note_start_threshold, listen=False):
    times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr, hop_length=hop_length)
    time_interval = np.mean(np.diff(times))
    
    amplitudes_previous = np.full_like(spectrogram.shape[0], -80)
    indexes_previous = set()
    pressed = set()
    
    pressed_notes = []
    notes = []
    time_from_prev = 0
    
    scale = get_scale_from_spectrogram(spectrogram)
    
    # for i, amplitudes in track(enumerate(spectrogram.T), total=spectrogram.shape[1]):
    for i, amplitudes in enumerate(spectrogram.T):
        open_threshold = max(-60, np.mean(amplitudes) + (np.std(amplitudes) * 1))
        close_threshold = np.mean(amplitudes)

        indexes_initial = set(np.where(amplitudes - amplitudes_previous > note_start_threshold)[0]) & \
                            set(np.where(amplitudes > open_threshold)[0])
        indexes = list(indexes_initial - indexes_previous)
        indexes_previous = set(indexes_initial)
        amplitudes_previous = amplitudes.copy()
        
        notes_to_press = set([NOTES[idx] for idx in indexes])
        keys = set([NOTE_TO_KEY[note] for note in notes_to_press])

        if listen and len(indexes) > 0:
            print_notes_info_to_press(amplitudes, notes_to_press, indexes, open_threshold, scale)
        
        for key, idx in zip(keys, indexes):
            if listen:
                fs.noteon(0, key, round(80+amplitudes[idx]-open_threshold))
            pressed_note = (key, times[i]-time_from_prev)
            pressed_notes.append(pressed_note)
            notes.append(('on', *pressed_note))
            time_from_prev = times[i]

        pressed = pressed.union(indexes)
        
        if listen:
            time.sleep(time_interval*0.9)

        for idx in pressed.copy():
            if spectrogram[idx, i] < close_threshold:
                key = NOTE_TO_KEY[NOTES[idx]]
                if listen:
                    fs.noteoff(0, key)
                idx_pressed_note = filter(lambda i: pressed_notes[i][0] == key, range(len(pressed_notes))).__next__()
                pressed_notes.pop(idx_pressed_note)
                notes.append(('off', key, times[i]-time_from_prev))
                time_from_prev = times[i]
                pressed.remove(idx)

        # print([NOTE_TO_KEY[NOTES[idx]] for idx in pressed])
        
    return notes


if __name__ == '__main__':
    audio_file = select_file('./data/audio')

    n_fft = 8192
    win_length = n_fft
    hop_length = 512
    note_start_threshold = 5

    spectrogram, sr = get_audio_frequency_spectrogram(audio_file, n_fft, win_length, hop_length, optimize=True, plot=True)

    notes = extract_notes(spectrogram, sr, hop_length, note_start_threshold, listen=True)
    write_notes_to_midi(notes, 'output/main_simplify_spec.mid')
