import numpy as np
import time
from constant import NOTE_TO_KEY, NOTES
from modules.test import synth
from modules.preprocess import get_audio_frequency_spectrogram
from modules.utils.midi import write_notes_to_midi
from rich.progress import track
import librosa


def print_notes_info_to_press(amplitudes, notes_to_press, indexes):
    keys_dict = {note: round(amplitudes[idx]) for note, idx in zip(notes_to_press, indexes)}
    keys_str = ' | '.join(map(lambda item: f'{item[0]:>3} {item[1]-open_threshold:>2}', keys_dict.items()))
    print(f'{keys_str}') # t: {round(times[frame_start+i], 2):.2f} 


@synth
def extract_notes(fs, spectrogram, sr, hop_length, open_threshold, close_threshold, note_start_threshold):
    times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr, hop_length=hop_length)
    time_interval = np.mean(np.diff(times))
    
    frame_start = 20
    amplitudes_previous = np.full_like(spectrogram.shape[0], -80)
    indexes_previous = set()
    pressed = set()
    
    pressed_notes = []
    notes = []
    time_from_prev = 0
    
    for i, amplitudes in track(enumerate(spectrogram[:, frame_start:].T), total=spectrogram.shape[1]-frame_start):
        indexes_initial = set(np.where(amplitudes - amplitudes_previous > note_start_threshold)[0]) & \
                            set(np.where(amplitudes > open_threshold)[0])
        indexes = list(indexes_initial - indexes_previous)
        indexes_previous = set(indexes_initial)
        amplitudes_previous = amplitudes.copy()
        
        notes_to_press = set([NOTES[idx] for idx in indexes])
        keys = set([NOTE_TO_KEY[note] for note in notes_to_press])
        
        print_notes_info_to_press(amplitudes, notes_to_press, indexes)
        
        for key in keys:
            fs.noteon(0, key, 100)
            pressed_note = (key, times[frame_start+i]-time_from_prev)
            pressed_notes.append(pressed_note)
            notes.append(('on', *pressed_note))
            time_from_prev = times[frame_start+i]

        pressed = pressed.union(indexes)
        
        time.sleep(time_interval)
        
        for idx in pressed.copy():
            if spectrogram[idx, i] < close_threshold:
                key = NOTE_TO_KEY[NOTES[idx]]
                fs.noteoff(0, key)
                
                idx_pressed_note = filter(lambda i: pressed_notes[i][0] == key, range(len(pressed_notes))).__next__()
                pressed_notes.pop(idx_pressed_note)
                notes.append(('off', key, times[frame_start+i]-time_from_prev))
                time_from_prev = times[frame_start+i]
                pressed.remove(idx)
        # print(pressed)
        
    return notes


if __name__ == '__main__':
    audio_file = 'data/audio/Yiruma, (이루마) - River Flows in You [7maJOI3QMu0].webm' ### 수정하세요
    # audio_file = "data/audio/Everything's Alright- Laura Shigihara- lyrics [nP-AAlZlCkM].m4a"

    n_fft = 8192
    open_threshold = -40
    close_threshold = -50
    note_start_threshold = 5
    win_length = 2048
    hop_length = 1024

    spectrogram, sr = get_audio_frequency_spectrogram(audio_file, n_fft, hop_length, optimize=True)
    # spectrogram = spectrogram[:, :spectrogram.shape[1] // 40]

    notes = extract_notes(spectrogram, sr, hop_length, open_threshold, close_threshold, note_start_threshold)
    write_notes_to_midi(notes, 'output/main_simplify_spec.mid')
