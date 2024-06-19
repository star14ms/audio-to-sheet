import numpy as np
import time
from constant import NOTE_TO_KEY, NOTES
from modules.test import get_synth
from modules.preprocess import get_audio_frequency_spectrogram
import librosa


def synth(func):
    def wrapper(*args, **kwargs):
        fs = get_synth()
        try:
            result = func(fs, *args, **kwargs)
        finally:
            fs.delete()
        return result
    return wrapper


@synth
def extract_notes(fs, spectrogram, sr, hop_length, threshold, close_threshold, note_start_threshold):
    times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr, hop_length=hop_length)

    frame_start = 30
    previous_pitches = np.full_like(spectrogram.shape[0], -80)
    previous_indexes = set()
    pressed = set()
    
    for i, pitches in enumerate(spectrogram[:, frame_start:].T):
        indexes = set(np.where(pitches - previous_pitches > note_start_threshold)[0]) & set(np.where(pitches > threshold)[0])
        indexes = list(indexes - previous_indexes)
        
        # check if the note is continuous
        for j in range(len(indexes)-1, -1, -1):
            if np.mean(spectrogram[i:i+5, indexes[j]]) < threshold - 10:
                indexes = np.delete(indexes, j)

        pressed = pressed.union(indexes)
        notes = set([NOTES[idx] for idx in indexes])
        keys = set([NOTE_TO_KEY[note] for note in notes])
        
        previous_indexes = set(indexes)
        previous_pitches = pitches.copy()

        print(round(times[frame_start+i], 2), [round(pitches[idx], 0) for idx in indexes], [note for note in notes])
        
        for key in keys:
            fs.noteon(0, key, 100)
        
        time.sleep(0.2)
        
        for idx in pressed.copy():
            if spectrogram[i, idx] < close_threshold:
                key = NOTE_TO_KEY[NOTES[idx]]
                fs.noteoff(0, key)
                pressed.remove(idx)


if __name__ == '__main__':
    audio_file = 'data/audio/Yiruma, (이루마) - River Flows in You [7maJOI3QMu0].webm' ### 수정하세요
    # audio_file = "data/Everything's Alright- Laura Shigihara- lyrics [nP-AAlZlCkM].m4a"

    n_fft = 8192
    threshold = -40
    close_threshold = -50
    note_start_threshold = 5
    win_length = 2048
    hop_length = 1024

    spectrogram, sr = get_audio_frequency_spectrogram(audio_file, n_fft, hop_length)
    extract_notes(spectrogram, sr, hop_length, threshold, close_threshold, note_start_threshold)
    