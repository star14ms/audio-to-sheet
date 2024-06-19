import numpy as np
import time
from constant import NOTE_TO_KEY
from modules.test import synth
from modules.preprocess import get_audio_frequency_spectrogram
import librosa


@synth
def extract_notes(fs, spectrogram, sr, n_fft, hop_length, threshold, close_threshold, note_start_threshold):
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    specidx_to_note = [librosa.hz_to_note(frequency).replace('♯', '#') for frequency in frequencies[2:]]
    times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr, hop_length=hop_length)

    frame_start = 30
    previous_pitches = np.full_like(spectrogram.shape[0], -80)
    previous_indexes = set()
    pressed = set()
    
    for i, pitches in enumerate(spectrogram[2:, frame_start:].T):
        indexes_initial = set(np.where(pitches - previous_pitches > note_start_threshold)[0]) & set(np.where(pitches > threshold)[0])
        previous_indexes = set(indexes_initial)
        indexes = list(indexes_initial - previous_indexes)
        
        # # check if the note is continuous
        # for j in range(len(indexes)-1, -1, -1):
        #     if np.mean(spectrogram[indexes[j], i:i+2]) < close_threshold:
        #         indexes = np.delete(indexes, j)

        pressed = pressed.union(indexes)
        notes = set([specidx_to_note[idx] for idx in indexes])
        keys = set([NOTE_TO_KEY[note] for note in notes])
        
        previous_pitches = pitches.copy()

        print(round(times[frame_start+i], 2), [round(pitches[idx], 0) for idx in indexes], [note for note in notes])
        
        for key in keys:
            fs.noteon(0, key, 100)
        
        time.sleep(0.2)
        
        for idx in pressed.copy():
            if np.mean(spectrogram[i:i+5, idx]) < close_threshold:
                key = NOTE_TO_KEY[specidx_to_note[idx]]
                fs.noteoff(0, key)
                pressed.remove(idx)


if __name__ == '__main__':
    audio_file = 'data/audio/Yiruma, (이루마) - River Flows in You [7maJOI3QMu0].webm' ### 수정하세요
    # audio_file = "data/audio/Everything's Alright- Laura Shigihara- lyrics [nP-AAlZlCkM].m4a"

    n_fft = 2048
    threshold = -40
    close_threshold = -50
    note_start_threshold = 5
    win_length = 2048
    hop_length = 512

    spectrogram, sr = get_audio_frequency_spectrogram(audio_file, n_fft, hop_length)
    extract_notes(spectrogram, sr, n_fft, hop_length, threshold, close_threshold, note_start_threshold)
    