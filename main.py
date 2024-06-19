
import numpy as np
import time
from constant import NOTE_TO_KEY
from modules.test import get_synth, get_frequency_spectrogram
import librosa


if __name__ == '__main__':

    wav_path = 'data/Yiruma, (이루마) - River Flows in You [7maJOI3QMu0].webm' ### 수정하세요
    # wav_path = "data/Everything's Alright- Laura Shigihara- lyrics [nP-AAlZlCkM].m4a"

    audio_file = 'data/Yiruma, (이루마) - River Flows in You [7maJOI3QMu0].webm'
    frequency_spectrogram = get_frequency_spectrogram(audio_file, plot=True)
    # threshold = -20
    # masks = np.where(frequency_spectrogram > threshold, 1, 0)
    # thresholded = (frequency_spectrogram - frequency_spectrogram.min()) * masks

    fs = get_synth()
    pressed = set()

    for pitches in frequency_spectrogram.T:
        print(pitches)
        indexes = np.where(pitches > 0)[0]
        notes = set([librosa.hz_to_note(p).replace('♯', '#') for p in indexes])
        keys = set([NOTE_TO_KEY[note] for note in notes])
        
        keys_to_press = keys - pressed
        keys_to_unpress = pressed - (pressed - keys)
        pressed = pressed | keys_to_press - keys_to_unpress

        print(keys_to_press)
        
        for key in keys_to_press:
            fs.noteon(0, key, 100)
        
        time.sleep(0.5)
        
        for key in keys_to_press:
            fs.noteoff(0, key)

    fs.delete()