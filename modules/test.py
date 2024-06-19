import fluidsynth
import os

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def get_synth(soundfont_path='./data/Yamaha Grand-v2.1.sf2'):
    synth = fluidsynth.Synth()
    sfid = synth.sfload(os.path.abspath(soundfont_path))
    synth.program_select(0, sfid, 0, 0)
    synth.start()
    
    return synth


def get_frequency_spectrogram(audio_file, plot=True):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Generate the Short-Time Fourier Transform (STFT) of the audio
    D = np.abs(librosa.stft(y))

    # Convert amplitude to decibels
    DB = librosa.amplitude_to_db(D, ref=np.max)
    DB = DB[:512, :44*10]

    if plot:
        # Display the spectrogram
        plt.figure(figsize=(20, 12))
        img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()
    else:
        img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
    
    return img.get_array()