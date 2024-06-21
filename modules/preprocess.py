import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from constant import NOTE_FREQUENCIES


def get_frequency_spectrogram(audio_file, n_fft, win_length, hop_length, plot=False):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Generate the Short-Time Fourier Transform (STFT) of the audio
    D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))

    # Convert amplitude to decibels
    DB = librosa.amplitude_to_db(D, ref=np.max) # Shape: [Frequency, Time]: dB

    if plot:
        # Display the spectrogram
        plt.figure(figsize=(15, 9))
        librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()
    
    return DB, sr


def optimize_spectrogram_best_represent_each_note(spectrogram, sr):
    note_to_specindex = {}
    n_fft = 2 * (spectrogram.shape[0] - 1)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    for note_frequency in NOTE_FREQUENCIES:
        note = librosa.note_to_hz(note_frequency)
        note_index = np.abs(frequencies - note).argmin()
        note_to_specindex[note_frequency] = note_index

    spectrogram = np.take(spectrogram, list(note_to_specindex.values()), axis=0)

    return spectrogram


def get_audio_frequency_spectrogram(audio_file, n_fft, win_length, hop_length, optimize=False, plot=False):
    spectrogram, sr = get_frequency_spectrogram(audio_file, n_fft, win_length, hop_length, plot)
    print(spectrogram.shape)
    
    if optimize:
        spectrogram = optimize_spectrogram_best_represent_each_note(spectrogram, sr)
    # print(spectrogram.shape)
        
    # # Display the spectrogram
    # plt.figure(figsize=(15, 9))
    # plt.pcolormesh(
    #     librosa.frames_to_time(np.arange(spectrogram.shape[1]+1), sr=sr, hop_length=hop_length), 
    #     np.arange(spectrogram.shape[0]+1), 
    #     spectrogram,
    #     cmap='magma'
    # )
    # plt.colorbar(format='%+2.0f dB')
    # # plt.xticks(np.arange(spectrogram.shape[1]+1, 30), np.arange(spectrogram.shape[1]+1, 30))
    # plt.yticks(np.arange(3, spectrogram.shape[0]+1, 12), ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
    # plt.title('Spectrogram')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.show()

    return spectrogram, sr
