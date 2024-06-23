import librosa
import librosa.display
import numpy as np

from modules.constants import NOTE_FREQUENCIES
from modules.utils import plot_spectrogram, plot_spectrogram_simplified


def get_frequency_spectrogram(audio_file, n_fft=2048, win_length=2048, hop_length=512, plot=False):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Generate the Short-Time Fourier Transform (STFT) of the audio
    D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))

    # Convert amplitude to decibels
    DB = librosa.amplitude_to_db(D, ref=np.max) # Shape: [Frequency, Time]: dB

    if plot:
        plot_spectrogram(DB, sr)
    
    return DB, sr


def simplify_spectrogram_best_represent_each_note(spectrogram, n_fft, sr):
    note_to_specindex = {}
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    for note_frequency in NOTE_FREQUENCIES:
        note = librosa.note_to_hz(note_frequency)
        note_index = np.abs(frequencies - note).argmin()
        note_to_specindex[note_frequency] = note_index

    spectrogram = np.take(spectrogram, list(note_to_specindex.values()), axis=0)

    return spectrogram


def get_simplified_frequency_spectrogram(audio_file, n_fft, win_length, hop_length, optimize=False, plot=False):
    spectrogram, sr = get_frequency_spectrogram(audio_file, n_fft, win_length, hop_length, plot=False)
    
    if optimize:
        spectrogram = simplify_spectrogram_best_represent_each_note(spectrogram, n_fft, sr)
        
    # Display the spectrogram
    if plot:
        plot_spectrogram_simplified(spectrogram, sr, hop_length)

    return spectrogram, sr
