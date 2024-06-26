import librosa
import librosa.display
import numpy as np

from audio2midi.constants import NOTE_FREQUENCIES


def get_frequency_spectrogram(audio_file, sr=22050, n_fft=2048, win_length=2048, hop_length=512, **kwargs):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=sr)

    # Generate the Short-Time Fourier Transform (STFT) of the audio
    D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))

    # Convert amplitude to decibels
    DB = librosa.amplitude_to_db(D, ref=np.max) # Shape: [Frequency, Time]: dB
    
    return DB, sr


def simplify_spectrogram_best_represent_each_note(spectrogram, sr=22050, n_fft=2048, **kwargs):
    note_to_specindex = {}
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    for note_frequency in NOTE_FREQUENCIES:
        note = librosa.note_to_hz(note_frequency)
        note_index = np.abs(frequencies - note).argmin()
        note_to_specindex[note_frequency] = note_index

    spectrogram = np.take(spectrogram, list(note_to_specindex.values()), axis=0)

    return spectrogram


def get_simplified_frequency_spectrogram(audio_file, sr=22050, n_fft=2048, win_length=2048, hop_length=512, **kwargs):
    spectrogram, sr = get_frequency_spectrogram(audio_file, sr, n_fft, win_length, hop_length)
    spectrogram = simplify_spectrogram_best_represent_each_note(spectrogram, sr, n_fft)

    return spectrogram, sr
