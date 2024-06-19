import numpy as np
import simpleaudio as sa
from pydub import AudioSegment, playback

# Define note frequencies (for the standard A4 = 440Hz tuning)
NOTE_FREQUENCIES = {
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
    'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    # Add more notes if needed
}

# Convert note to frequency
def note_to_frequency(note):
    return NOTE_FREQUENCIES.get(note, 440.00)  # Default to A4 if note not found

# Generate a sine wave for a given frequency
def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    wave = np.int16(wave * 32767)  # Convert to 16-bit PCM
    return wave

# Play a list of notes
def play_notes(notes, duration=1.0):
    sample_rate = 44100
    for note in notes:
        frequency = note_to_frequency(note)
        wave = generate_sine_wave(frequency, duration, sample_rate)
        audio = np.concatenate([wave, np.zeros(int(sample_rate * 0.1))])  # Add a short silence between notes
        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)
        play_obj.wait_done()

# Example notes
notes = ['C4', 'E4', 'G4', 'A4', 'B4', 'C5']

# Play the notes
play_notes(notes)
