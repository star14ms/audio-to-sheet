import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_file = 'data/Yiruma, (이루마) - River Flows in You [7maJOI3QMu0].webm'
y, sr = librosa.load(audio_file)

# Generate the Short-Time Fourier Transform (STFT) of the audio
D = np.abs(librosa.stft(y))

# Convert amplitude to decibels
DB = librosa.amplitude_to_db(D, ref=np.max)

# Display the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
