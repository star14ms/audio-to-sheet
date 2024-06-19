import librosa
import numpy as np

# Load the audio file
audio_file = 'data/Yiruma, (이루마) - River Flows in You [7maJOI3QMu0].webm'
y, sr = librosa.load(audio_file)

# Estimate pitch using librosa's pitch tracking function
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

# Select pitches where magnitude is above a threshold
threshold = np.median(magnitudes)
pitches_above_threshold = pitches[magnitudes > threshold]

# Convert frequency to note name
def hz_to_note_name(hz):
    note = librosa.hz_to_note(hz)
    return note

notes = [hz_to_note_name(p) for p in pitches_above_threshold if p > 0]

print("Extracted notes:", notes)


# save the extracted notes to a file
output_file = 'output/notes.txt'
with open(output_file, 'w') as f:
    f.write('\n'.join(notes))
    