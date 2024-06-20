import mido

# Create a new MIDI file with one track
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

# Set the tempo to 60 BPM (1,000,000 microseconds per quarter note)
tempo = mido.MetaMessage('set_tempo', tempo=1000000, time=0)
track.append(tempo)

# Define the notes and their durations (in ticks)
notes = [
    (60, 120),  # C4, 16th note
    (62, 120),  # D4, 16th note
    (64, 240)   # E4, 8th note
]

# Add sustain pedal on message (value 127)
track.append(mido.Message('control_change', control=64, value=127, time=0))

# Add the notes to the track
time = 0
for note, duration in notes:
    track.append(mido.Message('note_on', note=note, velocity=64, time=time))
    track.append(mido.Message('note_off', note=note, velocity=64, time=duration))
    time = 0  # Subsequent notes start immediately after the previous one

# Add sustain pedal off message (value 0) after the last note
track.append(mido.Message('control_change', control=64, value=0, time=0))

print(track)

# Save the MIDI file
mid.save('output/example_notes.mid')
