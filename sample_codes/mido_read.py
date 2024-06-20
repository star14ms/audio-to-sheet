import mido
import os


# file_name = 'output/example_notes1.mid'
file_name = 'data/midi/Everythings_Alright_-_To_the_Moon.mid'

# Load the MIDI file
mid = mido.MidiFile(os.path.abspath('.') + '/' + file_name)

with open(file_name + '.txt', 'w') as f:
    # Print the details of the MIDI file
    for i, track in enumerate(mid.tracks):
        print(f'Track {i}: {track.name}')
        for msg in track:
            print(msg)
            print(msg, file=f)
