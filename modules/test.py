import fluidsynth
import os


def get_synth(soundfont_path='./data/Yamaha Grand-v2.1.sf2'):
    synth = fluidsynth.Synth()
    sfid = synth.sfload(os.path.abspath(soundfont_path))
    synth.program_select(0, sfid, 0, 0)
    synth.start()
    
    return synth
