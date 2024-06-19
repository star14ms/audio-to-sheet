import time
import fluidsynth
import os

fs = fluidsynth.Synth()
fs.start()

rel_path = './data/Yamaha Grand-v2.1.sf2'

sfid = fs.sfload(os.path.abspath(rel_path))
fs.program_select(0, sfid, 0, 0)

fs.noteon(0, 60, 100)
fs.noteon(0, 67, 100)
fs.noteon(0, 76, 100)

time.sleep(1.0)

fs.noteoff(0, 60)
fs.noteoff(0, 67)
fs.noteoff(0, 76)

time.sleep(1.0)

fs.delete()
