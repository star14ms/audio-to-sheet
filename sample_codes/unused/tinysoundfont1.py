import tinysoundfont
import time

synth = tinysoundfont.Synth()
sfid = synth.sfload("/Users/minseo/Documents/Embers/Soundfonts/Yamaha Grand-v2.1.sf2")
synth.program_select(0, sfid, 0, 0)
synth.start()

time.sleep(0.5)

synth.noteon(0, 48, 100)
synth.noteon(0, 52, 100)
synth.noteon(0, 55, 100)

time.sleep(0.5)

synth.noteoff(0, 48)
synth.noteoff(0, 52)
synth.noteoff(0, 55)

time.sleep(0.5)

for i in range(60, 0, -12):
    synth.noteon(0, i, 100)
    time.sleep(0.5)
    synth.noteoff(0, i)
    time.sleep(0.5)
    input(i)