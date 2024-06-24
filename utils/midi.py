import mido
from mido.midifiles.tracks import _to_abstime, _to_reltime
import torch

# from constant import KEY_TO_NOTE


def write_notes_to_midi(notes, filename='output/example_notes.mid', bpm=120):
    # Create a new MIDI file with one track
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    
    mid.tracks.append(track)

    # Add the notes to the track
    sec_per_tick = 60/bpm/480

    for type, note, time_from_prev in notes:
        track.append(mido.Message('note_on', note=note, velocity=64 if type == 'on' else 0, time=round(time_from_prev/sec_per_tick)))
    
    # print(track)
    # Save the MIDI file
    mid.save(filename)


def get_merged_track_abs_time(mid):
  messages = []
  for track in mid.tracks:
      messages.extend(_to_abstime(track, skip_checks=True))
  messages.sort(key=lambda msg: msg.time)

  return mido.MidiTrack(messages)


def get_merged_track_rel_time(mid):
  messages = []
  for track in mid.tracks:
      messages.extend(_to_abstime(track, skip_checks=True))
  messages.sort(key=lambda msg: msg.time)

  return mido.MidiTrack(_to_reltime(messages, skip_checks=True))


def midi_fix_time_as_tempo_120(mid):
    times_tempo_changed = []
    tempo_list = []
    
    for track in mid.tracks:
      current = 0
      for msg in track:
          current += msg.time
          if msg.type == 'set_tempo':
              times_tempo_changed.append(current)
              tempo_list.append(msg.tempo)
  
    for track in mid.tracks:
      tempo_init = 500000  # default tempo (120 bpm)
      tempo = tempo_init
      current = 0
      idx_tempo = 0
    
      for msg in track:
        duration_prev_tempos = 0
        
        while idx_tempo < len(times_tempo_changed) and current + msg.time >= times_tempo_changed[idx_tempo]:
          frac_time = times_tempo_changed[idx_tempo] - current
          duration_prev_tempos += frac_time * tempo / tempo_init
          current += frac_time
          msg.time -= frac_time
          
          tempo = tempo_list[idx_tempo]
          idx_tempo += 1
      
        current += msg.time
        msg.time = round(duration_prev_tempos + msg.time * tempo / tempo_init)

    return mid


def second_per_tick(bpm=120, ticks_per_beat=480):
  return (60 / bpm) / ticks_per_beat


def midi_to_matrix(midi_file, audio_length_seconds=None, bpm=120):
  mid = mido.MidiFile(midi_file)
  mid = midi_fix_time_as_tempo_120(mid)
  track = get_merged_track_abs_time(mid)

  if audio_length_seconds:
    num_frames = round(audio_length_seconds / second_per_tick(bpm, mid.ticks_per_beat))
  else:
    num_frames = track[-1].time

  matrix = torch.zeros([num_frames, 88])
  matrix_last_pressed = torch.zeros(88, dtype=int)
  unused_time = 0
  
  for msg in track:
    if msg.type == 'note_on':
      if msg.note < 21 or msg.note > 108:
        unused_time += msg.time
        continue
      if msg.velocity != 0:
        matrix_last_pressed[msg.note - 21] = msg.time + unused_time
        unused_time = 0
      else:
        press_since = matrix_last_pressed[msg.note - 21] + unused_time
        unused_time = 0
        matrix[press_since:msg.time, msg.note - 21] = 1
      # print(KEY_TO_NOTE[msg.note], msg.time, 'on' if msg.velocity != 0 else 'off')
      
  return matrix


def matrix_to_midi(matrix, output_file):
  mid = mido.MidiFile()
  track = mido.MidiTrack()
  mid.tracks.append(track)

  # bpm = 80
  # track.append(mido.MetaMessage('set_tempo', tempo=round(1000000*60/bpm), time=0))

  last_time = 0
  matrix_last_pressed = torch.zeros(88, dtype=int)
  pressing = [False] * 88

  for i, row in enumerate(matrix):
    for j, pressed in enumerate(row):

      if pressed and not pressing[j]:
        pressing[j] = True
        msg = mido.Message('note_on', note=j+21, velocity=64, time=i-last_time)
        last_time = i
        track.append(msg)
        matrix_last_pressed[msg.note - 21] = i
      elif not pressed and pressing[j]:
        pressing[j] = False
        msg = mido.Message('note_on', note=j+21, velocity=0, time=i-last_time)
        last_time = i
        track.append(msg)
        
  mid.save(output_file)
  
  return track


if __name__ == '__main__':
  midi_file = 'data/midi/River Flows in You.mid'
  out_file = 'output/River Flows in You.mid'

  matrix = midi_to_matrix(midi_file)
  # print(matrix.sum(axis=1)[0:500])

  track = matrix_to_midi(matrix, out_file)
  
  mid = mido.MidiFile(midi_file)
  basic = get_merged_track_rel_time(mid)
  
  print(basic[:30])
  # encoded = track
  
  # print(basic[31:31+10])
  # print(encoded[10:10+10])
  
  # for b, e in zip(basic[21:], encoded):
  #   if b.type == 'note_on':
  #     print(f'{b.note} {b.velocity} {b.time}', end=' | ')  
  #   else:
  #     print(b, end=' | ')
      
  #   if e.type == 'note_on':
  #     print(f'{e.note} {e.velocity} {e.time}')
