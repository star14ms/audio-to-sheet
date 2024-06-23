import torch
from rich.progress import track


# Adjust the MIDI matrix to match the spectrogram frames (Only for static tempo)
class AlignTimeDimension:
    def __call__(self, midi_matrix, num_frames, seconds_per_tick, audio_length_seconds):
        audio_length_seconds = midi_matrix.shape[0] * seconds_per_tick
        frame_time = audio_length_seconds / num_frames
        rescaled_matrix = torch.zeros((num_frames, midi_matrix.shape[1]))

        for note in track(range(midi_matrix.shape[1]), total=midi_matrix.shape[1]):
            note_on_events = torch.where(midi_matrix[:, note] == 1)[0]
            for start_tick in note_on_events:
                if start_tick < midi_matrix.shape[0] - 1:
                    if not torch.any(midi_matrix[start_tick:, note] == 0):
                        end_frame = num_frames
                    else:
                        end_tick = start_tick + torch.where(midi_matrix[start_tick:, note] == 0)[0][0]
                        end_time = end_tick * seconds_per_tick
                        end_frame = torch.round(end_time / frame_time).to(torch.int32)

                    start_time = start_tick * seconds_per_tick
                    start_frame = torch.round(start_time / frame_time).to(torch.int32)
                    rescaled_matrix[start_frame:end_frame, note] = 1

        return rescaled_matrix


# add one colum to the beginning of inputs as no sound at the beginning
class PadPrefix:
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, inputs, labels):
        return \
            torch.cat((torch.full((self.pad_size, inputs.shape[1]), -80), inputs), dim=0), \
            torch.cat((torch.zeros((self.pad_size, labels.shape[1])), labels), dim=0)
