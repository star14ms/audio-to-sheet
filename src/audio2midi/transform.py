import torch
from torch.utils.data import TensorDataset, DataLoader, BatchSampler, SequentialSampler
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


def custom_collate_fn(batch, audio_length=24, watch_n_frames=12, watch_prev_n_frames=4, batch_size=16):
    inputs, labels = batch[0] # one batch only
    tgt_length = audio_length - watch_n_frames + 1

    batch_idxs_list = list(BatchSampler(SequentialSampler(range(inputs.size(0))), audio_length, True))

    # Precompute the batched inputs and labels
    x_batches = []
    t_prev_batches = []
    t_batches = []

    for batch_idxs in batch_idxs_list:
        batch_idxs = torch.tensor(batch_idxs, dtype=torch.long)
        x_batches.append(inputs[batch_idxs].unsqueeze(1))
        
        # Calculate indices for t_prev and t
        t_prev_indices = batch_idxs[:tgt_length] + watch_prev_n_frames - 1
        t_indices = batch_idxs[:tgt_length] + watch_prev_n_frames
        
        t_prev_batches.append(labels[t_prev_indices].unsqueeze(1))
        t_batches.append(labels[t_indices].unsqueeze(1))

    # Convert lists to tensors by stacking
    x_batches = torch.stack(x_batches)
    t_prev_batches = torch.stack(t_prev_batches)
    t_batches = torch.stack(t_batches)
    
    dataset = TensorDataset(x_batches, t_prev_batches, t_batches)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)

    [x_batches, t_prev_batches, t_batches] = \
        [torch.cat([x for x, _, _ in dataloader]), torch.cat([t_prev for _, t_prev, _ in dataloader]), torch.cat([t for _, _, t in dataloader])]
    
    return x_batches, t_prev_batches, t_batches


if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader, BatchSampler, SequentialSampler
    import glob
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dataset import AudioMIDIDataset

    transform = PadPrefix(pad_size=4)

    audio_files = sorted(glob.glob('./data/train/*.wav'))
    midi_files = sorted(glob.glob('./data/train/*.mid'))
    dataset = AudioMIDIDataset(audio_files, midi_files, transform=transform)

    # DataLoader with custom collate function
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    for x_batches, t_prev_batches, t_batches in data_loader:
        print(x_batches.shape, t_prev_batches.shape, t_batches.shape)
        input()
        