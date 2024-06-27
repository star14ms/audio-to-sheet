import torch
from torch.utils.data import TensorDataset, DataLoader
from rich.progress import track


# Adjust the MIDI matrix to match the spectrogram frames (Only for static tempo)
class AlignTimeDimension:
    def __init__(self, labeled_only_start_of_notes=True):
        self.labeled_only_start_of_notes = labeled_only_start_of_notes

    def __call__(self, midi_matrix, num_frames, seconds_per_tick, audio_length_seconds):
        audio_length_seconds = midi_matrix.shape[0] * seconds_per_tick
        frame_time = audio_length_seconds / num_frames
        rescaled_matrix = torch.zeros((num_frames, midi_matrix.shape[1]))
        
        if self.labeled_only_start_of_notes:
            for note in track(range(midi_matrix.shape[1]), total=midi_matrix.shape[1]):
                note_on_events = torch.where(midi_matrix[:, note] == 1)[0]
                for start_tick in note_on_events:
                    if start_tick < midi_matrix.shape[0] - 1:
                        start_time = start_tick * seconds_per_tick
                        start_frame = torch.round(start_time / frame_time).to(torch.int32)
                        rescaled_matrix[start_frame:start_frame+1, note] = 1 # Detect only the start of the note
        else:
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
    def __init__(self, prefix_size, min_db=-80):
        self.prefix_size = prefix_size
        self.min_db = min_db

    def __call__(self, inputs, labels):
        return \
            torch.cat((torch.full((self.prefix_size, inputs.shape[1]), self.min_db), inputs), dim=0), \
            torch.cat((torch.zeros((self.prefix_size, labels.shape[1])), labels), dim=0)


def collate_fn_making_t_prev(batch, audio_length=24, watch_prev_n_frames=4, watch_next_n_frames=12, batch_size=16, shuffle=True):
    inputs, labels = batch[0] # one batch only
    watch_n_frames = watch_prev_n_frames + 1 + watch_next_n_frames
    tgt_length = audio_length - watch_n_frames + 1
    
    idxes = torch.arange(inputs.size(0))
    batch_idxs_list = idxes.unfold(0, audio_length, tgt_length)

    if shuffle:
        batch_idxs_list = batch_idxs_list[torch.randperm(batch_idxs_list.size(0))]

    # Precompute the batched inputs and labels
    x_batches = []
    t_prev_batches = []
    t_batches = []

    for batch_idxs in batch_idxs_list:
        batch_idxs = torch.tensor(batch_idxs, dtype=torch.long)
        x_batches.append(inputs[batch_idxs])
        
        # Calculate indices for t_prev and t
        t_prev_indices = batch_idxs[:tgt_length] + watch_prev_n_frames - 1
        t_indices = batch_idxs[:tgt_length] + watch_prev_n_frames
        
        t_prev_batches.append(labels[t_prev_indices])
        t_batches.append(labels[t_indices])

    # Convert lists to tensors by stacking
    x_batches = torch.stack(x_batches)
    t_prev_batches = torch.stack(t_prev_batches)
    t_batches = torch.stack(t_batches)
    
    dataset = TensorDataset(x_batches, t_prev_batches, t_batches)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)

    # Transpose to fit model expectations
    x_batches = [x.transpose(0, 1) for x, _, _ in dataloader]
    t_prev_batches = [t_prev.transpose(0, 1) for _, t_prev, _ in dataloader]
    t_batches = [t.transpose(0, 1) for _, _, t in dataloader]
    
    return list(zip(x_batches, t_prev_batches, t_batches))


def collate_fn(batch, audio_length=24, watch_prev_n_frames=4, watch_next_n_frames=12, batch_size=16, shuffle=True):
    inputs, labels = batch[0] # one batch only
    watch_n_frames = watch_prev_n_frames + 1 + watch_next_n_frames
    tgt_length = audio_length - watch_n_frames + 1

    idxes = torch.arange(inputs.size(0), dtype=torch.long)
    batch_idxs_list = idxes.unfold(0, audio_length, tgt_length)

    if shuffle:
        batch_idxs_list = batch_idxs_list[torch.randperm(batch_idxs_list.size(0))]

    # Precompute the batched inputs and labels
    x_batches = []
    t_batches = []

    for batch_idxs in batch_idxs_list:
        x_batches.append(inputs[batch_idxs])
        t_indices = batch_idxs[:tgt_length] + watch_prev_n_frames
        t_batches.append(labels[t_indices])

    # Convert lists to tensors by stacking
    x_batches = torch.stack(x_batches)
    t_batches = torch.stack(t_batches)

    dataset = TensorDataset(x_batches, t_batches)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)

    # Transpose to fit model expectations
    x_batches = [x.transpose(0, 1) for x, _ in dataloader]
    t_batches = [t.transpose(0, 1) for _, t in dataloader]
    
    return list(zip(x_batches, t_batches))


if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader
    import glob
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dataset import AudioMIDIDataset

    transform = PadPrefix(prefix_size=4)

    audio_files = sorted(glob.glob('./data/train/*.wav'))
    midi_files = sorted(glob.glob('./data/train/*.mid'))
    dataset = AudioMIDIDataset(audio_files, midi_files, transform=transform)

    # DataLoader with custom collate function
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for song in data_loader:
        for batch in song:
            print(*[x.shape for x in batch])
        input()