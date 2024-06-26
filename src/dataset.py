import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import librosa
import numpy as np
import glob
import pickle
import sys
import os
from functools import partial

from utils.midi import midi_to_matrix, second_per_tick
from utils.visualize import plot_spectrogram_hightlighting_pressing_notes
from audio2midi.transform import AlignTimeDimension, PadPrefix, collate_fn_making_t_prev, collate_fn


class AudioMIDIDataset(Dataset):
    def __init__(self, audio_files, midi_files, transform=None, sr=22050, n_fft=2048, win_length=2048, hop_length=512, watch_prev_n_frames=4, bpm=120, *args, **kwargs):
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.audio_files = audio_files
        self.midi_files = midi_files
        self.watch_prev_n_frames = watch_prev_n_frames
        self.bpm = bpm

        if transform is None:
            transform = PadPrefix(prefix_size=watch_prev_n_frames)
        self.transform = transform
        self.prepare_data()
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        midi_file = self.midi_files[idx]
        
        inputs, _ = self.get_frequency_spectrogram(audio_file, sr=self.sr, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        inputs = torch.from_numpy(inputs.T)
        
        with open(midi_file.replace('.mid', '.pkl'), 'rb') as f:
            labels = pickle.load(f)
        
        if self.transform:
            inputs, labels = self.transform(inputs, labels)

        return inputs, labels

    @staticmethod
    def get_frequency_spectrogram(audio_file, sr=22050, n_fft=2048, win_length=2048, hop_length=512):
        y, sr = librosa.load(audio_file, sr=sr)
        D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        DB = librosa.amplitude_to_db(D, ref=np.max)
        return DB[1:,:], sr
    
    def prepare_data(self, labeled_only_start_of_notes=False):
        print('Preprocessing data...')
        align = AlignTimeDimension(labeled_only_start_of_notes=labeled_only_start_of_notes)

        for audio_file, midi_file in zip(self.audio_files, self.midi_files):
            if os.path.relpath(midi_file.replace('.mid', '.pkl')) in glob.glob('data/train/*.pkl'):
                continue

            # Load and transform the audio to a spectrogram
            spectrogram = self.get_frequency_spectrogram(audio_file, sr=self.sr, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)[0]
            audio_length_seconds = librosa.get_duration(path=audio_file)
            
            # add one colum to the beginning of inputs as no sound at the beginning
            spectrogram = torch.from_numpy(spectrogram.T)
            
            print('Aligning MIDI matrix...', midi_file.split('/')[-1])
            
            # Load and process the MIDI file
            midi_seconds_per_tick = second_per_tick(self.bpm)
            midi_matrix = midi_to_matrix(midi_file, audio_length_seconds, bpm=self.bpm)

            labels = align(midi_matrix, len(spectrogram), midi_seconds_per_tick, audio_length_seconds)
            
            with open(midi_file.replace('.mid', '.pkl'), 'wb') as f:
                pickle.dump(labels, f)
                
        print('Preprocessing done!')


class AudioDataModule(LightningDataModule):
    def __init__(
        self, 
        audio_files='data/train/*.wav', midi_files='data/train/*.mid', 
        batch_size=16, t_prev=False,
        sr=22050, n_fft=2048, win_length=2048, hop_length=512, bpm=120, 
        audio_length=24, watch_next_n_frames=8, watch_prev_n_frames=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.audio_length = audio_length
        self.win_length = win_length
        self.watch_prev_n_frames = watch_prev_n_frames
        self.watch_next_n_frames = watch_next_n_frames
        self.t_prev = t_prev

        kwargs_dataset = {
            'sr': sr,
            'n_fft': n_fft,
            'win_length': win_length,
            'hop_length': hop_length,
            'watch_prev_n_frames': watch_prev_n_frames,
            'bpm': bpm,
        }
        audio_files = sorted(glob.glob(audio_files))
        midi_files = sorted(glob.glob(midi_files))
        self.dataset = AudioMIDIDataset(audio_files, midi_files, **kwargs_dataset)

    def train_dataloader(self, num_workers=None):
        if num_workers is None:
            num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 0

        kwargs = {
            'audio_length': self.audio_length,
            'watch_next_n_frames': self.watch_next_n_frames,
            'watch_prev_n_frames': self.watch_prev_n_frames,
            'batch_size': self.batch_size,
            'shuffle': True,
        }

        if self.t_prev:
            collate_with_param = partial(collate_fn_making_t_prev, **kwargs)
        else:
            collate_with_param = partial(collate_fn, **kwargs)
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=collate_with_param)

        return dataloader


if __name__ == '__main__':
    from audio2midi.preprocess import simplify_spectrogram_best_represent_each_note
    from utils import print_matching_highlight

    audio_files='data/train/*.wav'
    midi_files='data/train/*.mid'

    audio_files = sorted(glob.glob(audio_files))
    midi_files = sorted(glob.glob(midi_files))

    # # select one file
    # from utils.menu import show_menu
    # file_path, _ = show_menu({audio_file.split('/')[-1]: os.path.abspath(audio_file) for audio_file in audio_files}, title='Select audio file')
    # audio_files = [file_path]
    # midi_files = list(map(lambda x: x.replace('.wav', '.mid'), audio_files))

    dataset = AudioMIDIDataset(audio_files, midi_files)

    # for i, data in enumerate(dataset):
    #     inputs, labels = data
    #     print(inputs.shape, labels.shape)

    #     # Visualize whole spectrogram
    #     spectrogram_simplified = simplify_spectrogram_best_represent_each_note(inputs.T, dataset.sr, dataset.n_fft)
    #     plot_spectrogram_hightlighting_pressing_notes(spectrogram_simplified[:], labels.T, dataset.sr, dataset.hop_length)

    #     # # Visualize each moment
    #     # for j in range(inputs.shape[1]):
    #     #     spectrogram = simplify_spectrogram_best_represent_each_note(inputs[j, :], dataset.sr, dataset.n_fft)
    #     #     hot_encoded = torch.where(spectrogram > max(-80, torch.max(spectrogram)-10), spectrogram-torch.max(spectrogram)+9, 0)
    #     #     input_ = hot_encoded.to(torch.int32)
    #     #     label_ = labels[j].to(torch.int32)
            
    #     #     print_matching_highlight(label_, input_)
    #     #     input()
    #     # break

    # DataLoader with custom collate function
    kwargs = {
        'audio_length': 24,
        'watch_next_n_frames': 8,
        'watch_prev_n_frames': 4,
        'batch_size': 16,
        'shuffle': False,
    }
    collate_with_param = partial(collate_fn, **kwargs)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_with_param)

    for batches in data_loader:
        for (inputs, labels) in batches:
            print(inputs.shape, labels.shape)
            print(inputs[0][0])
            breakpoint()
    