import torch
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule

import librosa
import numpy as np
import glob
import pickle
import sys

from modules.utils.midi import midi_to_matrix, ticks_to_time
from modules.utils import print_matching_highlight


class AudioMIDIDataset(Dataset):
    def __init__(self, audio_files, midi_files, transform=None, n_fft=2048, win_length=2048, hop_length=512):
        self.transform = transform
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.audio_files = audio_files
        self.midi_files = midi_files
        self.transform = transform
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        midi_file = self.midi_files[idx]
        
        # Load and transform the audio to a spectrogram
        spectrogram = self.get_frequency_spectrogram(audio_file, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)[0]
        audio_length_seconds = librosa.get_duration(path=audio_file)
        
        # add one colum to the beginning of inputs as no sound at the beginning
        spectrogram = torch.from_numpy(spectrogram.T)
        spectrogram_shift = torch.full_like(spectrogram[0:1, :], torch.tensor(-80))
        spectrogram = torch.cat((spectrogram_shift, spectrogram), dim=0)
        
        if midi_file.replace('.mid', '.pkl') in glob.glob('data/train/*.pkl'):
            with open(midi_file.replace('.mid', '.pkl'), 'rb') as f:
                labels = pickle.load(f)
            
            return spectrogram, labels
        
        print('Loading...', midi_file.split('/')[-1])
        
        # Load and process the MIDI file
        labels = midi_to_matrix(midi_file)
        midi_seconds_per_tick = ticks_to_time(midi_file)
        
        # Apply time alignment transform
        if self.transform:
            labels = self.transform(spectrogram[1:], labels, midi_seconds_per_tick, audio_length_seconds)
        
        # add one colum to the beginning of labels to shift the labels
        labels_shift = torch.zeros_like(labels[0:1, :])
        labels = torch.cat((labels_shift, labels), dim=0)
        
        with open(midi_file.replace('.mid', '.pkl'), 'wb') as f:
            pickle.dump(labels, f)
        
        return spectrogram, labels

    @staticmethod
    def get_frequency_spectrogram(audio_file, n_fft=2048, win_length=2048, hop_length=512):
        y, sr = librosa.load(audio_file)
        D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))
        DB = librosa.amplitude_to_db(D, ref=np.max)
        return DB, sr


# only for static tempo
class AlignTimeDimension:
    def __init__(self):
        pass
    
    def __call__(self, spectrogram, midi_matrix, seconds_per_tick, audio_length_seconds):
        # Adjust the MIDI matrix to match the spectrogram frames
        num_frames = spectrogram.shape[0]
        frame_time = audio_length_seconds / num_frames
        # Assuming midi_matrix is torch.Tensor
        rescaled_matrix = torch.zeros((num_frames, midi_matrix.shape[1]))
        # Aligning the MIDI matrix to the spectrogram

        for note in range(midi_matrix.shape[1]):
            note_on_events = torch.where(midi_matrix[:, note] == 1)[0]
            for start_tick in note_on_events:
                if start_tick < midi_matrix.shape[0] - 1:
                    end_tick = start_tick + torch.where(midi_matrix[start_tick:, note] == 0)[0][0]
                    start_time = start_tick * seconds_per_tick
                    end_time = end_tick * seconds_per_tick

                    start_frame = torch.round(start_time / frame_time).to(torch.int32)
                    end_frame = torch.round(end_time / frame_time).to(torch.int32)

                    rescaled_matrix[start_frame:end_frame, note] = 1

        return rescaled_matrix


class AudioDataModule(LightningDataModule):
    def __init__(self, batch_size=1, n_fft=2048, win_length=2048, hop_length=512):
        super().__init__()
        self.batch_size = batch_size
        self.n_fft = n_fft  
        self.win_length = win_length
        self.hop_length = hop_length

    def train_dataloader(self, num_workers=None, audio_files='data/train/*.wav', midi_files='data/train/*.mid'):
        if num_workers is None:
            num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        audio_files = glob.glob(audio_files)
        midi_files = glob.glob(midi_files)

        transform = AlignTimeDimension()
        dataset = AudioMIDIDataset(audio_files, midi_files, transform=transform, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        return dataloader


if __name__ == '__main__':
    from modules.preprocess import optimize_spectrogram_best_represent_each_note
    
    datamodule = AudioDataModule(batch_size=1)
    dataloader = datamodule.train_dataloader()

    for i, data in enumerate(dataloader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
        for j in range(inputs.shape[1]):
            spectrogram = optimize_spectrogram_best_represent_each_note(inputs[0, j, :], 2048, 22050)
            hot_encoded = torch.where(spectrogram > max(-80, torch.max(spectrogram)-10), spectrogram-torch.max(spectrogram)+9, 0)
            input_ = hot_encoded.to(torch.int32)
            label_ = labels[0, j].to(torch.int32)
            
            print_matching_highlight(label_, input_)
            input()
            
