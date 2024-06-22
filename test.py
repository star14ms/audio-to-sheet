import torch
import glob
import time

from model_lighting import Audio2MidiL # choose the model you want to train
from dataset import AudioMIDIDataset

from modules.utils import print_matching_highlight


model = Audio2MidiL() # change the model
model.load_state_dict(torch.load('output/model.pth'))
model.eval()

audio_files = glob.glob('data/train/*.wav')
midi_files = glob.glob('data/train/*.mid')

dataset = AudioMIDIDataset(audio_files, midi_files)


for i, data in enumerate(dataset):
    inputs, labels = data
    print(inputs.shape, labels.shape)
    y_prev = labels[:1, :]

    outputs = []
    for j in range(inputs.shape[0] - model.audio_length):
        x = inputs[None, None, j:j+model.audio_length, :]
        t = labels[j+1:j+2, :].to(torch.int32)
        y = model(x, y_prev)
        
        y_prob = torch.sigmoid(y)
        y = torch.where(y_prob > model.threshold, torch.tensor(1), torch.tensor(0)).to(torch.int32)
        y_prev = y_prob
        outputs.append(y)

        print_matching_highlight(t[0].tolist(), y[0].tolist())
        time.sleep(0.5)
