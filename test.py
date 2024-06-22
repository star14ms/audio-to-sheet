import torch
import glob
import time

from model_lighting import Audio2MidiL # choose the model you want to train
from dataset import AudioMIDIDataset


def print_matching_highlight(list_t, list_y):
    str_t = 'Correct: '
    str_y = 'Predict: '
    header= '         A B C' + ' D EF G A B C' * 7
    for i, (item_t, item_y) in enumerate(zip(list_t, list_y)):
        if i % 12 == 3:
            str_t += ' '
            str_y += ' '
        if item_t == 0:
            str_t += f"\033[90m{item_t}\033[0m"
        else:
            str_t += f"\033[97m{item_t}\033[0m"
        if item_t == item_y:
            str_y += f"\033[92m{item_y}\033[0m"
        else:
            str_y += f"\033[91m{item_y}\033[0m"

    return print(str_t, str_y, header, sep='\n')


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
