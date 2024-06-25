import torch
import glob
import time
import os

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from rich.traceback import install
from rich.progress import track
install()

from audio2midi import DataConfig, TrainConfig, Audio2MIDIConfig,  Audio2MIDITransformerConfig
from audio2midi.model_lighting import Audio2MIDITransformerL # choose the model you want to train
from audio2midi.preprocess import simplify_spectrogram_best_represent_each_note
from dataset import AudioMIDIDataset
from utils import print_matching_highlight
from utils.visualize import plot_spectrogram_hightlighting_pressing_notes


def test(config):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    audio_length = hparams_data.pop('audio_length')
    watch_prev_n_frames = hparams_data.pop('watch_prev_n_frames')
    _ = hparams_data.pop('watch_n_frames')
    _ = hparams_data.pop('batch_size')

    model = Audio2MIDITransformerL(**hparams_model) # change the model
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    audio_files = sorted(glob.glob('data/train/*.wav'))
    midi_files = sorted(glob.glob('data/train/*.mid'))
    dataset = AudioMIDIDataset(audio_files, midi_files, **hparams_data)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_printoptions(sci_mode=False)

    for data in dataset:
        inputs, labels = data
        inputs, labels = inputs.unsqueeze(1).to(device), labels.unsqueeze(1).to(device)
        print(inputs.shape, labels.shape)
        y_prev = torch.zeros_like(labels[0:1]).to(device)

        pressed_notes = []
        for j in track(range(inputs.shape[0] - audio_length + 1)):
            x = inputs[j:j+audio_length]
            # t = labels[j+watch_prev_n_frames+1:j+watch_prev_n_frames+2].to(torch.int32)
            y = model(x, y_prev)
            
            y_prob = torch.sigmoid(y)
            y = torch.where(y_prob > config.threshold, torch.tensor(1), torch.tensor(0)).to(torch.int32)
            y_prev = y_prob
            # print(torch.round(y_prob[0][0]*100))
            # print(y[0][0][torch.tensor([3, 15, 27, 39, 51, 63, 75, 87])])
            pressed_notes.append(y[0][0].cpu())

            # print_matching_highlight(t[0][0].tolist(), y[0][0].tolist())
            # time.sleep(0.3)

        pressed_notes = torch.stack(pressed_notes)
        spectrogram = simplify_spectrogram_best_represent_each_note(inputs.squeeze(1).T.cpu(), hparams_data['n_fft'], hparams_data['sr'])
        print(pressed_notes.T.shape, spectrogram.shape)
        plot_spectrogram_hightlighting_pressing_notes(spectrogram, pressed_notes.T, hparams_data['sr'], hparams_data['hop_length'])
        breakpoint()


cs = ConfigStore.instance()
cs.store(group="data", name="base_data", node=DataConfig, package="data")
cs.store(group="train", name="base_train", node=TrainConfig, package="train")
cs.store(group="model", name="base_Audio2MIDI_model", node=Audio2MIDIConfig, package="model")
cs.store(group="model", name="base_Audio2MIDITransformer_model", node=Audio2MIDITransformerConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="test", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    test(config)


if __name__ == '__main__':
    main()
