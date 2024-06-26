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

from audio2midi import DataConfig, TrainConfig, Audio2MIDITransformerConfig, AudioEncoderConfig
from audio2midi.model_lighting import Audio2MIDITransformerL, Audio2EncoderL # choose the model you want to train
from audio2midi.preprocess import simplify_spectrogram_best_represent_each_note
from dataset import AudioMIDIDataset
from utils import print_matching_highlight
from utils.visualize import plot_spectrogram_hightlighting_pressing_notes, plot_spectrograms_simplified
from utils.rich import console


def test(config):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    audio_length = hparams_data.pop('audio_length')
    watch_prev_n_frames = hparams_data.pop('watch_prev_n_frames')
    watch_next_n_frames = hparams_data.pop('watch_next_n_frames')
    watch_n_frames = watch_prev_n_frames + 1 + watch_next_n_frames
    _ = hparams_data.pop('batch_size')
    
    if config.model.name == 'Audio2MIDITransformer':
        model = Audio2MIDITransformerL(**hparams_model)
    elif config.model.name == 'AudioEncoder':
        shared_parames = {
            'watch_prev_n_frames': watch_prev_n_frames,
            'watch_next_n_frames': watch_next_n_frames,
        }
        model = Audio2EncoderL(**hparams_model, **shared_parames)
    else:
        raise ValueError(f"Model name {config.model.name} not found")

    console.log(OmegaConf.to_yaml(config))
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

        y_full = []
        for j in track(range(inputs.shape[0] - watch_n_frames + 1)):
            x = inputs[j:j+watch_n_frames]
            t = labels[j+watch_prev_n_frames:j+watch_prev_n_frames+1].to(torch.int32).squeeze(0).T.cpu()
            
            if config.model.name == 'Audio2MIDITransformer':
                y = model(x, y_prev)
            elif config.model.name == 'AudioEncoder':
                y = model(x)
            
            y_prob = torch.sigmoid(y)
            y = torch.where(y_prob >= config.threshold, torch.tensor(1), torch.tensor(0)).to(torch.int32)
            y_full.append(y[0][0].cpu())

            if config.model.name == 'Audio2MIDITransformer':
                y_prev = y_prob

            # visualize
            # print(torch.round(y_prob[0][0]*100))
            # print(y_prob[0][0][torch.tensor([3, 15, 27, 39, 51, 63, 75, 87])])

            print_matching_highlight(t.squeeze().to(torch.int32).tolist(), y.squeeze()[watch_prev_n_frames].tolist())

            # time.sleep(0.3)
        x_full = simplify_spectrogram_best_represent_each_note(inputs.squeeze(1).T.cpu(), **hparams_data)
        y_full = torch.stack(y_full)
        pad = torch.zeros([x_full.shape[1]-y_full.shape[0], len(y_full[0])])
        y_full = torch.cat([y_full, pad])

        plot_spectrogram_hightlighting_pressing_notes(x_full, y_full.T, **hparams_data)


cs = ConfigStore.instance()
cs.store(group="data", name="base_data", node=DataConfig, package="data")
cs.store(group="train", name="base_train", node=TrainConfig, package="train")
cs.store(group="model", name="base_Audio2MIDITransformer_model", node=Audio2MIDITransformerConfig, package="model")
cs.store(group="model", name="base_AudioEncoder_model", node=AudioEncoderConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="test", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    test(config)


if __name__ == '__main__':
    main()
