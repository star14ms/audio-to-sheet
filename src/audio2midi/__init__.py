from dataclasses import dataclass
from typing import List

from audio2midi.model_lighting import (
    AudioTransformerL, 
    AudioTransformerEncoderL, 
    AudioStartConvL,
    AudioStartConformerL,
)


def get_model_class(model_name: str):
    if model_name == 'AudioTransformer':
        model_class = AudioTransformerL
    elif model_name == 'AudioTransformerEncoder':
        model_class = AudioTransformerEncoderL
    elif model_name == 'AudioStartConv':
        model_class = AudioStartConvL
    elif model_name == 'AudioStartConformer':
        model_class = AudioStartConformerL
    else:
        raise ValueError(f"Model name {model_name} not found")

    return model_class


@dataclass
class DataConfig:
    name: str = 'default'
    sr: int = 22050
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    bpm: int = 120
    audio_length: int = 24
    watch_prev_n_frames: int = 4
    watch_prev_n_frames: int = 8
    batch_size: int = 16

@dataclass
class TrainConfig:
    name: str = 'default'
    epoch: int = 1

@dataclass
class ModelConfig:
    name: str = 'AudioTransformer'
    pass

@dataclass
class AudioTransformerConfig(ModelConfig):
    d_model: int = 1024
    hidden_dims: List[int] = (512, 256)
    n_notes: int = 88
    nhead_encoder: int = 16
    nhead_decoder: int = 11
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward_encoder: int = 2048
    dim_feedforward_decoder: int = 256
    batch_first: bool = False

@dataclass
class AudioTransformerEncoderConfig:
    d_model: int = 1024
    hidden_dims: List[int] = (512, 256)
    n_notes: int = 88
    nhead_encoder: int = 16
    num_encoder_layers: int = 6
    dim_feedforward: int = 2048
    batch_first: bool = False

@dataclass
class AudioStartConvConfig:
    d_model: int = 1024
    hidden_dims: List[int] = (512, 256)
    n_notes: int = 88
    conv_channels: List[int] = (1, 32, 64, 128)
    activation: str = 'relu'

@dataclass
class AudioStartConformerConfig:
    d_model: int = 1024
    hidden_dims: List[int] = (512, 256)
    n_notes: int = 88
    conv_channels: List[int] = (1, 32, 64, 128)
    nhead_encoder: int = 16
    num_encoder_layers: int = 4
    dim_feedforward: int = 2048
    activation: str = 'relu'
    batch_first: bool = False
