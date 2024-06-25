from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    name: str = 'default'
    sr: int = 22050
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    bpm: int = 120
    watch_prev_n_frames: int = 1

@dataclass
class TrainConfig:
    name: str = 'default'
    epoch: int = 1

@dataclass
class ModelConfig:
    name: str = 'Audio2MIDITransformer'
    pass

@dataclass
class Audio2MIDIConfig(ModelConfig):
    batch_size: int = 16
    threshold: float = 0.7
    conv_dims: List[List[int]] = ((1, 4), (4, 8), (8, 16), (16, 32))
    hidden_dims: List[int] = (512, 256)
    n_notes: int = 88
    nhead: int = 4
    num_layers: int = 4

@dataclass
class Audio2MIDITransformerConfig(ModelConfig):
    d_model: int = 1025
    n_notes: int = 88
    nhead_encoder: int = 5
    nhead_decoder: int = 11
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward_encoder: int = 2048
    dim_feedforward_decoder: int = 256
    batch_first: bool = False
