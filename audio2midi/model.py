from torch import nn
import torch

from utils.modelviz import profile_model


class AudioTimeEncoder(nn.Module):
    def __init__(self, conv_dims=[(1, 4), (4, 8), (8, 16), (16, 32)]):
        super().__init__()
        
        self.conv_layers = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(conv_dims):
            conv_black = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1), padding=(0, 0)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

            self.conv_layers.add_module(f'conv_block_{i}', conv_black)
            
    def forward(self, x, **kwargs):
        for conv_block in self.conv_layers:
            x = conv_block(x)
            # print(x.shape)
            
        return x


class AudioFreqEncoder(nn.Module):
    def __init__(self, in_featrue=1025, hidden_dims=(512, 256), n_notes=88):
        super().__init__()
        
        self.ff_layers = nn.Sequential()
        feature_sizes = [in_featrue] + [*hidden_dims]

        for i in range(len(feature_sizes)-1):
            module = nn.Sequential(
                nn.Linear(feature_sizes[i], feature_sizes[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(feature_sizes[i+1])
            )
            self.ff_layers.add_module(f'ff_layer_{i}', module)
            
        self.out_layer = nn.Linear(hidden_dims[-1], n_notes)

    def forward(self, x, **kwargs):
        for ff_layer in self.ff_layers:
            x = ff_layer(x)
            # print(x.shape)
            
        x = self.out_layer(x)
        return x


class AudioTransformerDecoder(nn.Module):
    def __init__(self, out_dim=88, nhead=8, num_layers=4):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=out_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, prev, memory, **kwargs):
        x = self.decoder(prev, memory)
        return x


class Audio2MIDI(nn.Module):
    def __init__(self, conv_dims=[(1, 4), (4, 8), (8, 16), (16, 32)], hidden_dims=(2048, 1024), n_fft=2048, n_notes=88, vec_size=8, nhead=4, num_layers=4):
        super().__init__()
        n_fft = 2048
        n_freq = n_fft // 2 + 1
        
        self.audio_length = 1 + 2*len(conv_dims)
        self.time_encoder = AudioTimeEncoder(conv_dims)
        self.freq_encoder = AudioFreqEncoder(in_featrue=n_freq*conv_dims[-1][-1], hidden_dims=hidden_dims, n_notes=n_notes, vec_size=vec_size)
        self.attn = AudioTransformerDecoder(out_dim=vec_size, nhead=nhead, num_layers=num_layers)
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_notes*vec_size, n_notes),
        )

    def forward(self, x, featrue_prev, **kwargs):
        x = self.time_encoder(x)
        x = torch.squeeze(x, dim=2)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        featrue_next = self.freq_encoder(x)
        print(featrue_next.shape)
        x = self.attn(featrue_next, featrue_prev)
        print(x.shape)
        x = self.predictor(x)
        print(x.shape)
        return x


class AudioFreqEncoderOld(nn.Module):
    def __init__(self, in_featrue=1025, hidden_dims=(512, 256), n_notes=88, n_channels=32):
        super().__init__()
        
        self.ff_layers = nn.Sequential()
        feature_sizes = [in_featrue] + [*hidden_dims]

        for i in range(len(feature_sizes)-1):
            module = nn.Sequential(
                nn.Linear(feature_sizes[i], feature_sizes[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(n_channels)
            )
            self.ff_layers.add_module(f'ff_layer_{i}', module)
            
        self.out_layer = nn.Linear(hidden_dims[-1], n_notes)

    def forward(self, x, **kwargs):
        for ff_layer in self.ff_layers:
            x = ff_layer(x)
            # print(x.shape)
            
        x = self.out_layer(x)
        return x


class Audio2MIDIOld(nn.Module):
    def __init__(self, conv_dims=[(1, 4), (4, 8), (8, 16), (16, 32)], hidden_dims=(512, 256), n_fft=2048, n_notes=88, nhead=4, num_layers=4):
        super().__init__()
        n_freq = n_fft // 2 + 1
        
        self.audio_length = 1 + 2*len(conv_dims)
        self.n_notes = n_notes
        self.time_encoder = AudioTimeEncoder(conv_dims)
        self.freq_encoder = AudioFreqEncoderOld(in_featrue=n_freq, hidden_dims=hidden_dims, n_notes=n_notes, n_channels=conv_dims[-1][-1])
        self.predictor = AudioTransformerDecoder(out_dim=n_notes, nhead=nhead, num_layers=num_layers)
        self.out_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dims[-1][-1]*n_notes, n_notes),
        )

    def forward(self, x, y_prev, **kwargs):
        x = self.time_encoder(x)
        x = x.squeeze(2)
        # print(x.shape)
        x = self.freq_encoder(x)
        # print(x.shape)
        x = self.predictor(x, y_prev.unsqueeze(1).expand(-1, 32, -1))
        x = self.out_layer(x)
        # print(x.shape)
        return x

