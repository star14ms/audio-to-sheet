from torch import nn
import torch
import math

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term_even = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_odd = torch.exp(torch.arange(1, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]

        return self.dropout(x)


class Audio2MIDITransformer(nn.Module):
    def __init__(
        self, d_model=1025, n_notes=88, 
        nhead_encoder=5, nhead_decoder=11, 
        num_encoder_layers=6, num_decoder_layers=6, 
        dim_feedforward=2048, batch_first=False
    ):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model)
        self.pos_encoding_tgt = PositionalEncoding(n_notes)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead_encoder, 
                dim_feedforward=dim_feedforward, 
                batch_first=batch_first,
            )
        , num_layers=num_encoder_layers)
        self.freq_encoder = AudioFreqEncoder(
            in_featrue=d_model, 
            hidden_dims=(512, 256), 
            n_notes=n_notes
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_notes, 
                nhead=nhead_decoder, 
                dim_feedforward=dim_feedforward, 
                batch_first=batch_first,
            )
        , num_layers=num_decoder_layers)


    def forward(self, x, tgt, **kwargs):
        def _get_encoder_kwargs(kwargs: dict):
            return {
                'mask': kwargs.get('src_mask'),
                'src_key_padding_mask': kwargs.get('src_key_padding_mask'),
                'is_causal': kwargs.get('src_is_causal', False),
            }

        def _get_decoder_kwargs(kwargs: dict):
            return {
                'tgt_mask': kwargs.get('tgt_mask'),
                'memory_mask': kwargs.get('memory_mask'),
                'tgt_key_padding_mask': kwargs.get('tgt_key_padding_mask'),
                'memory_key_padding_mask': kwargs.get('memory_key_padding_mask'),
                'tgt_is_causal': kwargs.get('tgt_is_causal', False),
                'memory_is_causal': kwargs.get('memory_is_causal', False),
            }

        x = self.pos_encoding(x)
        tgt = self.pos_encoding_tgt(tgt)
        memory = self.encoder(x, **_get_encoder_kwargs(kwargs))
        seq_len = memory.shape[0]
        memory = self.freq_encoder(memory.view(-1, memory.shape[2]))
        memory = memory.view(seq_len, -1, memory.shape[1])
        x = self.decoder(tgt, memory, **_get_decoder_kwargs(kwargs))

        return x


if __name__ == '__main__':
    max_len = 24
    win_length = 12
    watch_prev_n_frames = 4
    tgt_max_len = max_len - win_length - watch_prev_n_frames + 1
    batch_size = 2
    n_fft = 2048
    STFT_n_rows = 1 + n_fft//2
    n_notes = 88
    
    kwargs = {
        'd_model': STFT_n_rows,
        'n_notes': n_notes,
        'nhead_encoder': 5,
        'nhead_decoder': 11,
        'num_decoder_layers': 2,
        'num_encoder_layers': 2,
        'dim_feedforward': 512,
        'max_len': max_len,
        'tgt_max_len': tgt_max_len,
        'batch_first': False
    }

    model = Audio2MIDITransformer(**kwargs)
    model.eval()

    spec_next = torch.randn(win_length, batch_size, STFT_n_rows)
    feature_prev = torch.randn(tgt_max_len, batch_size, n_notes)
    print(spec_next.shape, feature_prev.shape)

    inputs = (spec_next, feature_prev)
    x = model(*inputs)
    print(x.shape)
    
    profile_model(model, inputs)
    # from modules.utils.modelviz import draw_graphs
    # draw_graphs(model, inputs, min_depth=1, max_depth=3, directory='./output/model_viz/', hide_module_functions=True, input_names=('spec_next', 'feature_prev'), output_names=('notes_next',))
    