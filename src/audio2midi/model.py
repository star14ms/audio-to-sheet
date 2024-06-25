from torch import nn
import torch
import math


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
        self, d_model=1025, hidden_dims=(512, 256), n_notes=88, 
        nhead_encoder=5, nhead_decoder=11, 
        num_encoder_layers=6, num_decoder_layers=6, 
        dim_feedforward_encoder=2048, dim_feedforward_decoder=256, 
        batch_first=False, audio_length=24, win_length=13
    ):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model)
        self.pos_encoding_tgt = PositionalEncoding(n_notes)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead_encoder, 
                dim_feedforward=dim_feedforward_encoder, 
                batch_first=batch_first,
            )
        , num_layers=num_encoder_layers)
        
        self.freq_encoder = AudioFreqEncoder(
            in_featrue=d_model, 
            hidden_dims=hidden_dims, 
            n_notes=n_notes
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_notes, 
                nhead=nhead_decoder, 
                dim_feedforward=dim_feedforward_decoder, 
                batch_first=batch_first,
            )
        , num_layers=num_decoder_layers)
        
        tgt_length = audio_length - win_length + 1
        memory_mask = torch.full((tgt_length, audio_length), float('-inf'))
        for i in range(tgt_length):
            memory_mask[i, i:i+win_length] = 0
        self.memory_mask = memory_mask

    def forward(self, x, tgt):
        x = self.pos_encoding(x)
        tgt = self.pos_encoding_tgt(tgt)

        memory = self.encoder(x)
        seq_len = memory.shape[0]
        memory = self.freq_encoder(memory.view(-1, memory.shape[2]))
        memory = memory.view(seq_len, -1, memory.shape[1])

        x = self.decoder(tgt, memory, memory_mask=self.memory_mask[:tgt.shape[0]])

        return x

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.memory_mask = self.memory_mask.to(*args, **kwargs)  # Ensure the mask is moved with the model
        return self


class AudioEncoder(nn.Module):
    def __init__(
        self, d_model=1025, hidden_dims=(512, 256), n_notes=88,
        nhead_encoder=5,
        num_encoder_layers=6,
        dim_feedforward=2048,
        batch_first=False
    ):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model)
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
            hidden_dims=hidden_dims, 
            n_notes=n_notes
        )

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.encoder(x)
        seq_len = x.shape[0]
        x = self.freq_encoder(x.view(-1, x.shape[2]))
        x = x.view(seq_len, -1, x.shape[1])

        return x


if __name__ == '__main__':
    audio_length = 24
    watch_next_n_frames = 8
    watch_prev_n_frames = 4
    win_length = watch_prev_n_frames + 1 + watch_next_n_frames
    tgt_length = audio_length - win_length + 1
    batch_size = 2
    n_fft = 2048
    STFT_n_rows = 1 + n_fft//2
    n_notes = 88

    # model = Audio2MIDITransformer(
    #     d_model=STFT_n_rows,
    #     n_notes=n_notes,
    #     nhead_encoder=5,
    #     nhead_decoder=11,
    #     num_decoder_layers=1,
    #     num_encoder_layers=1,
    #     dim_feedforward_encoder=16,
    #     dim_feedforward_decoder=16,
    #     batch_first=False
    # )
    model = AudioEncoder(
        d_model=STFT_n_rows,
        n_notes=n_notes,
        nhead_encoder=5,
        num_encoder_layers=1,
        dim_feedforward=16,
        batch_first=False
    )
    model.eval()

    spec_next = torch.randn(audio_length, batch_size, STFT_n_rows)
    # feature_prev = torch.randn(tgt_length, batch_size, n_notes)
    # print(spec_next.shape, feature_prev.shape)

    inputs = (spec_next,)
    x = model(*inputs)
    print(x.shape)
    
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.modelviz import profile_model

    profile_model(model, inputs)
    # from modules.utils.modelviz import draw_graphs
    # draw_graphs(model, inputs, min_depth=1, max_depth=3, directory='./output/model_viz/', hide_module_functions=True, input_names=('spec_next', 'feature_prev'), output_names=('notes_next',))
    