from torch import nn
import torch


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
    def __init__(self, in_featrue=1025*32, hidden_dims=(2048, 1024), n_notes=88, vec_size=8):
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
            
        self.out_layer = nn.Linear(hidden_dims[-1], n_notes*vec_size)
        self.n_notes = n_notes
        self.vec_size = vec_size

    def forward(self, x, **kwargs):
        for ff_layer in self.ff_layers:
            x = ff_layer(x)
            print(x.shape)
            
        x = self.out_layer(x)
        x = x.view(x.shape[0], self.n_notes, self.vec_size)

        return x


class AudioTransformerDecoder(nn.Module):
    def __init__(self, out_dim=88, nhead=8, num_layers=4):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=out_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, prev, memory, **kwargs):
        x = self.decoder(prev, memory)
        return x


class Audio2Midi(nn.Module):
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
    

class Audio2MidiOld(nn.Module):
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


def profile_model(model, inputs):
    from thop import profile
    macs, params = profile(model, inputs=inputs, verbose=False)
    print(model)
    print('모델 생성 완료! (MACs: {} G | Params: {} M)'.format(
        round(macs/1000/1000/1000, 2), 
        round(params/1000/1000, 2),
    ))


if __name__ == '__main__':
    conv_dims = [(1, 4), (4, 8), (8, 16), (16, 32)]
    audio_length = 1 + 2*len(conv_dims)
    batch_size = 2

    model = Audio2MidiOld(conv_dims)
    model.eval()

    spec_next = torch.randn(batch_size, 1, audio_length, 1025)
    # feature_prev = torch.randn(batch_size, 88, 8)
    feature_prev = torch.randn(batch_size, 88)

    inputs = (spec_next, feature_prev)
    x = model(*inputs)
    
    # profile_model(model, inputs)
    # from modules.utils.modelviz import draw_graphs
    # draw_graphs(model, inputs, min_depth=1, max_depth=3, directory='./output/model_viz/', hide_module_functions=True, input_names=('spec_next', 'feature_prev'), output_names=('notes_next',))
    