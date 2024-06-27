import pytorch_lightning as pl
import torch
from torch import nn
from rich import print

from audio2midi.model import AudioTransformer, AudioTransformerEncoder, AudioStartConv, AudioStartConformer
from audio2midi.preprocess import simplify_spectrogram_best_represent_each_note
from utils.visualize import plot_spectrograms_simplified


class AudioTransformerL(pl.LightningModule):
    def __init__(self, lr=0.001, watch_prev_n_frames=4, watch_next_n_frames=8, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.win_length = watch_prev_n_frames + 1 + watch_next_n_frames

        self.model = AudioTransformer(win_length=self.win_length, *args, **kwargs)
        self.criterion = nn.BCELoss() # multiple answers can be correct, so we use binary cross entropy loss
        
        device = 'mps' if torch.backends.mps.is_available() else None
        self.model.to(device)

    def forward(self, inputs, y_prev):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, y_prev)

    def training_step(self, batches):
        total_loss = 0
        total = len(batches)
        opt = self.optimizers()

        # Process each batch
        for i, (x_batch, t_prev_batch, t_batch) in enumerate(batches):

            # forward + backward + optimize
            y = self.model(x_batch, t_prev_batch)
            y_prob = torch.sigmoid(y)
            loss = self.criterion(y_prob, t_batch)
            total_loss += loss
            
            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Train loss: {round(total_loss.item() / len(batches), 6)}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self) -> torch.Any:
        return super().train_dataloader()
    

class AudioTransformerEncoderL(pl.LightningModule):
    def __init__(self, lr=0.001, watch_prev_n_frames=4, watch_next_n_frames=12, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.watch_prev_n_frames = watch_prev_n_frames
        self.watch_next_n_frames = watch_next_n_frames
        self.win_length = watch_prev_n_frames + 1 + watch_next_n_frames

        self.model = AudioTransformerEncoder(win_length=self.win_length, *args, **kwargs)
        self.criterion = nn.BCELoss() # multiple answers can be correct, so we use binary cross entropy loss
        
        device = 'mps' if torch.backends.mps.is_available() else None
        self.model.to(device)

    def forward(self, inputs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs)

    def training_step(self, batches):
        total_loss = 0
        total = len(batches)
        opt = self.optimizers()

        # Process each batch
        for i, (x, t) in enumerate(batches):

            # forward + backward + optimize
            y = self.model(x)
            y_prob = torch.sigmoid(y[self.watch_prev_n_frames:-self.watch_next_n_frames])
            loss = self.criterion(y_prob, t)
            total_loss += loss

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        # self.global_step, self._trainer.fit_loop.max_batches,
        # if self._trainer.is_last_batch: 
        #     self.visualize_dataset(x, y, t)
        print(f"Train loss: {round(total_loss.item() / len(batches), 6)}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self) -> torch.Any:
        return super().train_dataloader()
    
    def visualize_dataset(self, x, y, t):
        x, y, t = x[:,0,:].squeeze().T.detach().cpu(), y[:,0,:].squeeze().T.detach().cpu(), t[:,0,:].squeeze().T.detach().cpu()
        
        x_simplified = simplify_spectrogram_best_represent_each_note(x)
        pad_prev = torch.zeros([y.shape[0], self.watch_prev_n_frames])
        pad_next = torch.zeros([y.shape[0], self.watch_next_n_frames])
        t = torch.cat([pad_prev, t, pad_next], dim=1)

        plot_spectrograms_simplified(x_simplified, y, t)


class AudioStartConvL(pl.LightningModule):
    def __init__(self, lr=0.001, watch_prev_n_frames=4, watch_next_n_frames=12, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.watch_prev_n_frames = watch_prev_n_frames
        self.watch_next_n_frames = watch_next_n_frames

        self.model = AudioStartConv(*args, **kwargs)
        self.criterion = nn.BCELoss() # multiple answers can be correct, so we use binary cross entropy loss
        
        device = 'mps' if torch.backends.mps.is_available() else None
        self.model.to(device)

    def forward(self, inputs):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs)

    def training_step(self, batches):
        total_loss = 0
        total = len(batches)
        opt = self.optimizers()

        # Process each batch
        for i, (x, t) in enumerate(batches):

            # forward + backward + optimize
            y = self.model(x, self.watch_prev_n_frames, self.watch_next_n_frames)
            y_prob = torch.sigmoid(y)
            loss = self.criterion(y_prob, t)
            total_loss += loss

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        # self.global_step, self._trainer.fit_loop.max_batches,
        # if self._trainer.is_last_batch: 
        #     self.visualize_dataset(x, y, t)
        print(f"Train loss: {round(total_loss.item() / len(batches), 6)}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self) -> torch.Any:
        return super().train_dataloader()
    
    def visualize_dataset(self, x, y, t):
        x, y, t = x[:,0,:].squeeze().T.detach().cpu(), y[:,0,:].squeeze().T.detach().cpu(), t[:,0,:].squeeze().T.detach().cpu()
        
        x_simplified = simplify_spectrogram_best_represent_each_note(x)
        pad_prev = torch.zeros([y.shape[0], self.watch_prev_n_frames])
        pad_next = torch.zeros([y.shape[0], self.watch_next_n_frames])
        t = torch.cat([pad_prev, t, pad_next], dim=1)

        plot_spectrograms_simplified(x_simplified, y, t)


class AudioStartConformerL(AudioStartConvL):
    def __init__(self, lr=0.001, watch_prev_n_frames=4, watch_next_n_frames=12, *args, **kwargs):
        self.lr = lr
        self.watch_prev_n_frames = watch_prev_n_frames
        self.watch_next_n_frames = watch_next_n_frames

        self.model = AudioStartConformer(*args, **kwargs)
        self.criterion = nn.BCELoss() # multiple answers can be correct, so we use binary cross entropy loss
        
        device = 'mps' if torch.backends.mps.is_available() else None
        self.model.to(device)
