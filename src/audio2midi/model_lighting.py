import pytorch_lightning as pl
import torch
from torch import nn
from rich import print

from audio2midi.model import Audio2MIDITransformer, AudioEncoder


class Audio2MIDITransformerL(pl.LightningModule):
    def __init__(self, batch_size=16, lr=0.001, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr

        self.model = Audio2MIDITransformer(*args, **kwargs)
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
    
    
class Audio2EncoderL(pl.LightningModule):
    def __init__(self, batch_size=16, lr=0.001, watch_prev_n_frames=4, watch_next_n_frames=12, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.watch_prev_n_frames = watch_prev_n_frames
        self.watch_next_n_frames = watch_next_n_frames

        self.model = AudioEncoder(*args, **kwargs)
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
            y = torch.sigmoid(y[self.watch_prev_n_frames:-self.watch_next_n_frames])
            loss = self.criterion(y, t)
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