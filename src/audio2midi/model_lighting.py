import pytorch_lightning as pl
import torch
from torch import nn
from audio2midi.model import Audio2MIDI, Audio2MIDIOld, Audio2MIDITransformer
from utils.rich import new_progress
from torch.utils.data import TensorDataset, DataLoader, BatchSampler, SequentialSampler


class Audio2MIDIL(pl.LightningModule):
    def __init__(self, batch_size=16, lr=0.001, *args, **kwargs):
        super().__init__()
        self.model = Audio2MIDIOld(*args, **kwargs)
        self.audio_length = self.model.audio_length
        self.n_notes = self.model.n_notes
        self.criterion = nn.BCELoss() # multiple answers can be correct, so we use binary cross entropy loss
        
        self.batch_size = batch_size
        self.lr = lr

        self.progress = new_progress()
        self.progress.start()

    def forward(self, inputs, labels_shift):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, labels_shift)

    def training_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs[0], labels[0]
        total_loss = 0
        opt = self.optimizers()
    
        n_jumps = (inputs.size(0) - self.audio_length) // self.batch_size # 6 = (81 - 8) // 10
        total = n_jumps
        idxs_start = list(range(0, inputs.size(0) - self.audio_length - n_jumps, n_jumps)) 
        
        id = self.progress.add_task(f"iter: 0/{total}", total=total)
        
        for i in range(n_jumps):
            x = torch.stack([inputs[start+i+1:start+i+1+self.audio_length] for start in idxs_start], dim=0).unsqueeze(1)
            t_prev = \
                torch.stack([labels[start+i:start+i+1] for start in idxs_start], dim=0).squeeze(1)
            t = torch.stack([labels[start+i+1:start+i+2] for start in idxs_start], dim=0).squeeze(1)
            
            # forward + backward + optimize
            y = self.model(x, t_prev)
            y_prob = torch.sigmoid(y)
            loss = self.criterion(y_prob, t)
            total_loss += loss

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

            self.progress.update(id, advance=1, description="iter: {}/{}".format(i, total))

            # if i % 10 == 0:
                # self.progress.log(f"Train loss: {round(total_loss.item() / 10, 6)}")
                # total_loss = 0

        self.progress.log(f"Train loss: {round(total_loss.item(), 6)}")
        self.progress.remove_task(id)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self) -> torch.Any:
        return super().train_dataloader()
    

class Audio2MIDITransformerL(pl.LightningModule):
    def __init__(self, batch_size=16, audio_length=24, win_length=12, watch_prev_n_frames=4, lr=0.001, *args, **kwargs):
        super().__init__()
        self.model = Audio2MIDITransformer(*args, **kwargs)
        self.criterion = nn.BCELoss() # multiple answers can be correct, so we use binary cross entropy loss
        
        self.batch_size = batch_size
        self.audio_length = audio_length
        self.win_length = win_length
        self.tgt_length = audio_length - win_length + 1
        self.watch_prev_n_frames = watch_prev_n_frames
        self.lr = lr

        self.src_mask = nn.Transformer.generate_square_subsequent_mask(29)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(11)
        self.memory_mask = torch.triu(torch.full((11, 29), float('-inf')), diagonal=1)

        self.progress = new_progress()
        self.progress.start()

    def forward(self, inputs, y_prev):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, y_prev)

    def training_step(self, batch):
        x_batches, t_prev_batches, t_batches = batch
        total_loss = 0
        opt = self.optimizers()
        
        total = len(x_batches)
        id = self.progress.add_task(f"iter: 0/{total}", total=total)

        # Process each batch
        for i, (x_batch, t_prev_batch, t_batch) in enumerate(zip(x_batches, t_prev_batches, t_batches)):
            # Transpose to fit model expectations
            x_batch = x_batch.transpose(0, 1)
            t_prev_batch = t_prev_batch.transpose(0, 1)
            t_batch = t_batch.transpose(0, 1)

            # forward + backward + optimize
            y = self.model(x_batch, t_prev_batch)
            y_prob = torch.sigmoid(y)
            loss = self.criterion(y_prob, t_batch)
            total_loss += loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            self.progress.update(id, advance=1, description="iter: {}/{}".format(i, total))
            
            self.log('train_loss', total_loss / len(x_batches), on_step=True, on_epoch=True)
            return {'loss': total_loss / len(x_batches)}

        self.progress.remove_task(id)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self) -> torch.Any:
        return super().train_dataloader()
    