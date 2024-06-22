import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import one_hot
from model import Audio2Midi, Audio2MidiOld
from modules.utils.rich import new_progress


class Audio2MidiL(pl.LightningModule):
    def __init__(self, batch_size=16, threshold=0.7, *args, **kwargs):
        super().__init__()
        self.model = Audio2MidiOld(*args, **kwargs)
        self.audio_length = self.model.audio_length
        self.criterion = nn.BCELoss()
        self.batch_size = batch_size
        self.threshold = threshold

    def forward(self, inputs, labels_shift):
        # In Lightning, forward defines the prediction/inference actions
        return self.model(inputs, labels_shift)

    def training_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs[0], labels[0]
        total_loss = 0
        opt = self.optimizers()
        
        progress = new_progress()
        progress.start()

        total = int((inputs.size(0) - self.audio_length) / self.batch_size)
        id = progress.add_task(f"iter: 0/{total}", total=total)
        
        # add one colum to the beginning of labels to shift the labels
        labels_shift = torch.zeros_like(labels[0:1, :])
        labels = torch.cat((labels_shift, labels), dim=0)
        
        n_jumps = inputs.size(0) // self.batch_size
        idxs_start = list(range(0, inputs.size(0)-n_jumps, n_jumps))
        
        for i in range(total):
        # for i in range(inputs.shape[1] - self.audio_length):
            # x = inputs[:, i:i+self.audio_length, :].unsqueeze(1)
            # y_prev = labels[i:i+1, :]
            # y_next = labels[i+1:i+2, :]
            x = torch.stack([inputs[idx:idx+self.audio_length, :] for idx in idxs_start], dim=0).unsqueeze(1)
            y_prev = torch.stack([labels[idx:idx+1] for idx in idxs_start], dim=0).squeeze(1)
            y_next = torch.stack([labels[idx+1:idx+2] for idx in idxs_start], dim=0).squeeze(1)
            
            # forward + backward + optimize
            outputs = self.model(x, y_prev)
            outputs_prob = torch.softmax(outputs, dim=1)
            # if prob is more than threshold, then it is considered as 1
            outputs = (outputs_prob > self.threshold).to(torch.float32)
            # multiple answers can be correct, so we use binary cross entropy loss
            loss = self.criterion(outputs_prob, y_next)
            total_loss += loss

            if i == total - 1:
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

            progress.update(id, advance=1, description="iter: {}/{}".format(i, total))

            if i % 10 == 0:
                pass
                # progress.log(f"Train loss: {round(total_loss.item() / 10, 6)}")
                # total_loss = 0

        print(f"Train loss: {round(total_loss.item(), 6)}")
        progress.stop()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def train_dataloader(self) -> torch.Any:
        return super().train_dataloader()
    