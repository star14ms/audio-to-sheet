import torch
from pytorch_lightning import Trainer
from model_lighting import Audio2MidiL # choose the model you want to train
from dataset import AudioDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

hparams_model ={
  'batch_size': 16,
  'threshold': 0.7,
  'conv_dims': [(1, 4), (4, 8), (8, 16), (16, 32)], 
  'hidden_dims': (512, 256), 
  'n_notes': 88, 
  'nhead': 4, 
  'num_layers': 4
}

hparams_data = {
  # 'batch_size': 1,
  'win_length': 2048,
  'hop_length': 512,
  'bpm': 120,
}

hparams_shared = {
  'n_fft': 2048,
  'watch_prev_n_frames': 1,
}

model = Audio2MidiL(**hparams_shared, **hparams_model) # change the model
datamodule = AudioDataModule(**hparams_shared, **hparams_data)

# Initialize a trainer
logger = TensorBoardLogger("./lightning_logs/", name=model.__class__.__name__)
logger.log_hyperparams(params=hparams_model)
trainer = Trainer(max_epochs=10, logger=logger, accelerator='mps' if torch.backends.mps.is_available() else None, enable_progress_bar=False)

# Train the model
trainer.fit(model, datamodule=datamodule)

# Save the model to disk (optional)
torch.save(model.state_dict(), 'output/model.pth')
