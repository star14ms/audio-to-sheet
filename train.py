import torch
from pytorch_lightning import Trainer
from model_lighting import Audio2MidiL # choose the model you want to train
from dataset import AudioDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


model = Audio2MidiL() # change the model
datamodule = AudioDataModule(batch_size=128)

# Initialize a trainer
logger = TensorBoardLogger("./lightning_logs/", name=model.__class__.__name__)
trainer = Trainer(max_epochs=10, logger=logger, accelerator='mps' if torch.backends.mps.is_available() else None, enable_progress_bar=False)

# Train the model
trainer.fit(model, datamodule=datamodule)

# Save the model to disk (optional)
torch.save(model.state_dict(), 'output/model.pth')
