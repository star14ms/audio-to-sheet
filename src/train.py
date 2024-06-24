import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from rich.traceback import install
install()

from dataset import AudioDataModule
from audio2midi.model_lighting import Audio2MIDIL, Audio2MIDITransformerL # choose the model you want to train
from audio2midi import (
  DataConfig,
  TrainConfig,
  Audio2MIDIConfig, 
  Audio2MIDITransformerConfig, 
)
from utils.rich import console, RichProgressBarCustom


def train(config: DictConfig):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    hparams_train = OmegaConf.to_container(config.train.params, resolve=True)
    max_epochs = hparams_train.pop("epoch", None)

    datamodule = AudioDataModule(**hparams_data)

    if config.model.name == 'Audio2MIDI':
        hparams_shared = {
            'n_fft': hparams_data['n_fft'],
            'watch_prev_n_frames': hparams_data['watch_prev_n_frames'],
        }
        model = Audio2MIDIL(**hparams_model, **hparams_train, **hparams_shared)
    elif config.model.name == 'Audio2MIDITransformer':
        model = Audio2MIDITransformerL(**hparams_model, **hparams_train)
    console.log(OmegaConf.to_yaml(config))

    # Initialize a trainer
    logger = TensorBoardLogger("./lightning_logs/", name=model.__class__.__name__)
    logger.log_hyperparams(params=hparams_model)
    
    trainer = Trainer(
        max_epochs=max_epochs, 
        logger=logger, 
        log_every_n_steps=1, 
        accelerator='mps' if torch.backends.mps.is_available() else None,
        callbacks=[RichProgressBarCustom()]
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Save the model to disk (optional)
    torch.save(model.state_dict(), './output/model.pth')


cs = ConfigStore.instance()
cs.store(group="data", name="base_data", node=DataConfig, package="data")
cs.store(group="train", name="base_train", node=TrainConfig, package="train")
cs.store(group="model", name="base_Audio2MIDI", node=Audio2MIDIConfig, package="model")
cs.store(group="model", name="base_Audio2MIDITransformer_model", node=Audio2MIDITransformerConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    train(config)


if __name__ == '__main__':
    main()
