import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os

import warnings
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from rich import print
from rich.traceback import install
install()

from audio2midi import DataConfig, TrainConfig, Audio2MIDITransformerConfig, AudioEncoderConfig
from audio2midi.model_lighting import Audio2MIDITransformerL, Audio2EncoderL # choose the model you want to train
from dataset import AudioDataModule
from utils.lightning_custom import RichProgressBarCustom


def train(config: DictConfig):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    hparams_train = OmegaConf.to_container(config.train.params, resolve=True)
    max_epochs = hparams_train.pop("epoch", None)
    
    hparams_shared = {
        'audio_length': hparams_data['audio_length'],
        'watch_prev_n_frames': hparams_data['watch_prev_n_frames'],
        'watch_next_n_frames': hparams_data['watch_next_n_frames'],
    }

    if config.model.name == 'Audio2MIDITransformer':
        datamodule = AudioDataModule(t_prev=True, **hparams_data)
        model = Audio2MIDITransformerL(**hparams_model, **hparams_shared, **hparams_train)
    elif config.model.name == 'AudioEncoder':
        datamodule = AudioDataModule(**hparams_data)
        model = Audio2EncoderL(**hparams_model, **hparams_shared, **hparams_train)
    else:
        raise ValueError(f"Model name {config.model.name} not found")
    
    print(OmegaConf.to_yaml(config))

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
cs.store(group="model", name="base_Audio2MIDITransformer_model", node=Audio2MIDITransformerConfig, package="model")
cs.store(group="model", name="base_AudioEncoder_model", node=AudioEncoderConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    train(config)


if __name__ == '__main__':
    main()
