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

from audio2midi import (
    get_model_class,
    DataConfig, 
    TrainConfig, 
    AudioTransformerConfig, 
    AudioTransformerEncoderConfig, 
    AudioStartConvConfig, 
    AudioStartConformerConfig
)
from dataset import AudioDataModule
from utils.lightning_custom import RichProgressBarCustom


def train(config: DictConfig):
    hparams_data = OmegaConf.to_container(config.data.params, resolve=True)
    hparams_model = OmegaConf.to_container(config.model.params, resolve=True)
    hparams_train = OmegaConf.to_container(config.train.params, resolve=True)
    max_epochs = hparams_train.pop("epoch", None)

    hparams_shared = {
        'watch_prev_n_frames': hparams_data['watch_prev_n_frames'],
        'watch_next_n_frames': hparams_data['watch_next_n_frames'],
    }

    t_prev = True if config.model.name == 'AudioTransformer' else False
    datamodule = AudioDataModule(t_prev, **hparams_data)

    model_class = get_model_class(config.model.name)
    model = model_class(**hparams_model, **hparams_shared, **hparams_train)
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
cs.store(group="model", name="base_AudioTransformer", node=AudioTransformerConfig, package="model")
cs.store(group="model", name="base_AudioTransformerEncoder", node=AudioTransformerEncoderConfig, package="model")
cs.store(group="model", name="base_AudioStartConv", node=AudioStartConvConfig, package="model")
cs.store(group="model", name="base_AudioStartConformer", node=AudioStartConformerConfig, package="model")


@hydra.main(config_path=os.path.join('..', "configs"), config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    # warnings.filterwarnings('ignore')
    train(config)


if __name__ == '__main__':
    main()
