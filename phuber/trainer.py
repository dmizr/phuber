import os

import torch
import torchvision.transforms as transforms
import hydra


from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from phuber.datasets import NoisyMNIST
from phuber.classifiers import MNISTClassifier


def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    # Directories
    data_dir = hydra.utils.to_absolute_path(cfg.dataset.path)
    print(data_dir)

    save_dir = "./checkpoints/"
    logger_dir = "./tb_logs/"

    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_data = NoisyMNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform,
        corrupt_prob=cfg.dataset.corrupt_prob,
        noise_seed=cfg.dataset.noise_seed,
    )

    test_data = NoisyMNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform,
        corrupt_prob=0,
        noise_seed=None,
    )

    train_loader = DataLoader(
        train_data, batch_size=cfg.model.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_data, batch_size=cfg.model.batch_size, shuffle=False, num_workers=4
    )

    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    net: torch.nn.Module = instantiate(cfg.model.network)
    loss_fn: torch.nn.Module = instantiate(cfg.loss)
    optim: torch.optim.Optimizer = instantiate(cfg.model.optimizer, net.parameters())

    model = MNISTClassifier(net, loss_fn, optim)

    # Logger
    logger = TensorBoardLogger(save_dir=logger_dir, name="test_model")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=save_dir)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Devices

    # Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=cfg.model.epochs,
    )
    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
