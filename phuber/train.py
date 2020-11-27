from typing import Optional, Tuple

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from phuber.dataset import NoisyCIFAR10, NoisyCIFAR100, NoisyMNIST
from phuber.transform import cifar10_transform, cifar100_transform, mnist_transform
from phuber.utils import to_clean_str


def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Data
    train_loader, val_loader, test_loader = get_loaders(cfg)
    print(type(train_loader))
    print(len(train_loader))

    print(type(test_loader))
    print(len(test_loader))

    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    model: torch.nn.Module = instantiate(cfg.model)
    loss_fn: torch.nn.Module = instantiate(cfg.loss)
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer, model.parameters()
    )
    scheduler = instantiate(cfg.hparams.scheduler, optimizer)
    print(f"Net: {type(model)}")
    print(f"Loss: {type(loss_fn)}")
    print(f"Optimizer: {type(optimizer)}")
    print(f"Scheduler: {type(scheduler)}")


def get_loaders(
    cfg: DictConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """ Initializes the training, validation, test data & loaders from config

    Args:
        cfg: Hydra config

    Returns:
        Tuple containing the train dataloader, validation dataloader and test dataloader
    """

    name = to_clean_str(cfg.dataset.name)

    if name == "mnist":
        train_transform = mnist_transform()
        test_transform = mnist_transform()
        dataset = NoisyMNIST

    elif name == "cifar10":
        train_transform = cifar10_transform(augment=cfg.dataset.train.augment)
        test_transform = cifar10_transform(augment=False)
        dataset = NoisyCIFAR10

    elif name == "cifar100":
        train_transform = cifar100_transform(augment=cfg.dataset.train.augment)
        test_transform = cifar100_transform(augment=False)
        dataset = NoisyCIFAR100

    else:
        raise ValueError(f"Invalid dataset: {name}")

    root = hydra.utils.to_absolute_path(cfg.dataset.root)

    # Train
    train_set = dataset(
        root,
        train=True,
        transform=train_transform,
        download=cfg.dataset.download,
        corrupt_prob=cfg.dataset.train.corrupt_prob,
        noise_seed=cfg.dataset.train.noise_seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.hparams.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
    )

    # Validation
    val_loader = None

    # Test
    if cfg.dataset.test.use:
        test_set = dataset(
            root,
            train=False,
            transform=test_transform,
            download=cfg.dataset.download,
            corrupt_prob=0,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.hparams.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
