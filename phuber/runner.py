import logging
import os
from typing import Optional, Tuple

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from phuber.dataset import NoisyCIFAR10, NoisyCIFAR100, NoisyMNIST
from phuber.scheduler import ExponentialDecayLR
from phuber.trainer import Trainer
from phuber.transform import cifar10_transform, cifar100_transform, mnist_transform
from phuber.utils import flatten, to_clean_str


def train(cfg: DictConfig):
    logger = logging.getLogger()
    logger.info(type(cfg.resume))
    # Data
    train_loader, val_loader, test_loader = get_loaders(cfg)
    logger.info(type(train_loader))
    logger.info(len(train_loader))

    logger.info(type(test_loader))
    logger.info(len(test_loader))

    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    model: torch.nn.Module = instantiate(cfg.model)
    loss_fn: torch.nn.Module = instantiate(cfg.loss)
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer, model.parameters()
    )
    scheduler = instantiate(cfg.hparams.scheduler, optimizer)
    update_sched_on_iter = True if isinstance(scheduler, ExponentialDecayLR) else False

    logger.info(f"Net: {type(model)}")
    logger.info(f"Loss: {type(loss_fn)}")
    logger.info(f"Optimizer: {type(optimizer)}")
    logger.info(f"Scheduler: {type(scheduler)}")

    writer = SummaryWriter(os.getcwd()) if cfg.tensorboard else None
    save_path = os.getcwd() if cfg.save else None
    checkpoint_path = (
        hydra.utils.to_absolute_path(cfg.resume) if cfg.resume is not None else None
    )

    device = get_device(cfg)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=cfg.hparams.epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        update_sched_on_iter=update_sched_on_iter,
        grad_clip_val=cfg.hparams.grad_clip_val,
        writer=writer,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
    )

    # Launch training process
    trainer.train()

    if test_loader is not None:
        if writer is not None:
            res_path = hydra.utils.to_absolute_path(f"results/{cfg.name}/")
            params = flatten(OmegaConf.to_container(cfg, resolve=True))
            for k, v in params.items():
                print(f"{k} : {v} | {type(v)}")
            with SummaryWriter(res_path) as w:
                w.add_hparams(params, {"loss": 10})


def get_device(cfg: DictConfig) -> torch.device:
    if cfg.auto_cpu_if_no_gpu:
        device = (
            torch.device(cfg.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(cfg.device)

    return device


def get_loaders(
    cfg: DictConfig,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
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
    if cfg.dataset.train.use:
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
    else:
        train_loader = None

    # Validation
    # TODO
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
