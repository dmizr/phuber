import collections
import logging
import os
from typing import Optional, Tuple, TypeVar

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from phuber.dataset import NoisyCIFAR10, NoisyCIFAR100, NoisyMNIST
from phuber.evaluator import Evaluator
from phuber.scheduler import ExponentialDecayLR
from phuber.trainer import Trainer
from phuber.transform import cifar10_transform, cifar100_transform, mnist_transform
from phuber.utils import flatten, to_clean_str

T = TypeVar("T")


def train(cfg: DictConfig) -> None:
    """Trains model from config

    Args:
        cfg: Hydra config

    """
    # Logger
    logger = logging.getLogger()

    # Device
    device = get_device(cfg)

    # Data
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    model: torch.nn.Module = instantiate(cfg.model).to(device)
    loss_fn: torch.nn.Module = instantiate(cfg.loss).to(device)
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer, model.parameters()
    )
    scheduler = instantiate(cfg.hparams.scheduler, optimizer)
    update_sched_on_iter = True if isinstance(scheduler, ExponentialDecayLR) else False

    # Paths
    save_path = os.getcwd() if cfg.save else None
    checkpoint_path = (
        hydra.utils.to_absolute_path(cfg.checkpoint)
        if cfg.checkpoint is not None
        else None
    )

    # Tensorboard
    if cfg.tensorboard:
        # Note: global step is in epochs here
        writer = SummaryWriter(os.getcwd())
        # Indicate to TensorBoard that the text is pre-formatted
        text = f"<pre>{OmegaConf.to_yaml(cfg)}</pre>"
        writer.add_text("config", text)
    else:
        writer = None

    # Trainer init
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
        logger.info("\nEvaluating on test data")
        evaluator = Evaluator(
            model=model, device=device, loader=test_loader, checkpoint_path=None
        )
        accuracy = evaluator.evaluate()

        if writer:
            writer.add_scalar("Eval/Accuracy/test", accuracy, -1)

        if cfg.tensorboard:
            res_path = hydra.utils.to_absolute_path(f"results/{cfg.dataset.name}/")
            params = flatten(OmegaConf.to_container(cfg, resolve=True))
            with SummaryWriter(res_path) as w:
                w.add_hparams(params, {"accuracy": accuracy})


def evaluate(cfg: DictConfig) -> None:
    """Evaluates model from config

    Args:
        cfg: Hydra config
    """
    # Logger
    logger = logging.getLogger()

    # Device
    device = get_device(cfg)

    # Data
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # Model
    model: torch.nn.Module = instantiate(cfg.model).to(device)

    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)

    if train_loader is not None:
        logger.info("Evaluating on training data")
        evaluator = Evaluator(
            model=model,
            device=device,
            loader=train_loader,
            checkpoint_path=checkpoint_path,
        )
        evaluator.evaluate()
        # Remove checkpoint loading for other loaders
        checkpoint_path = None

    if val_loader is not None:
        logger.info("Evaluating on validation data")
        evaluator = Evaluator(
            model=model,
            device=device,
            loader=test_loader,
            checkpoint_path=checkpoint_path,
        )
        evaluator.evaluate()
        # Remove checkpoint loading for other loaders
        checkpoint_path = None

    if test_loader is not None:
        logger.info("Evaluating on test data")
        evaluator = Evaluator(
            model=model,
            device=device,
            loader=test_loader,
            checkpoint_path=checkpoint_path,
        )
        evaluator.evaluate()


def get_device(cfg: DictConfig) -> torch.device:
    """Initializes the device from config

    Args:
        cfg: Hydra config

    Returns:
        device on which the model will be trained or evaluated

    """
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
    """Initializes the training, validation, test data & loaders from config

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
        if cfg.dataset.val.use and cfg.dataset.val.split is not None:
            train_set, _ = split_dataset(
                dataset=train_set,
                split=cfg.dataset.val.split,
                seed=cfg.dataset.val.seed,
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
    if cfg.dataset.val.use:
        if cfg.dataset.val.split is not None and cfg.dataset.val.split != 0.0:
            val_set = dataset(
                root,
                train=True,
                transform=test_transform,
                download=cfg.dataset.download,
                corrupt_prob=cfg.dataset.train.corrupt_prob,
                noise_seed=cfg.dataset.train.noise_seed,
            )
            _, val_set = split_dataset(
                dataset=val_set,
                split=cfg.dataset.val.split,
                seed=cfg.dataset.val.seed,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=cfg.hparams.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
            )

        else:
            logger = logging.getLogger()
            logger.info("No validation set will be used, as no split value was given.")
            val_loader = None
    else:
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
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def split_dataset(dataset: Dataset, split: float, seed: int) -> Tuple[Subset, Subset]:
    """Splits dataset into a train / val set based on a split value and seed

    Args:
        dataset: dataset to split
        split: The proportion of the dataset to include in the validation split,
            must be between 0 and 1.
        seed: Seed used to generate the split

    Returns:
        Subsets of the input dataset

    """
    # Verify that the dataset is Sized
    if not isinstance(dataset, collections.abc.Sized):
        raise ValueError("Dataset is not Sized!")

    if not (0 <= split <= 1):
        raise ValueError(f"Split value must be between 0 and 1. Value: {split}")

    val_length = int(len(dataset) * split)
    train_length = len(dataset) - val_length
    splits = random_split(
        dataset,
        [train_length, val_length],
        generator=torch.Generator().manual_seed(seed),
    )
    return splits
