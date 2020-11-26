import os

import torch
import hydra

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from phuber.dataset import NoisyMNIST


def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    model: torch.nn.Module = instantiate(cfg.model)
    loss_fn: torch.nn.Module = instantiate(cfg.loss)
    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer, net.parameters()
    )
    scheduler = instantiate(cfg.hparams.scheduler, optimizer)
    print(f"Net: {type(model)}")
    print(f"Loss: {type(loss_fn)}")
    print(f"Optimizer: {type(optimizer)}")
    print(f"Scheduler: {type(scheduler)}")
