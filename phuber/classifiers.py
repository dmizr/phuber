import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from omegaconf import DictConfig
import hydra


class MNISTClassifier(pl.LightningModule):
    def __init__(self, net: nn.Module, loss_fn: nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        y_hat = self.net(x)
        return F.softmax(y_hat, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_acc",
            self.train_acc(y_hat, y),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )

        self.log(
            "val_acc",
            self.val_acc(y_hat, y),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)

        self.log_dict(
            {"test_loss": loss, "test_acc": self.test_acc(y_hat, y)},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.cfg.model.optimizer, self.net.parameters())
