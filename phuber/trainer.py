import logging
import os
import time
from typing import Optional

import torch
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from phuber.metrics import AccuracyMetric, LossMetric


class Trainer:
    """Model trainer

    Args:
        model: model to train
        loss_fn: loss function
        optimizer: model optimizer
        epochs: number of epochs
        device: device to train the model on
        train_loader: training dataloader
        val_loader: validation dataloader
        scheduler: learning rate scheduler
        update_sched_on_iter: whether to call the scheduler every iter or every epoch
        grad_clip_max_norm: gradient clipping max norm (disabled if None)
        writer: writer which logs metrics to TensorBoard (disabled if None)
        save_path: folder in which to save models (disabled if None)
        checkpoint_path: path to model checkpoint, to resume training

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional = None,  # Type: torch.optim.lr_scheduler._LRScheduler
        update_sched_on_iter: bool = False,
        grad_clip_max_norm: Optional[float] = None,
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        mixed_precision: bool = False,
    ) -> None:

        # Logging
        self.logger = logging.getLogger()
        self.writer = writer

        # Saving
        self.save_path = save_path

        # Device
        self.device = device

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Model
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.update_sched_on_iter = update_sched_on_iter
        self.grad_clip_max_norm = grad_clip_max_norm
        self.epochs = epochs
        self.start_epoch = 0

        # Floating-point precision
        self.mixed_precision = (
            True if self.device.type == "cuda" and mixed_precision else False
        )
        self.scaler = GradScaler() if self.mixed_precision else None

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        # Metrics
        self.train_loss_metric = LossMetric()
        self.train_acc_metric = AccuracyMetric(k=1)

        self.val_loss_metric = LossMetric()
        self.val_acc_metric = AccuracyMetric(k=1)

    def train(self) -> None:
        """Trains the model"""
        self.logger.info("Beginning training")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            start_epoch_time = time.time()
            if self.mixed_precision:
                self._train_loop_amp(epoch)
            else:
                self._train_loop(epoch)

            if self.val_loader is not None:
                self._val_loop(epoch)

            epoch_time = time.time() - start_epoch_time
            self._end_loop(epoch, epoch_time)

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model(os.path.join(self.save_path, "final_model.pt"), self.epochs)

    def _train_loop(self, epoch: int) -> None:
        """
        Regular train loop

        Args:
            epoch: current epoch
        """
        # Progress bar
        pbar = tqdm.tqdm(total=len(self.train_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Loop
        for data, target in self.train_loader:
            # To device
            data, target = data.to(self.device), target.to(self.device)

            # Forward + backward
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.loss_fn(out, target)
            loss.backward()

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            self.optimizer.step()

            # Update scheduler if it is iter-based
            if self.scheduler is not None and self.update_sched_on_iter:
                self.scheduler.step()

            # Update metrics
            self.train_loss_metric.update(loss.item(), data.shape[0])
            self.train_acc_metric.update(out, target)

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)

        # Update scheduler if it is epoch-based
        if self.scheduler is not None and not self.update_sched_on_iter:
            self.scheduler.step()

        pbar.close()

    def _train_loop_amp(self, epoch: int) -> None:
        """
        Train loop with Automatic Mixed Precision

        Args:
            epoch: current epoch
        """
        # Progress bar
        pbar = tqdm.tqdm(total=len(self.train_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()

        # Loop
        for data, target in self.train_loader:
            # To device
            data, target = data.to(self.device), target.to(self.device)

            # Forward + backward
            self.optimizer.zero_grad()

            # Use amp in forward pass
            with autocast():
                out = self.model(data)
                loss = self.loss_fn(out, target)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            # Update optimizer and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update scheduler if it is iter-based
            if self.scheduler is not None and self.update_sched_on_iter:
                self.scheduler.step()

            # Update metrics
            self.train_loss_metric.update(loss.item(), data.shape[0])
            self.train_acc_metric.update(out, target)

            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)

        # Update scheduler if it is epoch-based
        if self.scheduler is not None and not self.update_sched_on_iter:
            self.scheduler.step()

        pbar.close()

    def _val_loop(self, epoch: int) -> None:
        """
        Standard validation loop

        Args:
            epoch: current epoch
        """
        # Progress bar
        pbar = tqdm.tqdm(total=len(self.val_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        # Set to eval
        self.model.eval()

        # Loop
        for data, target in self.val_loader:
            with torch.no_grad():
                # To device
                data, target = data.to(self.device), target.to(self.device)

                # Forward
                out = self.model(data)
                loss = self.loss_fn(out, target)

                # Update metrics
                self.val_loss_metric.update(loss.item(), data.shape[0])
                self.val_acc_metric.update(out, target)

                # Update progress bar
                pbar.update()
                pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)

        pbar.close()

    def _end_loop(self, epoch: int, epoch_time: float):
        # Print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # Write to tensorboard
        if self.writer is not None:
            self._write_to_tb(epoch)

        # Save model
        if self.save_path is not None:
            self._save_model(os.path.join(self.save_path, "most_recent.pt"), epoch)

        # Clear metrics
        self.train_loss_metric.reset()
        self.train_acc_metric.reset()
        if self.val_loader is not None:
            self.val_loss_metric.reset()
            self.val_acc_metric.reset()

    def _epoch_str(self, epoch: int, epoch_time: float):
        s = f"Epoch {epoch} "
        s += f"| Train loss: {self.train_loss_metric.compute():.3f} "
        s += f"| Train acc: {self.train_acc_metric.compute():.3f} "
        if self.val_loader is not None:
            s += f"| Val loss: {self.val_loss_metric.compute():.3f} "
            s += f"| Val acc: {self.val_acc_metric.compute():.3f} "
        s += f"| Epoch time: {epoch_time:.1f}s"

        return s

    def _write_to_tb(self, epoch):
        self.writer.add_scalar("Loss/train", self.train_loss_metric.compute(), epoch)
        self.writer.add_scalar("Accuracy/train", self.train_acc_metric.compute(), epoch)

        if self.val_loader is not None:
            self.writer.add_scalar("Loss/val", self.val_loss_metric.compute(), epoch)
            self.writer.add_scalar("Accuracy/val", self.val_acc_metric.compute(), epoch)

    def _save_model(self, path, epoch):
        obj = {
            "epoch": epoch + 1,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler": self.scaler.state_dict() if self.mixed_precision else None,
        }
        torch.save(obj, os.path.join(self.save_path, path))

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = checkpoint["epoch"]

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.mixed_precision and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scheduler"])

        if self.start_epoch > self.epochs:
            raise ValueError("Starting epoch is larger than total epochs")

        self.logger.info(f"Checkpoint loaded, resuming from epoch {self.start_epoch}")
