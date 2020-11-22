import pytorch_lightning as pl
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

from phuber.datasets import NoisyMNIST
from phuber.lenet import LeNet5
from phuber.classifiers import MNISTClassifier

if __name__ == "__main__":
    # Directories
    save_dir = "./experiments/checkpoints/"
    logger_dir = "./experiments/tb_logs/"
    data_dir = "./data/"

    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_data = NoisyMNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform,
        noise_prob=0,
        noise_seed=0,
    )

    test_data = NoisyMNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform,
        noise_prob=0,
        noise_seed=None,
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # Models
    net = LeNet5()
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.005
    model = MNISTClassifier(net, loss_fn, lr)

    # Logger
    logger = TensorBoardLogger(save_dir=logger_dir, name="test_model")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=save_dir)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Devices

    # Trainer
    trainer = Trainer(
        logger=logger, callbacks=[checkpoint_callback, early_stop_callback]
    )
    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)
