import collections
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


class NoisyMNIST(MNIST):
    """Extends `torchvision.datasets.MNIST
    <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`_
    class by corrupting the labels with a fixed probability
    """

    num_classes = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        corrupt_prob: float = 0.0,
        noise_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.corrupt_prob = corrupt_prob
        self.noise_seed = noise_seed
        self._add_label_noise()

    def _add_label_noise(self) -> None:
        if self.corrupt_prob < 0 or self.corrupt_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.corrupt_prob}")

        if self.corrupt_prob == 0:
            return

        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)

        p = np.ones((len(self.targets), self.num_classes))
        p = p * (self.corrupt_prob / (self.num_classes - 1))
        p[np.arange(len(self.targets)), self.targets] = 1 - self.corrupt_prob

        for i in range(len(self.targets)):
            self.targets[i] = np.random.choice(self.num_classes, p=p[i])


class NoisyCIFAR10(CIFAR10):
    """Extends `torchvision.datasets.CIFAR10
    <https://pytorch.org/docs/stable/torchvision/datasets.html#cifar>`_
    class by corrupting the labels with a fixed probability
    """

    num_classes = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        corrupt_prob: float = 0.0,
        noise_seed: Optional[int] = None,
    ) -> None:

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.corrupt_prob = corrupt_prob
        self.noise_seed = noise_seed
        self._add_label_noise()

    def _add_label_noise(self) -> None:
        if self.corrupt_prob < 0 or self.corrupt_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.corrupt_prob}")

        if self.corrupt_prob == 0:
            return

        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)

        p = np.ones((len(self.targets), self.num_classes))
        p = p * (self.corrupt_prob / (self.num_classes - 1))
        p[np.arange(len(self.targets)), self.targets] = 1 - self.corrupt_prob

        for i in range(len(self.targets)):
            self.targets[i] = np.random.choice(self.num_classes, p=p[i])


class NoisyCIFAR100(CIFAR100):
    """Extends `torchvision.datasets.CIFAR100
    <https://pytorch.org/docs/stable/torchvision/datasets.html#cifar>`_
    class by corrupting the labels with a fixed probability
    """

    num_classes = 100

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        corrupt_prob: float = 0.0,
        noise_seed: Optional[int] = None,
    ) -> None:

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.corrupt_prob = corrupt_prob
        self.noise_seed = noise_seed
        self._add_label_noise()

    def _add_label_noise(self) -> None:
        if self.corrupt_prob < 0 or self.corrupt_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.corrupt_prob}")

        if self.corrupt_prob == 0:
            return

        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)

        p = np.ones((len(self.targets), self.num_classes))
        p = p * (self.corrupt_prob / (self.num_classes - 1))
        p[np.arange(len(self.targets)), self.targets] = 1 - self.corrupt_prob

        for i in range(len(self.targets)):
            self.targets[i] = np.random.choice(self.num_classes, p=p[i])


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
