from typing import Optional, Callable
import numpy as np

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100


class NoisyMNIST(MNIST):
    """Extends `torchvision.datasets.MNIST <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`_
    class by adding symmetric label noise with a specific probability
    """

    num_classes = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        noise_prob: float = 0.0,
        noise_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.noise_prob = noise_prob
        self.noise_seed = noise_seed
        self._add_label_noise()

    def _add_label_noise(self):
        if self.noise_prob < 0 or self.noise_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.noise_prob}")

        if self.noise_prob > 0:
            if self.noise_seed is not None:
                np.random.seed(self.noise_seed)

            p = np.ones((len(self.targets), self.num_classes))
            p = p * (self.noise_prob / (self.num_classes - 1))
            p[np.arange(len(self.targets)), self.targets.numpy()] = 1 - self.noise_prob

            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])


class NoisyCIFAR10(CIFAR10):
    """Extends `torchvision.datasets.CIFAR10 <https://pytorch.org/docs/stable/torchvision/datasets.html#cifar>`_
    class by adding symmetric label noise with a specific probability
    """

    num_classes = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        noise_prob: float = 0.0,
        noise_seed: Optional[int] = None,
    ) -> None:

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.noise_prob = noise_prob
        self.noise_seed = noise_seed
        self._add_label_noise()

    def _add_label_noise(self):
        if self.noise_prob < 0 or self.noise_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.noise_prob}")

        if self.noise_prob > 0:
            if self.noise_seed is not None:
                np.random.seed(self.noise_seed)

            p = np.ones((len(self.targets), self.num_classes))
            p = p * (self.noise_prob / (self.num_classes - 1))
            p[np.arange(len(self.targets)), self.targets.numpy()] = 1 - self.noise_prob

            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])


class NoisyCIFAR100(CIFAR100):
    """Extends `torchvision.datasets.CIFAR100 <https://pytorch.org/docs/stable/torchvision/datasets.html#cifar>`_
    class by adding symmetric label noise with a specific probability
    """

    num_classes = 100

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        noise_prob: float = 0.0,
        noise_seed: Optional[int] = None,
    ) -> None:

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.noise_prob = noise_prob
        self.noise_seed = noise_seed
        self._add_label_noise()

    def _add_label_noise(self):
        if self.noise_prob < 0 or self.noise_prob > 1:
            raise ValueError(f"Invalid noise probability: {self.noise_prob}")

        if self.noise_prob > 0:
            if self.noise_seed is not None:
                np.random.seed(self.noise_seed)

            p = np.ones((len(self.targets), self.num_classes))
            p = p * (self.noise_prob / (self.num_classes - 1))
            p[np.arange(len(self.targets)), self.targets.numpy()] = 1 - self.noise_prob

            for i in range(len(self.targets)):
                self.targets[i] = np.random.choice(self.num_classes, p=p[i])
