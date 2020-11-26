from typing import Tuple, Callable

import torch
import torchvision.transforms as transforms


def transform_mnist() -> Callable:
    """PIL Image to Tensor transform for MNIST, with standardization

    Returns:
        transform function
    """

    # Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    mean = (0.1307,)
    std = (0.3081,)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    return transform


def transform_cifar(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    augment: bool = True,
) -> Callable:
    """PIL Image to Tensor transform for CIFAR, with standardization and data augmentation
    Args:
        augment: if True, adds random horizontal flip and random cropping
        mean: RGB channels mean
        std: RGB channels standard deviation


    Returns:
        transform function
    """
    if augment:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    return transform


def transform_cifar10(augment: bool = True) -> Callable:
    """PIL Image to Tensor transform for CIFAR-10, with standardization and data augmentation
    Args:
        augment: if True, adds random horizontal flip and random cropping

    Returns:
        transform function
    """
    # Source: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return transform_cifar(mean, std, augment)


def transform_cifar100(augment: bool = True) -> Callable:
    """PIL Image to Tensor transform for CIFAR-100, with standardization and data augmentation
    Args:
        augment: if True, adds random horizontal flip and random cropping

    Returns:
        transform function
    """
    # Source: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    return transform_cifar(mean, std, augment)
