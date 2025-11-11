"""Torch Helpers, Utilities for working with PyTorch tensors and torchvision tensors."""

import torch
from torch import Tensor
from torchvision.tv_tensors import Image


def move_to_device(obj: object, device: torch.device) -> object:
    """Move an object to a specified device.

    Args:
        obj: The object to move.
        device: The device to move the object to.

    Returns:
        The object moved to the specified device.
    """
    if isinstance(obj, (Tensor, Image)):
        return obj.to(device)
    else:
        return obj
