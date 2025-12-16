"""Module for data loading and preprocessing."""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

from torchvision.datasets import Flowers102, VisionDataset
from torchvision.io import decode_image
from torchvision.tv_tensors import Image

from .constants import DATA_DIR, ID2LABEL
from .data_models import Target
from .log_utils import log


class FlowersDataset(VisionDataset):
    """Flowers dataset class."""

    def __init__(
        self,
        split: str,
        root: str | Path = DATA_DIR,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        """Initialize the Flowers Dataset.

        Args:
            root:
                Path to the root directory of the dataset.
            split:
                Split of the dataset to use.
            image_ids:
                List of image IDs to load. If None, all images are loaded.
            transform:
                Optional transform to apply to the images.
            target_transform:
                Optional transform to apply to the labels.
            transforms:
                Optional transforms to apply to the images and labels.
        """
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(root) if isinstance(root, str) else root
        self.split = split
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory {self.root} does not exist.")
        if not self.split:
            raise ValueError("Split is not set, please provide a valid split.")
        log(
            f"Initializing Flowers Dataset from {self.root}, with the split "
            f"{self.split}"
        )

        self.id_to_class = ID2LABEL
        self.images, self.targets = self._load_data()
        self.ids = list(self.images.keys())
        log(f"Successfully loaded {len(self.ids)} samples!", level=logging.DEBUG)

    def _load_data(
        self,
    ) -> Tuple[dict, dict]:
        dataset = Flowers102(
            self.root,
            split=self.split,
            download=True,
            loader=decode_image,  # pyright: ignore [reportArgumentType]
        )
        images: dict[int, dict] = {}
        targets: dict[int, dict] = {}

        for i in range(len(dataset)):
            image, label = dataset[i]
            images[i] = image
            targets[i] = label

        return images, targets

    def _load_image(self, id: int) -> Image:
        image_tensor = self.images[id]
        return Image(image_tensor)

    def _load_target(self, id: int) -> Target:
        label = self.targets[id]
        return Target(label=label, class_name=self.id_to_class[label], id=id)

    def __getitem__(self, index: int) -> Tuple[Image, Target]:
        """Return a sample from the dataset."""
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ids)
