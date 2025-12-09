"""Training utilities for the AML4CV project."""

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import wandb
from safetensors.torch import load_file, save_model
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import VisionTransformer
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from .constants import CLASSES, ID2LABEL, LABEL2ID, RESULTS_DIR
from .dataset import FlowersDataset
from .log_utils import log
from .metrics import Metric
from .utils import move_to_device


def get_model_and_processor(
    model_id: str,
    device: str,
) -> Tuple[ViTForImageClassification | VisionTransformer, ViTImageProcessor]:
    """Get the model and processor for training.

    Args:
        model_id: ID of the model to load.
        device: Device to load the model on.

    Return:
        A tuple of the model and its corresponding processor.
    """
    if model_id == "base":
        model = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=len(CLASSES),
        ).to(device)
        image_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            revision="main",
        )
    else:
        model = ViTForImageClassification.from_pretrained(
            model_id,
            # make the model ready for fine-tuning on new dataset
            num_labels=len(CLASSES),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
            device_map=device,
            revision="main",
        )
        image_processor = ViTImageProcessor.from_pretrained(
            model_id,
            revision="main",
        )

    return model, image_processor


def extract_target_labels(targets) -> torch.Tensor:
    """Extract labels from Target TypedDict objects.

    Args:
        targets: List or tuple of Target objects

    Returns:
        Tensor of labels
    """
    if isinstance(targets[0], dict):
        return torch.tensor([t["label"] for t in targets])
    return targets


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    images, targets = zip(*batch)
    # Extract labels from Target TypedDict
    labels = extract_target_labels(targets)
    return list(images), labels


def get_data_loaders(
    train_transforms: v2.Compose,
    val_test_transforms: v2.Compose,
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get the data loaders for train, val and test.

    Args:
        train_transforms:
            Transformations to apply to the training data.
        val_test_transforms:
            Transformations to apply to the validation and test data.
        args:
            Command-line arguments.
    """
    if args.swap_train_test:
        log("Swapping train and test splits as per argument.", level=logging.WARNING)
        train_dataset = FlowersDataset(
            root=args.data_path, split="test", transforms=train_transforms
        )
        test_dataset = FlowersDataset(
            root=args.data_path, split="train", transforms=val_test_transforms
        )
    else:
        train_dataset = FlowersDataset(
            root=args.data_path, split="train", transforms=train_transforms
        )
        test_dataset = FlowersDataset(
            root=args.data_path, split="test", transforms=val_test_transforms
        )

    val_dataset = FlowersDataset(
        root=args.data_path, split="val", transforms=val_test_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader


def get_data_transforms(
    image_mean: List[float | int],
    image_std: List[float | int],
    image_width: int,
    image_height: int,
    augmentation_proba: float,
) -> Tuple[v2.Compose, v2.Compose]:
    """Get data transformations for training and validation/test.

    Args:
        image_mean:
            Mean values for each channel for normalization.
        image_std:
            Standard deviation values for each channel for normalization.
        image_width:
            Width to resize images to.
        image_height:
            Height to resize images to.
        augmentation_proba:
            Probability of applying augmentations.

    Returns:
        A tuple of (train_transforms, val_test_transforms)
    """
    train_transforms = v2.Compose(
        [
            v2.RandomApply(
                [v2.Pad(padding=10, padding_mode="constant")], p=augmentation_proba
            ),
            v2.RandomApply(
                [v2.RandomRotation(degrees=[-180, 180])], p=augmentation_proba
            ),
            v2.RandomVerticalFlip(p=augmentation_proba),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=(19, 19), sigma=(5.0, 10.0))],
                p=augmentation_proba,
            ),
            v2.RandomApply(
                # about 1.50 of 224 - since inputs are between 500-1168 in height/width
                [v2.CenterCrop(380)],
                p=augmentation_proba,
            ),
            v2.Resize((image_height, image_width)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std),
        ]
    )

    val_test_transforms = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_height, image_width)),
            v2.Normalize(mean=image_mean, std=image_std),
        ]
    )
    return train_transforms, val_test_transforms


def prepare_batch(images, targets, device):  # type: ignore # noqa
    """Prepare batch for model."""
    images = torch.stack(images)
    targets = (
        torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
    )
    images = move_to_device(images, device)
    targets = move_to_device(targets, device)
    return images, targets


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    run: Any,
    loss_fn: torch.nn.modules.loss._Loss,
    prepare_batch_fn: Optional[Callable] = None,
    short_run: bool = False,
    args: argparse.Namespace = None,
) -> Tuple[float, Dict[str, float]]:
    """Train the model for one epoch.

    Args:
        model:
            The model to train
        train_loader:
            DataLoader for training data
        optimizer:
            Optimizer for training
        device:
            Device to run on
        run:
            WandB run object for logging
        loss_fn:
            Loss function.
        prepare_batch_fn:
            Function to prepare batch data. If None, uses default
        short_run:
            If True, only run one batch for testing
        args:
            Command line arguments

    Returns:
        Tuple of (average epoch loss, dict of individual loss components)
    """
    model.train()
    epoch_loss = 0.0
    loss_components: Dict[str, float] = {}

    for batch_data in tqdm(train_loader, desc="Training", leave=False):
        # Prepare batch
        if prepare_batch_fn is not None:
            batch_data = prepare_batch_fn(*batch_data, device)

        # Unpack based on type
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            images, targets = batch_data
        else:
            images = batch_data
            targets = None

        if model.__class__.__name__ == "ViTForImageClassification":
            predictions = model(images).logits
        else:
            predictions = model(images)

        loss = loss_fn(predictions, targets)
        losses = {"loss": loss}

        # Accumulate losses
        run.log({"train/batch_loss": loss.item()})
        epoch_loss += loss.item()

        # Track individual loss components
        if isinstance(losses, dict):
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v.item() if isinstance(v, torch.Tensor) else v

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if short_run:
            break

    # Average the losses
    num_batches = len(train_loader) if not short_run else 1
    epoch_loss /= num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    return epoch_loss, loss_components


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    metric: Metric,
    loss_fn: torch.nn.modules.loss._Loss | None = None,
    prepare_batch_fn: Optional[Callable] = None,
    short_run: bool = False,
) -> Tuple[Dict[str, float], float]:
    """Validate the model and compute metrics.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to run on
        metric: Metric instance for computing task-specific metrics
        compute_loss: Whether to compute validation loss
        loss_fn: Optional custom loss function
        prepare_batch_fn: Function to prepare batch data
        short_run: If True, only run one batch for testing

    Returns:
        Tuple of (metrics dict, average validation loss)
    """
    model.eval()
    metric.reset()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation", leave=False):
            # Prepare batch
            if prepare_batch_fn is not None:
                batch_data = prepare_batch_fn(*batch_data, device)

            # Unpack based on type
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                images, targets = batch_data
            else:
                images = batch_data
                targets = None

            if model.__class__.__name__ == "ViTForImageClassification":
                predictions = model(images).logits
            else:
                predictions = model(images)

            # Compute loss if requested
            if loss_fn is not None:
                loss = loss_fn(predictions, targets)
                epoch_loss += loss.item()

            # Get predictions for metrics
            metric.update(predictions, targets)

            if short_run:
                break

    metrics = metric.compute()

    # Average the losses
    num_batches = len(val_loader) if not short_run else 1
    epoch_loss /= num_batches if num_batches > 0 else 1

    return metrics, epoch_loss


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    metric: Metric,
    prepare_batch_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """Evaluate model on test set and return metrics."""
    model.eval()
    metric.reset()

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            # Prepare batch
            if prepare_batch_fn is not None:
                batch_data = prepare_batch_fn(*batch_data, device)

            # Unpack based on type
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                images, targets = batch_data
            else:
                images = batch_data
                targets = None

            if model.__class__.__name__ == "ViTForImageClassification":
                predictions = model(images).logits
            else:
                predictions = model(images)

            metric.update(predictions, targets)

    return metric.compute()


def save_checkpoint(
    run: Any,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    is_best: bool,
    val_metric: Optional[float] = None,
) -> None:
    """Save model checkpoint."""
    suffix = f"best_epoch_{epoch}" if is_best else f"epoch_{epoch}"
    checkpoint_path = str(Path(run.dir).joinpath(f"model_{suffix}.safetensors"))
    save_model(model, checkpoint_path)

    aliases = ["best_model"] if is_best else []

    # Log to wandb
    run.log_artifact(
        checkpoint_path,
        name=f"model_checkpoint_{suffix}",
        type="model",
        aliases=aliases,
    )

    if val_metric is not None and is_best:
        log(
            f"Saved best model checkpoint at epoch {epoch} with "
            f"val_metric {val_metric:.4f}"
        )


def log_predictions(
    run: Any,
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    num_images: int = 25,
    table_name: str = "predictions",
) -> None:
    """Log classification prediction visualizations to wandb."""
    model.eval()
    table = wandb.Table(columns=["Image", "Ground Truth", "Prediction", "Confidence"])
    img_idx = 0
    log("Logging predictions to wandb, this may take some time...")

    with torch.no_grad():
        for images, targets in data_loader:
            images = torch.stack(images).to(device)
            targets = (
                torch.tensor(targets).to(device)
                if not isinstance(targets, torch.Tensor)
                else targets.to(device)
            )

            if model.__class__.__name__ == "ViTForImageClassification":
                predictions = model(images).logits
            else:
                predictions = model(images)

            # Get predicted classes and confidence
            if predictions.dim() > 1:
                probs = torch.softmax(predictions, dim=1)
                confidences, pred_classes = torch.max(probs, dim=1)
            else:
                pred_classes = predictions
                confidences = torch.ones_like(predictions)

            for img, target, pred_class, confidence in zip(
                images.cpu(), targets.cpu(), pred_classes.cpu(), confidences.cpu()
            ):
                # Convert tensor to numpy and scale
                img = img.add(1).mul(127.5).clamp(0, 255)
                img_np = img.permute(1, 2, 0).numpy()

                gt_label = (
                    class_names[int(target)]
                    if int(target) < len(class_names)
                    else f"Class {int(target)}"
                )
                pred_label = (
                    class_names[int(pred_class)]
                    if int(pred_class) < len(class_names)
                    else f"Class {int(pred_class)}"
                )

                wandb_image = wandb.Image(img_np)
                table.add_data(wandb_image, gt_label, pred_label, float(confidence))
                img_idx += 1

                if img_idx >= num_images:
                    run.log({table_name: table}, commit=True)
                    return

        run.log({table_name: table}, commit=True)


def load_local_checkpoint(
    run,
    model,
    model_path="",
    use_best_checkpoint=False,
) -> Tuple[nn.Module, dict[str, str]]:
    """Load a local checkpoint from the specified paths or the best checkpoint.

    If use_best_checkpoint is True, the best checkpoint will be loaded.

    Args:
        run:
            The wandb run object.
        model:
            The model to load the checkpoint into.
        model_path:
            The path to the model checkpoint file.
        use_best_checkpoint:
            Whether to load the best checkpoint.

    Returns:
            A tuple of the loaded model and checkpoint info.
    """
    checkpoint_info = {}

    if use_best_checkpoint:
        from datetime import datetime

        best_model_path = sorted(
            Path(run.dir).glob("model_best_epoch_*.safetensors"),
            key=lambda file: datetime.fromtimestamp(file.lstat().st_ctime),
        )[-1]
        model_path = best_model_path

        artifact_name = best_model_path.stem
        checkpoint_info["artifact_name"] = (
            f"model_checkpoint_{artifact_name.replace('model_', '')}"
        )
    else:
        model_path = Path(model_path)
        checkpoint_info["artifact_name"] = None

    checkpoint_info["model_path"] = str(model_path)

    if model_path.exists():
        model_weights = load_file(model_path)
        model.load_state_dict(model_weights)
        return model, checkpoint_info
    else:
        return model, checkpoint_info
