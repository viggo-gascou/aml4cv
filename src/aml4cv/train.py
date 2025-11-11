"""Training utilities for the AML4CV project."""

import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import wandb
from safetensors.torch import save_model
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from .constants import CLASSES, ID2LABEL, LABEL2ID, RESULTS_DIR
from .dataset import FlowersDataset
from .metrics import Metric
from .utils import move_to_device


def get_model_and_processor(
    model_id: str,
    device: str,
) -> Tuple[ViTForImageClassification, ViTImageProcessor]:
    """Get the model and processor for training.

    Args:
        model_id: ID of the model to load.
        device: Device to load the model on.

    Return:
        A tuple of the model and its corresponding processor.
    """
    if model_id == "google/vit-large-patch32-224-in21k":
        revision = "refs/pr/1"
    else:
        revision = "main"
    model = ViTForImageClassification.from_pretrained(
        model_id,
        # make the model ready for fine-tuning on new dataset
        num_labels=len(CLASSES),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
        device_map=device,
        revision=revision,
    )
    image_processor = ViTImageProcessor.from_pretrained(
        model_id,
        revision=revision,
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
    train_dataset = FlowersDataset(
        root=args.data_path, split="train", transforms=train_transforms
    )
    val_dataset = FlowersDataset(
        root=args.data_path, split="val", transforms=val_test_transforms
    )
    test_dataset = FlowersDataset(
        root=args.data_path, split="test", transforms=val_test_transforms
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

        predictions = model(images).logits
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

            # Compute loss if requested
            if loss_fn is not None:
                predictions = model(images).logits
                loss = loss_fn(predictions, targets)
                epoch_loss += loss.item()

            # Get predictions for metrics
            predictions = model(images).logits
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

            predictions = model(images).logits
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
    # Regular checkpoint
    if epoch % 5 == 0 or is_best:
        suffix = f"best_epoch_{epoch}" if is_best else f"epoch_{epoch}"
        checkpoint_path = str(RESULTS_DIR.joinpath(f"model_{suffix}.safetensors"))
        save_model(model, checkpoint_path)

        if optimizer:
            optimizer_path = str(RESULTS_DIR.joinpath(f"optimizer_{suffix}.pth"))
            torch.save(optimizer.state_dict(), optimizer_path)

        # Log to wandb
        aliases = ["best_model"] if is_best else []
        run.log_artifact(
            checkpoint_path,
            name=f"model_checkpoint_{suffix}",
            type="model",
            aliases=aliases,
        )

        if val_metric is not None and is_best:
            print(
                f"Saved best model checkpoint at epoch {epoch} with "
                f"val_metric {val_metric:.4f}"
            )
        else:
            print(f"Saved checkpoint at epoch {epoch}")


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

    with torch.no_grad():
        for images, targets in data_loader:
            images = torch.stack(images).to(device)
            targets = (
                torch.tensor(targets).to(device)
                if not isinstance(targets, torch.Tensor)
                else targets.to(device)
            )

            predictions = model(images).logits

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
