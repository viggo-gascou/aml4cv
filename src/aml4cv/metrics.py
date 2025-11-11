"""Metrics module for the AML4CV project."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import wandb

from .constants import RESULTS_DIR


class Metric(ABC):
    """Abstract base class for computing task-specific metrics."""

    @abstractmethod
    def update(self, predictions: Any, targets: Any) -> None:
        """Update metrics with a batch of predictions and targets."""
        pass

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """Compute and return all metrics."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass


class ClassificationMetric(Metric):
    """Metric computer for image classification tasks."""

    def __init__(self, num_classes: int, task: str = "multiclass"):
        """Initialize the metric computer for image classification tasks.

        Args:
            num_classes:
                Number of classes in the classification task.
            task:
                Type of classification task, either "multiclass" or "binary".
        """
        if task not in ["multiclass", "binary"]:
            raise ValueError(f"Unsupported task: {task}")
        if task == "multiclass":
            from torchmetrics.classification import (
                MulticlassAccuracy,
                MulticlassF1Score,
                MulticlassPrecision,
                MulticlassRecall,
            )

            self.accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
            self.f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
            self.f1_micro = MulticlassF1Score(num_classes=num_classes, average="micro")
            self.precision = MulticlassPrecision(
                num_classes=num_classes, average="macro"
            )
            self.recall = MulticlassRecall(num_classes=num_classes, average="macro")
        elif task == "binary":
            from torchmetrics.classification import (
                BinaryAccuracy,
                BinaryF1Score,
                BinaryPrecision,
                BinaryRecall,
            )

            self.accuracy = BinaryAccuracy()
            self.f1_macro = BinaryF1Score()
            self.f1_micro = BinaryF1Score()
            self.precision = BinaryPrecision()
            self.recall = BinaryRecall()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update with classification predictions and targets.

        Args:
            predictions:
                Logits or class predictions [batch_size, num_classes] or [batch_size]
            targets:
                Ground truth labels [batch_size]
        """
        # If predictions are logits, get the predicted class
        if predictions.dim() > 1:
            preds = predictions.argmax(dim=1)
        else:
            preds = predictions

        self.accuracy.update(preds, targets)
        self.f1_macro.update(preds, targets)
        self.f1_micro.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)

    def compute(self) -> Dict[str, Any]:
        """Compute classification metrics."""
        return {
            "accuracy": self.accuracy.compute(),
            "f1_macro": self.f1_macro.compute(),
            "f1_micro": self.f1_micro.compute(),
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
        }

    def reset(self) -> None:
        """Reset metrics."""
        self.accuracy.reset()
        self.f1_macro.reset()
        self.f1_micro.reset()
        self.precision.reset()
        self.recall.reset()
