"""Metrics module for the AML4CV project."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


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

    def __init__(
        self,
        num_classes: int,
    ):
        """Initialize the metric computer for image classification tasks.

        Args:
            num_classes:
                Number of classes in the classification task.
        """
        from torchmetrics.classification import (
            MulticlassAccuracy,
            MulticlassF1Score,
            MulticlassPrecision,
            MulticlassRecall,
        )

        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.f1_micro = MulticlassF1Score(num_classes=num_classes, average="micro")
        self.precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.recall = MulticlassRecall(num_classes=num_classes, average="macro")

        self.metrics = {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_micro": self.f1_micro,
            "precision": self.precision,
            "recall": self.recall,
        }

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update with classification predictions and targets.

        Args:
            predictions:
                Logits or class predictions [batch_size, num_classes] or [batch_size]
            targets:
                Ground truth labels [batch_size]
        """
        for metric in self.metrics.values():
            metric.update(predictions.cpu(), targets.cpu())

    def compute(self) -> Dict[str, Any]:
        """Compute classification metrics."""
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        """Reset metrics."""
        for metric in self.metrics.values():
            metric.reset()
