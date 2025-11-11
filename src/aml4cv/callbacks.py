"""Module for Torch training callbacks."""

import logging

from .log_utils import log


class EarlyStopper:
    """Early stopping utility."""

    def __init__(
        self, patience: int = 3, min_delta: float = 0.01, minimize: bool = False
    ):
        """Initialize early stopper.

        Args:
            patience:
                Number of epochs to wait before stopping.
            min_delta:
                Minimum change to qualify as improvement.
            minimize:
                Whether to minimize the metric (True for loss, False for accuracy).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.minimize = minimize
        self.counter = 0
        self.best_metric = float("inf") if minimize else float("-inf")

    def early_stop(self, metric: float) -> bool:
        """Check if should stop early.

        Args:
            metric: Current metric value.

        Returns:
            True if should stop, False otherwise.
        """
        if self.minimize:
            improved = metric < (self.best_metric - self.min_delta)
        else:
            improved = metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
