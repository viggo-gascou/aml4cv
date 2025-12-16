"""Utility functions used for argument parsing in the AML4CV project."""

import argparse
import logging
import os

from ..constants import DATA_DIR, RANDOM_SEED, RESULTS_DIR
from ..log_utils import set_logging_level


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="pretrained",
        choices=["pretrained", "base"],
        help="Model for training",
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debug logging"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--early-stop-criterion",
        type=str,
        default="f1_macro",
        help="Criterion for early stopping",
    )
    parser.add_argument(
        "--patience-min-delta",
        type=float,
        default=0.001,
        help="Minimum delta for early stopping",
    )
    parser.add_argument(
        "--swap-train-test",
        action="store_true",
        default=False,
        help="Swap training and test splits",
    )
    parser.add_argument(
        "-p",
        "--augmentation-proba",
        type=float,
        default=0.1,
        help="Probability of applying each augmentation",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest="num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_DIR),
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "checkpoints"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--short-run",
        action="store_true",
        default=False,
        help="Only run one batch in train and validation for each epoch",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-5,
        help="Minimum learning rate for the learning rate scheduler",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "aml4cv-flowers102"),
        help="W&B project name",
    )

    args = parser.parse_args()

    _parse_log_level(args)
    args.model = _parse_model(args)

    return args


def _parse_model(args: argparse.Namespace) -> str:
    """Parse the model_id from command line arguments."""
    model_ids = {
        "pretrained": "google/vit-base-patch16-224-in21k",
        "base": "base",
    }
    model_id = model_ids[args.model]
    return model_id


def _parse_log_level(args: argparse.Namespace) -> None:
    """Parse log level from command line arguments."""
    if args.debug:
        set_logging_level(logging.DEBUG)
    else:
        # Map verbosity count to log levels
        log_levels = {
            0: logging.WARNING,  # default
            1: logging.INFO,  # -v
            2: logging.DEBUG,  # -vv
        }
        # set to specified level - if more than 2 use DEBUG
        set_logging_level(log_levels.get(args.verbose, logging.DEBUG))
