import logging
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2
from tqdm import tqdm

from aml4cv.callbacks import EarlyStopper
from aml4cv.constants import CLASSES, RESULTS_DIR
from aml4cv.log_utils import log, set_logging_level
from aml4cv.metrics import ClassificationMetric
from aml4cv.train import (
    evaluate,
    get_data_loaders,
    get_data_transforms,
    get_model_and_processor,
    load_local_checkpoint,
    log_predictions,
    prepare_batch,
    save_checkpoint,
    save_model,
    train_one_epoch,
    validate,
)
from aml4cv.utils import parse_args

set_logging_level(logging.INFO)


def train() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, but CUDA was requested!")
    torch_device = torch.device(args.device)

    log(f"Using device: {args.device}")
    model_id = args.model

    training_config = {
        "model_name": model_id,
        "num_classes": len(CLASSES),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "augmentation_proba": args.augmentation_proba,
        "seed": args.seed,
        "early_stopping_criterion": args.early_stop_criterion,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.patience_min_delta,
    }

    # Setup W&B
    log("Initializing Weights & Biases...")
    wandb.login()
    run = wandb.init(
        project=args.wandb_project,
        name=f"{model_id.split('/')[-1]}_flowers102",
        config=training_config,
        tags=["short_run"] if args.short_run else [],
    )
    # set log level again after wandb init because they clear the logging :(
    set_logging_level(logging.INFO)

    model, processor = get_model_and_processor(model_id, args.device)
    image_mean = (
        processor.image_mean
        if isinstance(processor.image_mean, list)
        else [processor.image_mean] * 3
    )
    image_std = (
        processor.image_std
        if isinstance(processor.image_std, list)
        else [processor.image_std] * 3
    )
    image_width, image_height = processor.size["width"], processor.size["height"]
    train_transforms, val_test_transforms = get_data_transforms(
        image_mean, image_std, image_width, image_height, args.augmentation_proba
    )

    train_loader, val_loader, test_loader = get_data_loaders(
        train_transforms, val_test_transforms, args
    )

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = run.config
    log(f"Training configuration:\n{dict(config)}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=args.min_learning_rate,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Metric computer
    metric = ClassificationMetric(num_classes=len(CLASSES))

    # Early stopper
    early_stopper = EarlyStopper(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        minimize=False,  # False for accuracy (higher is better)
    )

    # Training loop
    log("Starting training...")
    best_val_metric = float("-inf")  # For accuracy (higher is better)
    is_best = False

    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        # Train
        train_loss, loss_components = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch_device,
            run=run,
            loss_fn=criterion,
            prepare_batch_fn=prepare_batch,
            short_run=args.short_run,
        )

        # Validate
        val_metrics, val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=torch_device,
            metric=metric,
            loss_fn=criterion,
            short_run=args.short_run,
            prepare_batch_fn=prepare_batch,
        )

        # Update learning rate
        lr_scheduler.step()

        # Log metrics
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": val_metrics["accuracy"],
            "val/f1_macro": val_metrics["f1_macro"],
            "val/f1_micro": val_metrics["f1_micro"],
            "val/precision": val_metrics["precision"],
            "val/recall": val_metrics["recall"],
            "learning_rate": lr_scheduler.get_last_lr()[0],
        }

        run.log(metrics)

        # Print progress
        log(
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
            f"Val F1 (macro): {val_metrics['f1_macro']:.4f}"
        )

        # Determine metric for checkpointing
        val_metric = val_metrics[config.early_stopping_criterion]
        is_best = val_metric > best_val_metric

        if is_best:
            best_val_metric = val_metric
            log(f"New best model! Accuracy: {best_val_metric:.4f}")
            # Save checkpoint
            save_checkpoint(
                run=run,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                is_best=is_best,
                val_metric=val_metric if is_best else None,
            )

        if args.short_run:
            log("Doing a short run, so not logging models or predictions.")
            run.finish()
            return

        log_predictions(
            run=run,
            model=model,
            data_loader=val_loader,
            device=torch_device,
            class_names=CLASSES,
            num_images=25,
            table_name=f"val/epoch_{epoch}_predictions",
        )

        # Early stopping
        if early_stopper.early_stop(val_metric):
            log(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # if last epoch was not the best, load the best model
    best_checkpoint_info = None
    if not is_best:
        model, best_checkpoint_info = load_local_checkpoint(
            run=run,
            model=model,
            use_best_checkpoint=True,
        )
        log(f"Loaded best checkpoint: {best_checkpoint_info['artifact_name']}")

    # Final evaluation on test set
    log("\nEvaluating on test set...")
    test_metrics = evaluate(
        model=model,
        test_loader=test_loader,
        device=torch_device,
        metric=metric,
        prepare_batch_fn=prepare_batch,
    )

    # Log final test metrics
    run.summary["test/accuracy"] = test_metrics["accuracy"]
    run.summary["test/f1_macro"] = test_metrics["f1_macro"]
    run.summary["test/f1_micro"] = test_metrics["f1_micro"]
    run.summary["test/precision"] = test_metrics["precision"]
    run.summary["test/recall"] = test_metrics["recall"]

    log("\nTest Results:")
    log(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    log(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    log(f"  F1 (micro): {test_metrics['f1_micro']:.4f}")
    log(f"  Precision: {test_metrics['precision']:.4f}")
    log(f"  Recall: {test_metrics['recall']:.4f}")

    # Add "final" alias to the best model artifact
    if best_checkpoint_info and best_checkpoint_info["artifact_name"]:
        log(
            f"Adding 'final' alias to artifact: {best_checkpoint_info['artifact_name']}"
        )
        best_artifact = run.use_artifact(
            f"{best_checkpoint_info['artifact_name']}:latest"
        )
        best_artifact.aliases.append("model_final")
        best_artifact.save()
    else:
        # Last epoch was the best, save the current model as final
        final_model_path = str(RESULTS_DIR / "final_model.safetensors")
        save_model(model, final_model_path)
        run.log_artifact(
            final_model_path,
            name="final_model",
            type="model",
            aliases=["model_final"],
        )

    # Log test predictions
    log_predictions(
        run=run,
        model=model,
        data_loader=test_loader,
        device=torch_device,
        class_names=CLASSES,
        num_images=50,
        table_name="test/predictions",
    )
    if best_checkpoint_info:
        # save the final model to local folder
        final_model_path = str(Path(run.dir) / "final_model.safetensors")
        save_model(model, final_model_path)

    run.finish()
    log("Training complete!")


if __name__ == "__main__":
    train()
