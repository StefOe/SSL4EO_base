from pathlib import Path

import wandb
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data import GeobenchDataset
from eval.helper_modules import (
    LinearMultiLabelClassifier, FinetuneMultiLabelClassifier, LinearClassifier, FinetuneEvalClassifier,
)


def geobench_clf_eval(
    model: Module,
    dataset_name: str,
    partition: str,
    method: str,  # "either finetune or linear
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    debug: [bool, str] = False,
) -> None:
    """Runs a linear evaluation on the given model.

    Parameters follow SimCLR [0] settings.

    The most important settings are:
        - Backbone: Frozen
        - Epochs: 90
        - Optimizer: SGD
        - Base Learning Rate: 0.1
        - Momentum: 0.9
        - Weight Decay: 0.0
        - LR Schedule: Cosine without warmup

    References:
        - [0]: SimCLR, 2020, https://arxiv.org/abs/2002.05709
    """
    assert dataset_name in [
        "m-eurosat",
        "m-so2sat",
        "m-bigearthnet",
    ], f"dataset '{dataset_name}' not supported"
    print_rank_zero("Running geobench evaluation...")

    # Setup training data.

    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
    )
    train_dataset = GeobenchDataset(
        dataset_name=dataset_name,
        split="train",
        partition=partition,
        transform=train_transform,
        benchmark_name="classification",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # Setup validation data.
    val_dataset = GeobenchDataset(
        dataset_name=dataset_name,
        split="val",
        transform=None,
        benchmark_name="classification",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    # Train linear classifier.
    metric_callback = MetricCallback()
    model_checkpoint = ModelCheckpoint(
        monitor="val_top1",
        mode="max", auto_insert_metric_name=True
    )
    epochs = 90 if method == "linear" else 30
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            model_checkpoint,
            metric_callback,
        ],
        logger=WandbLogger(
            save_dir=str(log_dir),
            name=f"{dataset_name}_{method}_eval",
            project="ssl4eo",
            # log model config
            config=model.hparams,
            offline=debug,
        ),
        precision=precision,
        # strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,

        # this is just for debug
        fast_dev_run=debug and debug != "long",
        limit_train_batches=1 if debug else None,
        limit_val_batches=1 if debug else None,
        limit_test_batches=1 if debug else None,
    )

    classifier = get_geobench_classifier(
        model,
        method,
        is_multi_label=dataset_name == "m-bigearthnet",
        num_classes=train_dataset.num_classes,
        batch_size_per_device=batch_size_per_device,
    )

    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # clean memory
    del train_dataloader, val_dataloader

    if not debug:
        print_rank_zero(
            f"max {dataset_name} {method} val_top1: {max(metric_callback.val_metrics['val_top1'])}"
        )

    # get test results for best val model
    test_dataset = GeobenchDataset(
        dataset_name=dataset_name,
        split="test",
        transform=None,
        benchmark_name="classification",
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    best_model_path = model_checkpoint.best_model_path if model_checkpoint.best_model_path != "" else None
    trainer.test(
        model=classifier,
        dataloaders=test_dataloader,
        ckpt_path=best_model_path,
    )

    wandb.finish()


def get_geobench_classifier(
    model: Module,
    method: str,
    is_multi_label: bool,
    num_classes: int,
    batch_size_per_device: int,
):
    if method == "linear":
        # if dataset is multi-label, we need a different classifier class
        clf_class = LinearMultiLabelClassifier if is_multi_label else LinearClassifier
    else:
        # if dataset is multi-label, we need a different classifier class
        clf_class = (
            FinetuneMultiLabelClassifier if is_multi_label else FinetuneEvalClassifier
        )

    classifier = clf_class(
        model=model,
        batch_size_per_device=batch_size_per_device,
        feature_dim=model.last_backbone_channel,
        num_classes=num_classes,
        freeze_model=method == "linear",
    )
    return classifier
