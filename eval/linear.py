from pathlib import Path

import wandb
from lightly.utils.benchmarking import MetricCallback, LinearClassifier
from lightly.utils.dist import print_rank_zero
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Module
from torchvision import transforms as T

from data.mmearth_dataset import (
    get_mmearth_dataloaders,
)


def linear_eval(
    model: Module,
    input_modality: dict,
    target_modality: [dict],
    data_dir: Path,
    processed_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    num_classes: int,
    no_ffcv: bool,
    debug: bool = False,
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
    assert (
        target_modality is not None
    ), "target modality needs to be set for offline evaluation"
    print_rank_zero("Running linear evaluation...")

    # Setup training data.
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]
    )
    train_dataloader, val_dataloader = get_mmearth_dataloaders(
        train_transform,
        data_dir,
        processed_dir,
        input_modality,
        target_modality,
        num_workers,
        batch_size_per_device,
        ["train", "val"],
        no_ffcv,
    )

    # Train linear classifier.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=90,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            # ModelCheckpoint(ckpt_path, monitor="val_online_cls_top1", filename='{epoch}-{val_online_cls_top1:.2f}),
            metric_callback,
        ],
        logger=WandbLogger(
            save_dir=str(log_dir),
            name=f"linear_eval",
            project="ssl4eo",
            # log model config
            config=model.hparams,
            offline=debug,
        ),
        precision=precision,
        # strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        fast_dev_run=debug,
    )
    classifier = LinearClassifier(
        model=model,
        batch_size_per_device=batch_size_per_device,
        feature_dim=model.last_backbone_channel,
        num_classes=num_classes,
        freeze_model=True,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    wandb.finish()
    if debug:
        return
    if val_dataloader is None:
        for metric in ["train_top1", "train_top5"]:
            print_rank_zero(
                f"max linear {metric}: {max(metric_callback.train_metrics[metric])}"
            )
    else:
        for metric in ["val_top1", "val_top5"]:
            print_rank_zero(
                f"max linear {metric}: {max(metric_callback.val_metrics[metric])}"
            )
