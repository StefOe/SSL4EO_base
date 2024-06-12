from pathlib import Path

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.utils.benchmarking import LinearClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero
from torchvision.datasets import CIFAR10


def linear_eval(
    model: Module,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    num_classes: int,
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
    print_rank_zero("Running linear evaluation...")

    # Setup training data.
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    train_dataset = CIFAR10(
        "datasets/cifar10", download=True, transform=train_transform
    )
    # train_dataset = LightlyDataset(input_dir=str(train_dir), transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True,
    )

    # Setup validation data.
    val_transform = T.Compose(
        [
            T.ToTensor(),
            # T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    val_dataset = CIFAR10(
        "datasets/cifar10", download=True, transform=val_transform, train=False
    )
    # val_dataset = LightlyDataset(input_dir=str(val_dir), transform=val_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # Train linear classifier.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=90,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            metric_callback,
        ],
        logger=WandbLogger(
            save_dir=str(log_dir), name=f"linear_eval", project="ssl4eo",
            # log model config
            config=model.hparams
        ),
        precision=precision,
        # strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        enable_checkpointing=False,
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
    for metric in ["val_top1", "val_top5"]:
        print_rank_zero(
            f"max linear {metric}: {max(metric_callback.val_metrics[metric])}"
        )
    wandb.finish()
