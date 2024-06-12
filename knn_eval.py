from pathlib import Path

import torch
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.utils.benchmarking import KNNClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero
from torchvision.datasets import CIFAR10


def knn_eval(
    model: LightningModule,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    num_classes: int,
) -> None:
    """Runs KNN evaluation on the given model.

    Parameters follow InstDisc [0] settings.

    The most important settings are:
        - Num nearest neighbors: 200
        - Temperature: 0.1

    References:
       - [0]: InstDict, 2018, https://arxiv.org/abs/1805.01978
    """
    print_rank_zero("Running KNN evaluation...")

    # Setup training data.
    transform = T.Compose(
        [
            T.ToTensor(),
            # T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    train_dataset = CIFAR10(
        "datasets/cifar10", download=True, transform=transform
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
    val_dataset = CIFAR10(
        "datasets/cifar10", download=True, transform=transform, train=False
    )
    # val_dataset = LightlyDataset(input_dir=str(val_dir), transform=val_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    classifier = KNNClassifier(
        model=model,
        num_classes=num_classes,
        feature_dtype=torch.float16,
    )

    # Run KNN evaluation.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=devices,
        logger=WandbLogger(
            save_dir=log_dir / "knn_eval", name="knn_eval", project="ssl4eo",
            # log model config
            config=model.hparams
        ),
        callbacks=[
            metric_callback,
        ],
        # strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    for metric in ["val_top1", "val_top5"]:
        print_rank_zero(f"knn {metric}: {max(metric_callback.val_metrics[metric])}")
    wandb.finish()
