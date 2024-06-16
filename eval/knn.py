from pathlib import Path

import torch
import wandb
from lightly.utils.benchmarking import KNNClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data.mmearth_dataset import (
    create_MMEearth_args,
    MultimodalDataset,
)


def knn_eval(
    model: LightningModule,
    input_modality: dict,
    target_modality: [dict],
    data_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    num_classes: int,
    debug:bool=False
) -> None:
    """Runs KNN evaluation on the given model.

    Parameters follow InstDisc [0] settings.

    The most important settings are:
        - Num nearest neighbors: 200
        - Temperature: 0.1

    References:
       - [0]: InstDict, 2018, https://arxiv.org/abs/1805.01978
    """
    assert (
        target_modality is not None
    ), "target modality needs to be set for offline evaluation"
    print_rank_zero("Running KNN evaluation...")

    # Setup training data.
    args = create_MMEearth_args(data_dir, input_modality, target_modality)

    train_dataset = MultimodalDataset(args, split="train", transform=None, return_tuple=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    # Setup validation data.
    val_dataset = MultimodalDataset(args, split="val", transform=None, return_tuple=True)
    val_dataloader = None
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_per_device,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
    else:
        print_rank_zero("No validation data found, skipping it...")

    classifier = KNNClassifier(
        model=model,
        num_classes=num_classes,
        knn_k=1 if debug else min(len(train_dataset), 200),
        feature_dtype=torch.float16,
    )

    # Run KNN evaluation.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=devices,
        logger=WandbLogger(
            save_dir=log_dir / "knn_eval",
            name="knn_eval",
            project="ssl4eo",
            # log model config
            config=model.hparams,
            offline=debug,
        ),
        callbacks=[
            metric_callback,
        ],
        # strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        fast_dev_run=debug
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    if val_dataloader is None:
        for metric in ["train_top1", "train_top5"]:
            print_rank_zero(
                f"max finetune {metric}: {max(metric_callback.train_metrics[metric])}"
            )
    else:
        for metric in ["val_top1", "val_top5"]:
            print_rank_zero(
                f"max finetune {metric}: {max(metric_callback.val_metrics[metric])}"
            )
    wandb.finish()
