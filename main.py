from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import torch
import torchvision
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

import finetune_eval
import knn_eval
import linear_eval
from methods import simclr, vicreg, barlowtwins, byol, mae

parser = ArgumentParser("MMEarth Benchmark")
parser.add_argument("--log-dir", type=Path, default="benchmark_logs")
parser.add_argument("--batch-size-per-device", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--precision", type=str, default="16-mixed")
parser.add_argument("--ckpt-path", type=Path, default=None)
parser.add_argument("--compile-model", action="store_true")
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--backbone", type=str, default="default")
parser.add_argument("--last-backbone-channel", type=int, default=None)
parser.add_argument("--num-classes", type=int, default=10)
parser.add_argument("--skip-knn-eval", action="store_true")
parser.add_argument("--skip-linear-eval", action="store_true")
parser.add_argument("--skip-finetune-eval", action="store_true")

METHODS = {
    "barlowtwins": {"model": barlowtwins.BarlowTwins, "transform": barlowtwins.transform},
    "simclr": {"model": simclr.SimCLR, "transform": simclr.transform},
    "byol": {"model": byol.BYOL, "transform": byol.transform},
    "vicreg": {"model": vicreg.VICReg, "transform": vicreg.transform},
    "mae": {"model": mae.MAE, "transform": mae.transform},
}


def main(
        log_dir: Path,
        batch_size_per_device: int,
        epochs: int,
        num_workers: int,
        accelerator: str,
        devices: int,
        precision: str,
        compile_model: bool,
        methods: Union[Sequence[str], None],
        backbone: str,
        last_backbone_channel: int,
        num_classes: int,
        skip_knn_eval: bool,
        skip_linear_eval: bool,
        skip_finetune_eval: bool,
        ckpt_path: Union[Path, None],
) -> None:
    torch.set_float32_matmul_precision("high")

    method_names = methods or METHODS.keys()

    # This might change for EO
    in_channels = 3

    for method in method_names:
        method_dir = (
                log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        model = METHODS[method]["model"](
            backbone, batch_size_per_device=batch_size_per_device, num_classes=num_classes, in_channels=in_channels,
            last_backbone_channel=last_backbone_channel
        )

        if compile_model and hasattr(torch, "compile"):
            # Compile model if PyTorch supports it.
            print_rank_zero("Compiling model...")
            model = torch.compile(model)

        if epochs <= 0:
            print_rank_zero("Epochs <= 0, skipping pretraining.")
            if ckpt_path is not None:
                model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        else:
            pretrain(
                model=model,
                method=method,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                epochs=epochs,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                ckpt_path=ckpt_path,
            )

        if skip_knn_eval:
            print_rank_zero("Skipping KNN eval.")
        else:
            knn_eval.knn_eval(
                model=model,
                num_classes=num_classes,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
            )

        if skip_linear_eval:
            print_rank_zero("Skipping linear eval.")
        else:
            linear_eval.linear_eval(
                model=model,
                num_classes=num_classes,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
            )

        if skip_finetune_eval:
            print_rank_zero("Skipping fine-tune eval.")
        else:
            finetune_eval.finetune_eval(
                model=model,
                num_classes=num_classes,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
            )


def pretrain(
        model: LightningModule,
        method: str,
        log_dir: Path,
        batch_size_per_device: int,
        epochs: int,
        num_workers: int,
        accelerator: str,
        devices: int,
        precision: str,
        ckpt_path: Union[Path, None],
) -> None:
    print_rank_zero(f"Running pretraining for {method}...")

    # Setup training data.
    train_transform = METHODS[method]["transform"]
    # train_dataset = LightlyDataset(input_dir=str(train_dir), transform=train_transform)
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
    val_transform = T.Compose([
        T.ToTensor(),
        # T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ])
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

    # Train model.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            # Stop if training loss diverges.
            EarlyStopping(monitor="train_loss", patience=int(1e12), check_finite=True),
            # DeviceStatsMonitor(),
            metric_callback,
        ],
        logger=WandbLogger(
            save_dir=str(log_dir), name=f"pretrain", project="ssl4eo",
            # log model config
            config=model.hparams
        ),
        # logger=TensorBoardLogger(save_dir=str(log_dir), name="pretrain"),
        precision=precision,
        # strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=accelerator != "cpu",  # Sync batchnorm is not supported on CPU.
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1, #TODO
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
    for metric in ["val_online_cls_top1", "val_online_cls_top5"]:
        print_rank_zero(f"max {metric}: {max(metric_callback.val_metrics[metric])}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
