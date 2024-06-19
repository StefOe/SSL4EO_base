from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import Sequence, Union

import torch
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from data.constants import (
    INP_MODALITIES,
    RGB_MODALITIES,
    MODALITIES_FULL,
    CLASSIFICATION_CLASSES,
)
from data.mmearth_dataset import MultimodalDataset, create_MMEearth_args
from eval.finetune import finetune_eval
from eval.geobench_clf import geobench_clf
from eval.knn import knn_eval
from eval.linear import linear_eval
from methods.barlowtwins.module import BarlowTwins
from methods.barlowtwins.transform import (
    BarlowTwinsView2Transform,
    BarlowTwinsView1Transform,
    BarlowTwinsTransform,
)
from methods.byol.module import BYOL
from methods.byol.transform import BYOLTransform, BYOLView1Transform, BYOLView2Transform
from methods.mae.module import MAE
from methods.mae.transform import MAETransform
from methods.simclr.module import SimCLR
from methods.simclr.transform import SimCLRTransform
from methods.vicreg.module import VICReg
from methods.vicreg.transform import VICRegTransform

parser = ArgumentParser("MMEarth Benchmark")
parser.add_argument("--data-dir", type=Path, default="./datasets/data_1k")
parser.add_argument("--log-dir", type=Path, default="experiment_logs")
parser.add_argument("--batch-size-per-device", type=int, default=128)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Defines number of epochs for pretraining. 0 epochs skips directly to eval steps "
    "(good for loading and testing models)",
)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--precision", type=str, default="16-mixed")
parser.add_argument("--ckpt-path", type=Path, default=None)
parser.add_argument("--compile-model", action="store_true")
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--backbone", type=str, default="default")
parser.add_argument(
    "--input-channel",
    "-i",
    type=str,
    default="all",
    help="e.g. 'all', 'rgb' (default: 'all')",
)
parser.add_argument(
    "--target",
    "-t",
    type=str,
    default="biome",
    help="select a target modality for the online classifier "
    "(e.g., 'biome', 'eco_region')",
)
parser.add_argument("--last-backbone-channel", type=int, default=None)
parser.add_argument("--skip-knn-eval", action="store_true")
parser.add_argument("--skip-linear-eval", action="store_true")
parser.add_argument("--skip-finetune-eval", action="store_true")
parser.add_argument(
    "--geobench-datasets",
    type=str,
    nargs="+",
    help="classification: 'm-eurosat', 'm-so2sat', 'm-bigearthnet'; "
    "segmentation: 'm-cashew-plant', 'm-SA-crop-type'",
)
parser.add_argument(
    "--geobench-partitions",
    type=str,
    nargs="+",
    help="How much data to train on (default: 'default')",
)

input_size = 112
METHODS = {
    "barlowtwins": {
        "model": BarlowTwins,
        "transform": BarlowTwinsTransform(
            BarlowTwinsView1Transform(input_size=input_size),
            BarlowTwinsView2Transform(input_size=input_size),
        ),
    },
    "simclr": {"model": SimCLR, "transform": SimCLRTransform(input_size=input_size)},
    "byol": {
        "model": BYOL,
        "transform": BYOLTransform(
            BYOLView1Transform(input_size=input_size),
            BYOLView2Transform(input_size=input_size),
        ),
    },
    "vicreg": {"model": VICReg, "transform": VICRegTransform(input_size=input_size)},
    "mae": {
        "model": partial(MAE, img_size=input_size),
        "transform": MAETransform(input_size=input_size),
    },
}

IN_MODALITIES = {
    "all": INP_MODALITIES,
    "rgb": RGB_MODALITIES,
}


def main(
    data_dir: Path,
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
    input_channel: str,
    target: str,
    last_backbone_channel: int,
    skip_knn_eval: bool,
    skip_linear_eval: bool,
    skip_finetune_eval: bool,
    geobench_datasets: Union[Sequence[str], None],
    geobench_partitions: Union[Sequence[str], None],
    ckpt_path: Union[Path, None],
    debug: bool = False,
) -> None:
    assert data_dir.exists(), f"data folder does not exist: {data_dir}"

    # store the requested channel combination
    input_modality = IN_MODALITIES[input_channel]
    input_key = list(input_modality.keys())[0]  # should be sentinel2 with defaults

    # number depend on which channel combination is chosen
    in_channels = sum([len(input_modality[k]) for k in input_modality])

    # checking if target is given (if not, no onlineclassifier will be trained)
    if target is None or target.lower() == "none":
        target = None
        target_modality = None
        assert skip_knn_eval and skip_linear_eval and skip_finetune_eval, (
            "if no target is set, all offline evaluation needs to be skipped "
            "(e.g., add --skip-linear-eval --skip-finetune-eval --skip-knn-eval)"
        )

    else:
        target_modality = {target: MODALITIES_FULL[target]}
    num_classes = CLASSIFICATION_CLASSES[target]

    log_dir.mkdir(exist_ok=True)

    torch.set_float32_matmul_precision("high")

    method_names = methods or METHODS.keys()

    for method in method_names:
        method_dir = (
            log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        method_dir.mkdir(exist_ok=True, parents=True)

        model = METHODS[method]["model"](
            input_key=input_key,
            target_key=target,
            backbone=backbone,
            batch_size_per_device=batch_size_per_device,
            num_classes=num_classes,
            in_channels=in_channels,
            last_backbone_channel=last_backbone_channel,
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
                input_modality=input_modality,
                target_modality=target_modality,
                data_dir=data_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                epochs=epochs,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                ckpt_path=ckpt_path,
            )

        if not geobench_datasets:
            print_rank_zero("Skipping geobench eval.")
        else:
            geobench_partitions = geobench_partitions or ["default"]
            for dataset_name, partition in product(
                geobench_datasets, geobench_partitions
            ):
                if dataset_name in ["m-eurosat", "m-so2sat", "m-bigearthnet"]:
                    geobench_clf(
                        model=model,
                        dataset_name=dataset_name,
                        partition=partition,
                        log_dir=method_dir,
                        batch_size_per_device=batch_size_per_device,
                        num_workers=num_workers,
                        accelerator=accelerator,
                        devices=devices,
                        precision=precision,
                        debug=debug,
                    )
                else:
                    raise NotImplementedError(
                        f"Geobench dataset '{dataset_name}' is not implemented."
                    )

        if target is None:
            print_rank_zero("Skipping offline eval because no target is selected.")
            return

        if skip_knn_eval:
            print_rank_zero("Skipping KNN eval.")
        else:
            knn_eval(
                model=model,
                num_classes=num_classes,
                input_modality=input_modality,
                target_modality=target_modality,
                data_dir=data_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                debug=debug,
            )

        if skip_linear_eval:
            print_rank_zero("Skipping linear eval.")
        else:
            linear_eval(
                model=model,
                num_classes=num_classes,
                input_modality=input_modality,
                target_modality=target_modality,
                data_dir=data_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                debug=debug,
            )

        if skip_finetune_eval:
            print_rank_zero("Skipping fine-tune eval.")
        else:
            finetune_eval(
                model=model,
                num_classes=num_classes,
                input_modality=input_modality,
                target_modality=target_modality,
                data_dir=data_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                debug=debug,
            )


def pretrain(
    model: LightningModule,
    method: str,
    input_modality: dict,
    target_modality: dict,
    log_dir: Path,
    data_dir: Path,
    batch_size_per_device: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    ckpt_path: Union[Path, None],
    debug: bool = False,
) -> None:
    print_rank_zero(f"Running pretraining for {method}...")

    # Setup training data.
    args = create_MMEearth_args(data_dir, input_modality, target_modality)

    train_transform = METHODS[method]["transform"]
    train_dataset = MultimodalDataset(args, split="train", transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    # Setup validation data.
    val_dataset = MultimodalDataset(args, split="val", transform=None)
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
            # ModelCheckpoint(ckpt_path, monitor="val_online_cls_top1", filename='{epoch}-{val_online_cls_top1:.2f}),
            metric_callback,
        ],
        logger=WandbLogger(
            save_dir=str(log_dir),
            name=f"pretrain",
            project="ssl4eo",
            # log model config
            config=model.hparams,
            offline=debug,
        ),
        precision=precision,
        # strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=accelerator != "cpu",  # Sync batchnorm is not supported on CPU.
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,  # TODO
        fast_dev_run=debug,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
    if target_modality is not None:
        if val_dataloader is None:
            for metric in ["train_top1", "train_top5"]:
                print_rank_zero(
                    f"max finetune {metric}: {max(metric_callback.train_metrics[metric])}"
                )
        else:
            for metric in ["val_online_cls_top1", "val_online_cls_top5"]:
                print_rank_zero(
                    f"max {metric}: {max(metric_callback.val_metrics[metric])}"
                )
    wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
