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

import wandb
from data import get_mmearth_dataloaders
from data.constants import (
    MODALITIES_FULL,
    CLASSIFICATION_CLASSES,
    MMEARTH_DIR,
    input_size,
    IN_MODALITIES,
)
from eval import finetune_eval, geobench_clf_eval, knn_eval, linear_eval
from methods import modules
from methods import transforms

# Argparser for all your configuration needs
parser = ArgumentParser("MMEarth Benchmark")

parser.add_argument(
    "--data-dir",
    type=Path,
    default=None,
    help="Path to the raw MMEarth dataset folder (default: None). "
    "If not given the environment variable MMEARTH_DIR will be used",
)
parser.add_argument(
    "--processed-dir",
    type=Path,
    default=None,
    help="Path to the processed MMEarth dataset folder (default: None). "
    "If not given the data_dir will be used",
)
parser.add_argument(
    "--log-dir",
    type=Path,
    default="experiment_logs",
    help="Path to the directory where logs will be stored (default: 'experiment_logs').",
)
parser.add_argument(
    "--batch-size-per-device",
    type=int,
    default=128,
    help="Batch size per device (default: 128).",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs for pretraining. Set to 0 to skip pretraining and go straight to evaluation (default: 100).",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Number of threads to use for data loading (default: 8).",
)
parser.add_argument(
    "--accelerator",
    type=str,
    default="gpu",
    help="Type of accelerator to use: 'cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', or 'auto' (default: 'gpu').",
)
parser.add_argument(
    "--devices", type=int, default=1, help="Number of devices to use (default: 1)."
)
parser.add_argument(
    "--precision",
    type=str,
    default="16-mixed",
    help="Model precision: '16-mixed', '32', etc. (default: '16-mixed').",
)
parser.add_argument(
    "--ckpt-path",
    type=Path,
    default=None,
    help="Path to a checkpoint file to resume training or evaluate (default: None).",
)
parser.add_argument(
    "--compile-model",
    action="store_true",
    help="If set, the model will be compiled for optimization.",
)
parser.add_argument(
    "--methods",
    type=str,
    nargs="+",
    help="SSL methods to apply: 'byol', 'simclr', 'mae', 'barlowtwins', 'vicreg'.",
)
parser.add_argument(
    "--backbone",
    type=str,
    default="default",
    help="Encoder architecture to use (default: 'default').",
)
parser.add_argument(
    "--input-channel",
    "-i",
    type=str,
    default="all",
    help="Sentinel-2 input channel selection: 'all', 'rgb' (default: 'all').",
)
parser.add_argument(
    "--target",
    "-t",
    type=str,
    default="biome",
    help="Target modality for the online classifier: 'biome', 'eco_region' (default: 'biome').",
)
parser.add_argument(
    "--last-backbone-channel",
    type=int,
    default=None,
    help="If provided, adds another backbone layer to change output size (default: None).",
)
parser.add_argument(
    "--enable-knn-eval",
    action="store_true",
    help="If set, offline KNN evaluation will be enabled.",
)
parser.add_argument(
    "--enable-linear-eval",
    action="store_true",
    help="If set, offline linear evaluation will be enabled.",
)
parser.add_argument(
    "--enable-finetune-eval",
    action="store_true",
    help="If set, offline fine-tuning evaluation will be enabled.",
)
parser.add_argument(
    "--no-ffcv",
    action="store_true",
    help="If set, pretraining will be done with regular pytorch DataLoader instead of ffcv.Loader (should be slower).",
)
parser.add_argument(
    "--geobench-datasets",
    type=str,
    nargs="+",
    help="GeoBench datasets for classification: 'm-eurosat', 'm-so2sat', 'm-bigearthnet'; "
    "for segmentation: 'm-cashew-plant', 'm-SA-crop-type'.",
)
parser.add_argument(
    "--geobench-partitions",
    type=str,
    nargs="+",
    help="Amount of GeoBench data to train on. "
    'Available: "default", "0.01x_train", "0.02x_train", "0.05x_train", "0.10x_train", '
    '"0.20x_train", "0.50x_train", "1.00x_train" (default: "default").',
)
parser.add_argument(
    "--geobench-eval-method",
    type=str,
    default="linear",
    help="How to evaluate on GeoBench, either 'linear' or 'finetune' or 'both'  (default: 'linear').",
)
parser.add_argument(
    "--geobench-processed-dir",
    type=Path,
    default=None,
    help="Path to the processed geobench dataset folder (default: None). "
    "If not given the processed_dir will be used",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, run in debug mode (for code checking).",
)

METHODS = {
    "barlowtwins": {
        "model": modules.BarlowTwins,
        "transform": transforms.BarlowTwinsTransform(
            transforms.BarlowTwinsView1Transform(input_size=input_size),
            transforms.BarlowTwinsView2Transform(input_size=input_size),
        ),
    },
    "simclr": {
        "model": modules.SimCLR,
        "transform": transforms.SimCLRTransform(input_size=input_size),
    },
    "byol": {
        "model": modules.BYOL,
        "transform": transforms.BYOLTransform(
            transforms.BYOLView1Transform(input_size=input_size),
            transforms.BYOLView2Transform(input_size=input_size),
        ),
    },
    "vicreg": {
        "model": modules.VICReg,
        "transform": transforms.VICRegTransform(input_size=input_size),
    },
    "mae": {
        "model": partial(modules.MAE, img_size=input_size),
        "transform": transforms.MAETransform(input_size=input_size),
    },
}


def main(
    data_dir: Path,
    processed_dir: Path,
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
    enable_knn_eval: bool,
    enable_linear_eval: bool,
    enable_finetune_eval: bool,
    geobench_datasets: Union[Sequence[str], None],
    geobench_partitions: Union[Sequence[str], None],
    geobench_processed_dir: Path,
    geobench_eval_method: str,
    ckpt_path: Union[Path, None],
    no_ffcv: bool,
    debug: bool = False,
) -> LightningModule:
    if data_dir is None:
        data_dir = MMEARTH_DIR  # Use default directory if data_dir is not specified

    # Ensure the data directory exists
    assert data_dir.exists(), (
        f"data folder does not exist: {data_dir}, "
        f"either --data-dir <folder> or set environment variable: export MMEARTH_DIR=<folder>"
    )

    # Retrieve input modality configuration
    input_modality = IN_MODALITIES[input_channel]

    # Calculate the number of input channels based on the chosen modality
    in_channels = sum([len(input_modality[k]) for k in input_modality])

    # Check if target is specified; if not, skip online classifier training
    if target is None or target.lower() == "none":
        target = None
        target_modality = None
        print_rank_zero("if no target is set, all offline evaluation will be skipped ")

    else:
        target_modality = {target: MODALITIES_FULL[target]}
    num_classes = CLASSIFICATION_CLASSES[target]

    # Create log directory if it doesn't exist
    log_dir.mkdir(exist_ok=True)

    # Set high precision for matrix multiplication in PyTorch
    torch.set_float32_matmul_precision("high")

    # Use all methods if none are specified
    method_names = methods or METHODS.keys()

    for method in method_names:
        # Create method-specific log directory
        method_dir = (
            log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        method_dir.mkdir(exist_ok=True, parents=True)

        # Initialize model with method-specific parameters
        model = METHODS[method]["model"](
            backbone=backbone,
            batch_size_per_device=batch_size_per_device,
            num_classes=num_classes,
            in_channels=in_channels,
            has_online_classifier=target is not None,
            train_transform=METHODS[method]["transform"],
            last_backbone_channel=last_backbone_channel,
        )

        # Compile the model if PyTorch supports it
        if compile_model and hasattr(torch, "compile"):
            print_rank_zero("Compiling model...")
            model = torch.compile(model)

        # Default configuration for training and evaluation
        default_config = {
            "model": model,
            "input_modality": input_modality,
            "target_modality": target_modality,
            "data_dir": data_dir,
            "processed_dir": processed_dir,
            "log_dir": method_dir,
            "batch_size_per_device": batch_size_per_device,
            "num_workers": num_workers,
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
            "no_ffcv": no_ffcv,
            "debug": debug,
        }

        # Skip pretraining if epochs is <= 0
        if epochs <= 0:
            print_rank_zero("Epochs <= 0, skipping pretraining.")
            if ckpt_path is not None:
                print_rank_zero(f"Loading model weights from {ckpt_path}")
                model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        else:
            # Pretraining configuration and execution
            pretrain_config = default_config.copy()
            pretrain_config["epochs"] = epochs
            pretrain_config["ckpt_path"] = ckpt_path

            print_rank_zero(f"Running pretraining for {method}...")
            pretrain(**pretrain_config)

        # Skip geobench evaluation if no datasets are specified
        if not geobench_datasets:
            print_rank_zero("Skipping geobench eval.")
        else:
            # Evaluate on specified geobench datasets, partitions, and evaluation method
            geobench_partitions = geobench_partitions or ["default"]
            if geobench_eval_method == "both":
                geobench_eval_methods = ["linear", "finetune"]
            else:
                geobench_eval_methods = [
                    geobench_eval_method
                ]  # expected to be "linear" or "finetune"

            # use processed dir if geobench_processed_dir is not defined, else use data_dir
            geobench_processed_dir = processed_dir if geobench_processed_dir is None else geobench_processed_dir
            geobench_processed_dir = data_dir if geobench_processed_dir is None else geobench_processed_dir

            for dataset_name, partition, eval_method in product(
                geobench_datasets, geobench_partitions, geobench_eval_methods
            ):
                if dataset_name in ["m-eurosat", "m-so2sat", "m-bigearthnet"]:
                    geobench_clf_eval(
                        model=model,
                        method=eval_method,
                        dataset_name=dataset_name,
                        partition=partition,
                        log_dir=method_dir,
                        processed_dir=geobench_processed_dir,
                        batch_size_per_device=batch_size_per_device,
                        num_workers=num_workers,
                        accelerator=accelerator,
                        devices=devices,
                        precision=precision,
                        no_ffcv=no_ffcv,
                        debug=debug,
                    )
                else:
                    raise NotImplementedError(
                        f"Geobench dataset '{dataset_name}' is not implemented."
                    )

        # Skip offline evaluation if no target is specified
        if target is None:
            print_rank_zero(f"Skipping offline eval because no target is selected.")
            return model
        else:
            print_rank_zero(f"Starting offline eval of '{target}' target.")

        eval_config = default_config.copy()
        eval_config["num_classes"] = num_classes

        # Perform linear evaluation if enabled
        if enable_linear_eval and target is not None:
            linear_eval(**eval_config)
        else:
            print_rank_zero("Skipping linear eval.")

        # Perform fine-tuning evaluation if enabled
        if enable_finetune_eval and target is not None:
            finetune_eval(**eval_config)
        else:
            print_rank_zero("Skipping fine-tune eval.")

        # Perform KNN evaluation if enabled
        if enable_knn_eval and target is not None:
            del eval_config["precision"]
            knn_eval(**eval_config)
        else:
            print_rank_zero("Skipping KNN eval.")

        return model


def pretrain(
    model: LightningModule,
    input_modality: dict,
    target_modality: dict,
    log_dir: Path,
    data_dir: Path,
    processed_dir: Path,
    batch_size_per_device: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    ckpt_path: Union[Path, None],
    no_ffcv: bool,
    debug: bool = False,
) -> None:
    # Setup training data.
    train_dataloader, val_dataloader = get_mmearth_dataloaders(
        data_dir,
        processed_dir,
        input_modality,
        target_modality,
        num_workers,
        batch_size_per_device,
        ["train", "val"],
        no_ffcv,
    )

    # Train model.
    metric_callback = MetricCallback()
    wandb_config = model.hparams.copy()
    wandb_config["log_dir"] = str(log_dir)
    wandb_config["ckpt_path"] = ckpt_path
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            # Stop if training loss diverges.
            EarlyStopping(monitor="train_loss", patience=int(1e12), check_finite=True),
            # ModelCheckpoint(monitor="val_top1", mode="max", auto_insert_metric_name=True),
            metric_callback,
        ],
        logger=WandbLogger(
            save_dir=str(log_dir),
            name=f"pretrain",
            project="ssl4eo",
            # log model config
            config=wandb_config,
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
    if target_modality is not None and not debug:
        if val_dataloader is None:
            for metric in ["train_online_cls_top1", "train_online_cls_top5"]:
                print_rank_zero(
                    f"max {metric}: {max(metric_callback.train_metrics[metric])}"
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
