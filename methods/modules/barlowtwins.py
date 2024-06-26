from typing import Dict

import torch
from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import Tensor
from torch.nn import Module

from methods.modules.base import EOModule


class BarlowTwins(EOModule):
    default_backbone = "resnet50"

    def __init__(
        self,
        backbone: str,
        batch_size_per_device: int,
        in_channels: int,
        num_classes: int,
        has_online_classifier: bool,
        train_transform: Module,
        last_backbone_channel: int = None,
    ) -> None:
        self.save_hyperparameters(ignore=["train_transform"])
        self.hparams["method"] = self.__class__.__name__
        super().__init__(
            backbone,
            batch_size_per_device,
            in_channels,
            num_classes,
            has_online_classifier,
            train_transform,
            last_backbone_channel,
        )

        self.projection_head = BarlowTwinsProjectionHead(self.last_backbone_channel)
        self.criterion = BarlowTwinsLoss(lambda_param=5e-3, gather_distributed=True)

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Forward pass and loss calculation.
        images = batch[0]
        # Create views
        with torch.no_grad():
            views = self.train_transform(images)
        features = self.forward(torch.cat(views)).flatten(start_dim=1)
        z = self.projection_head(features)
        z0, z1 = z.chunk(len(views))
        loss = self.criterion(z0, z1)

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(views[0])
        )

        # Online linear evaluation.
        if self.has_online_classifier:
            targets = batch[1]
            cls_loss, cls_log = self.online_classifier.training_step(
                (features.detach(), targets.repeat(len(views))), batch_idx
            )
            self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
            loss = loss + cls_loss
        return loss

    def configure_optimizers(self):
        lr_factor = self.batch_size_per_device * self.trainer.world_size / 256

        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        param_list = [
            {"name": "barlowtwins", "params": params},
            {
                "name": "barlowtwins_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
                "lr": 0.0048 * lr_factor,
            },
        ]
        if self.has_online_classifier:
            param_list.append(
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                }
            )
        optimizer = LARS(
            param_list,
            lr=0.2 * lr_factor,
            momentum=0.9,
            weight_decay=1.5e-6,
        )

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
