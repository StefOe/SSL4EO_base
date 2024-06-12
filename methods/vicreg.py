from typing import List, Tuple

import torch
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.transforms.vicreg_transform import VICRegTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import Tensor

from methods.base import EOModule


class VICReg(EOModule):
    default_backbone = "resnet50"

    def __init__(self, backbone: str, batch_size_per_device: int, in_channels: int, num_classes: int,
                 last_backbone_channel: int = None):
        self.save_hyperparameters()
        self.hparams["method"] = self.__class__.__name__
        self.batch_size_per_device = batch_size_per_device
        super().__init__(backbone, in_channels, last_backbone_channel)

        self.projection_head = VICRegProjectionHead(self.last_backbone_channel, num_layers=2)
        self.criterion = VICRegLoss()

        self.online_classifier = OnlineLinearClassifier(self.last_backbone_channel, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x = nn.functional.interpolate(x, 224) # if fixed input size is required
        features = self.backbone(x)
        return self.global_pool(features)

    def training_step(
            self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]
        features = self.forward(torch.cat(views)).flatten(start_dim=1)
        z = self.projection_head(features)
        z_a, z_b = z.chunk(len(views))
        loss = self.criterion(z_a=z_a, z_b=z_b)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # Online linear evaluation.
        cls_loss, cls_log = self.online_classifier.training_step(
            (features.detach(), targets.repeat(len(views))), batch_idx
        )

        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
            self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        global_batch_size = self.batch_size_per_device * self.trainer.world_size
        base_lr = _get_base_learning_rate(global_batch_size=global_batch_size)
        optimizer = LARS(
            [
                {"name": "vicreg", "params": params},
                {
                    "name": "vicreg_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Linear learning rate scaling with a base learning rate of 0.2.
            # See https://arxiv.org/pdf/2105.04906.pdf for details.
            lr=base_lr * global_batch_size / 256,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                        self.trainer.estimated_stepping_batches
                        / self.trainer.max_epochs
                        * 10
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
                end_value=0.01,  # Scale base learning rate from 0.2 to 0.002.
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


# VICReg transform
transform = VICRegTransform(input_size=32)


def _get_base_learning_rate(global_batch_size: int) -> float:
    """Returns the base learning rate for training 100 epochs with a given batch size.

    This follows section C.4 in https://arxiv.org/pdf/2105.04906.pdf.

    """
    if global_batch_size == 128:
        return 0.8
    elif global_batch_size == 256:
        return 0.5
    elif global_batch_size == 512:
        return 0.4
    else:
        return 0.3
