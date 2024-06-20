from typing import Dict

import torch
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import Tensor

from methods.modules.base import EOModule


class VICReg(EOModule):
    default_backbone = "resnet50"

    def __init__(
        self,
        input_key: str,
        target_key: [str, None],
        backbone: str,
        batch_size_per_device: int,
        in_channels: int,
        num_classes: int,
        last_backbone_channel: int = None,
    ):
        self.save_hyperparameters()
        self.hparams["method"] = self.__class__.__name__
        super().__init__(
            input_key, target_key, backbone, batch_size_per_device, in_channels, num_classes, last_backbone_channel
        )

        self.projection_head = VICRegProjectionHead(
            self.last_backbone_channel, num_layers=2
        )
        self.criterion = VICRegLoss()

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        views = batch[self.input_key]
        features = self.forward(torch.cat(views)).flatten(start_dim=1)
        z = self.projection_head(features)
        z_a, z_b = z.chunk(len(views))
        loss = self.criterion(z_a=z_a, z_b=z_b)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(views[0])
        )

        # Online linear evaluation.
        if self.target_key is not None:
            targets = batch[self.target_key]
            cls_loss, cls_log = self.online_classifier.training_step(
                (features.detach(), targets.repeat(len(views))), batch_idx
            )

            self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
            loss = loss + cls_loss
        return loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        global_batch_size = self.batch_size_per_device * self.trainer.world_size
        base_lr = _get_base_learning_rate(global_batch_size=global_batch_size)
        param_list = [
            {"name": "vicreg", "params": params},
            {
                "name": "vicreg_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            },
        ]
        if self.target_key is not None:
            param_list.append({
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
            })
        optimizer = LARS(
            # Linear learning rate scaling with a base learning rate of 0.2.
            # See https://arxiv.org/pdf/2105.04906.pdf for details.
            param_list,
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
