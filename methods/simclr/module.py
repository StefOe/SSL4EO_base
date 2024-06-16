import math
from typing import Dict

import torch
from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import Tensor

from methods.base import EOModule


class SimCLR(EOModule):
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
            input_key,
            target_key,
            backbone,
            batch_size_per_device,
            in_channels,
            num_classes,
            last_backbone_channel,
        )

        self.projection_head = SimCLRProjectionHead(self.last_backbone_channel)
        self.criterion = NTXentLoss(temperature=0.1, gather_distributed=True)

    def forward(self, x: Tensor) -> Tensor:
        # x = nn.functional.interpolate(x, 224) # if fixed input size is required
        features = self.backbone(x)
        return self.global_pool(features)

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        views = batch[self.input_key]
        features = self.forward(torch.cat(views)).flatten(start_dim=1)
        z = self.projection_head(features)
        z0, z1 = z.chunk(len(views))
        loss = self.criterion(z0, z1)
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
        param_list = [
            {"name": "simclr", "params": params},
            {
                "name": "simclr_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            },
        ]
        if self.target_key is not None:
            param_list.append(
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                }
            )
        optimizer = LARS(
            param_list,
            # Square root learning rate scaling improves performance for small
            # batch sizes (<=2048) and few training epochs (<=200). Alternatively,
            # linear scaling can be used for larger batches and longer training:
            #   lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256
            # See Appendix B.1. in the SimCLR paper https://arxiv.org/abs/2002.05709
            lr=0.075 * math.sqrt(self.batch_size_per_device * self.trainer.world_size),
            momentum=0.9,
            # Note: Paper uses weight decay of 1e-6 but reference code 1e-4. See:
            # https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/README.md?plain=1#L103
            weight_decay=1e-6,
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
