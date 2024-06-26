import copy
from typing import Tuple, Dict

import torch
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import get_weight_decay_parameters, update_momentum
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from torch import Tensor
from torch.nn import Module

from methods.modules.base import EOModule


class BYOL(EOModule):
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

        self.projection_head = BYOLProjectionHead(self.last_backbone_channel)
        self.prediction_head = BYOLPredictionHead()
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_projection_head = BYOLProjectionHead(self.last_backbone_channel)
        self.criterion = NegativeCosineSimilarity()

    def forward_student(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self(x).flatten(start_dim=1)
        projections = self.projection_head(features)
        predictions = self.prediction_head(projections)
        return features, predictions

    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tensor:
        features = self.global_pool(self.teacher_backbone(x)).flatten(start_dim=1)
        projections = self.teacher_projection_head(features)
        return projections

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Momentum update teacher.
        # Settings follow original code for 100 epochs which are slightly different
        # from the paper, see:
        # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.99,
            end_value=1.0,
        )
        update_momentum(self.backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)

        # Forward pass and loss calculation.
        images = batch[0]
        # Create views
        with torch.no_grad():
            views = self.train_transform(images)
        teacher_projections_0 = self.forward_teacher(views[0])
        teacher_projections_1 = self.forward_teacher(views[1])
        student_features_0, student_predictions_0 = self.forward_student(views[0])
        _, student_predictions_1 = self.forward_student(views[1])
        # NOTE: Factor 2 because: L2(norm(x), norm(y)) = 2 - 2 * cossim(x, y)
        loss_0 = 2 * self.criterion(teacher_projections_0, student_predictions_1)
        loss_1 = 2 * self.criterion(teacher_projections_1, student_predictions_0)
        # NOTE: No mean because original code only takes mean over batch dimension, not
        # views.
        loss = loss_0 + loss_1
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(views[0])
        )

        # Online linear evaluation.
        if self.has_online_classifier:
            targets = batch[1]
            cls_loss, cls_log = self.online_classifier.training_step(
                (student_features_0.detach(), targets), batch_idx
            )
            self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
            loss = loss + cls_loss
        return loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [
                self.backbone,
                self.projection_head,
                self.prediction_head,
            ]
        )
        param_list = [
            {"name": "byol", "params": params},
            {
                "name": "byol_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
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
            # Settings follow original code for 100 epochs which are slightly different
            # from the paper, see:
            # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
            lr=0.45 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
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
