from typing import List, Tuple, Dict

from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.dist import print_rank_zero
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from methods import get_backbone


class BackboneExpander(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_out: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone_out = backbone_out

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)[
            -1
        ]  # model returns all intermediate results, only use last one
        return self.backbone_out(features)

    def _get_name(self):
        return f"{self.backbone.__class__.__name__} and {self.__class__.__name__}"


class EOModule(LightningModule):
    def __init__(
        self,
        input_key: str,
        target_key: [str, None],backbone: str,
        batch_size_per_device: int,in_channels: int,
        num_classes: int,last_backbone_channel: int = None
    ):
        super().__init__()
        self.batch_size_per_device = batch_size_per_device

        if backbone == "default":
            backbone = self.default_backbone
            print_rank_zero(f"Using default backbone: {backbone}")

        model = get_backbone(backbone, in_channels=in_channels)

        # saving some parameters by deleting unused model parts
        if hasattr(model, "fc"):
            del model.fc
        if hasattr(model, "classifier"):
            del model.classifier

        self.backbone = model
        feat_out = model.feature_info[-1]["num_chs"]

        self.last_backbone_channel = last_backbone_channel
        if last_backbone_channel is None:
            last_backbone_channel = feat_out
            backbone_out = nn.Identity()
        else:
            # this module could also reflect the backbone structure better
            backbone_out = nn.Sequential(
                nn.Conv2d(feat_out, last_backbone_channel, 1),
                # TODO activation function and norm layer?
            )

        self.backbone = BackboneExpander(self.backbone, backbone_out)

        self.last_backbone_channel = last_backbone_channel
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.input_key = input_key
        self.target_key = target_key
        if target_key is not None:
            self.online_classifier = OnlineLinearClassifier(
                self.last_backbone_channel, num_classes=num_classes
            )

    # these are the interfaces for torch and torchlightning to fill for each method
    def forward(self, x: Tensor) -> Tensor:
        # x = nn.functional.interpolate(x, 224) # if fixed input size is required
        features = self.backbone(x)
        return self.global_pool(features)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        raise NotImplementedError

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        images = batch[self.input_key]
        features = self.forward(images).flatten(start_dim=1)
        if self.target_key is not None:
            targets = batch[self.target_key]
            cls_loss, cls_log = self.online_classifier.validation_step(
                (features.detach(), targets), batch_idx
            )
            self.log_dict(
                cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets)
            )
            return cls_loss
        else:
            return None  # Could return variance and covariance loss instead

    def configure_optimizers(self):
        raise NotImplementedError
