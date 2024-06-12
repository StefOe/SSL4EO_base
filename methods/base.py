from typing import List, Tuple

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
        features = self.backbone(x)[-1]  # model returns all intermediate results, only use last one
        return self.backbone_out(features)


class EOModule(LightningModule):
    def __init__(self, backbone: str, in_channels: int, last_backbone_channel: int = None):
        super().__init__()

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

    # these are the interfaces for torch and torchlightning to fill for each method
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def training_step(
            self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        raise NotImplementedError

    def validation_step(
            self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
