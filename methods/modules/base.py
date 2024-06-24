from typing import List, Tuple, Dict

from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.dist import print_rank_zero
from pytorch_lightning import LightningModule
from timm import create_model
from torch import Tensor, nn


class BackboneExpander(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_out: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone_out = backbone_out

    def forward(self, x: Tensor) -> Tensor:
        # model returns all intermediate results, only use last one
        features = self.backbone(x)[-1]
        return self.backbone_out(features)

    def _get_name(self):
        return f"{self.backbone.__class__.__name__} and {self.__class__.__name__}"


class EOModule(LightningModule):
    def __init__(
        self,
        backbone: str,
        batch_size_per_device: int,
        in_channels: int,
        num_classes: int,
        has_online_classifier: bool,
        train_transform: nn.Module,
        last_backbone_channel: int = None,
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

        self.has_online_classifier = has_online_classifier
        if has_online_classifier is not None:
            self.online_classifier = OnlineLinearClassifier(
                self.last_backbone_channel, num_classes=num_classes
            )

        self.train_transform = train_transform

    # these are the interfaces for torch and torchlightning to fill for each method
    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.global_pool(features)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        raise NotImplementedError


    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        images = batch[0]
        features = self.forward(images).flatten(start_dim=1)
        if self.has_online_classifier:
            targets = batch[1]
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


def get_backbone(name: str, in_channels: int, feautures_only: bool = True):
    try:
        model = create_model(name, pretrained=False, features_only=feautures_only)
    except RuntimeError:
        print_rank_zero(
            f"Could not find '{name}' backbone or it does not support 'features_only' mode, quitting now"
        )
        quit()

    change_input_dims(model, in_channels)
    return model


def change_input_dims(model, in_channels):
    default_in_channels = 3

    # find modules with default inputs:
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            module.weight = nn.parameter.Parameter(
                Tensor(
                    module.out_channels,
                    in_channels // module.groups,
                    *module.kernel_size,
                )
            )
            module.reset_parameters()
        elif (
            isinstance(module, nn.Linear) and module.in_features == default_in_channels
        ):
            module.weight = nn.parameter.Parameter(
                Tensor(
                    module.out_features,
                    in_channels,
                )
            )
            module.reset_parameters()

    # only changing the obvious setting (there are more like "test_input_size" that are not always present)
    model.default_cfg["input_size"] = (
        in_channels,
        *model.default_cfg["input_size"][1:],
    )
    return model
