from typing import Dict

import torch
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.dist import print_rank_zero
from lightly.utils.scheduler import CosineWarmupScheduler
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import MSELoss, Parameter
from torch.optim import AdamW
from torch.nn import Module

from methods.modules.base import get_backbone


class MAE(LightningModule):
    # TODO maybe vit_base_patch8_112 is more appropriate since we halve the input size
    default_backbone = "vit_base_patch16_224"

    def __init__(
        self,
        backbone: str,
        batch_size_per_device: int,
        in_channels: int,
        img_size: int,
        num_classes: int,
        has_online_classifier: bool,
        train_transform: Module,
        last_backbone_channel: int = None,
    ):
        assert (
            "vit" in backbone or backbone == "default"
        ), f"only vit backbone supported (given: {backbone})"
        assert (
            last_backbone_channel is None
        ), f"change of last backbone channel is not supported (given: {last_backbone_channel})"

        super().__init__()
        self.save_hyperparameters(ignore=["train_transform"])
        self.hparams["method"] = self.__class__.__name__
        self.batch_size_per_device = batch_size_per_device
        if backbone == "default":
            backbone = self.default_backbone
            print_rank_zero(f"Using default backbone: {backbone}")

        self.img_size = img_size
        vit = get_backbone(backbone, in_channels=in_channels, feautures_only=False)
        vit.default_cfg["input_size"] = (in_channels, img_size, img_size)
        # overriding the patch embedding for new channel and image size
        vit.patch_embed = vit.patch_embed.__class__(
            img_size=img_size,
            patch_size=vit.patch_embed.patch_size,
            in_chans=in_channels,
            embed_dim=vit.embed_dim,
        )
        # fixing learned position embedding
        vit.num_patches = vit.patch_embed.num_patches
        self.sequence_length = vit.patch_embed.num_patches + vit.num_prefix_tokens
        vit.pos_embed = Parameter(
            torch.randn(1, self.sequence_length, vit.embed_dim) * 0.02
        )

        self.last_backbone_channel = vit.embed_dim

        decoder_embed_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        num_patches = vit.patch_embed.num_patches
        mask_token = Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(mask_token, std=0.02)
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.decoder = MAEDecoderTIMM(
            in_chans=in_channels,
            num_patches=num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            mask_token=mask_token,
        )
        self.criterion = MSELoss()

        self.has_online_classifier = has_online_classifier
        if has_online_classifier:
            self.online_classifier = OnlineLinearClassifier(
                feature_dim=vit.embed_dim, num_classes=num_classes
            )

        self.train_transform = train_transform

    def forward(self, x: Tensor) -> Tensor:
        # ensuring that the img size requirements are met (this should only be triggered for offline eval)
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = torch.nn.functional.interpolate(x, self.img_size)
        return self.backbone(images=x)

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        images = batch[0]
        # Create views
        with torch.no_grad():
            images = self.train_transform(images)[0] # only expecting single view

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        features = self.forward_encoder(images, idx_keep)
        predictions = self.forward_decoder(features, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(predictions, target)
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(images)
        )

        # Online linear evaluation.
        if self.has_online_classifier:
            targets = batch[1]
            cls_features = features[:, 0]
            cls_loss, cls_log = self.online_classifier.training_step(
                (cls_features.detach(), targets), batch_idx
            )
            self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
            loss = loss + cls_loss
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        images = batch[0]
        if self.has_online_classifier:
            # ensuring that the img size requirements are met
            if images.shape[2] != self.img_size or images.shape[3] != self.img_size:
                images = torch.nn.functional.interpolate(images, self.img_size)

            targets = batch[1]
            cls_features = self.forward(images).flatten(start_dim=1)
            cls_loss, cls_log = self.online_classifier.validation_step(
                (cls_features.detach(), targets), batch_idx
            )
            self.log_dict(
                cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets)
            )
            return cls_loss
        else:
            return None  # Could return variance and covariance loss instead

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = utils.get_weight_decay_parameters(
            [self.backbone, self.decoder]
        )
        param_list = [
            {"name": "mae", "params": params},
            {
                "name": "mae_no_weight_decay",
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
        optimizer = AdamW(
            param_list,
            lr=1.5e-4 * self.batch_size_per_device * self.trainer.world_size / 256,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 40
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
