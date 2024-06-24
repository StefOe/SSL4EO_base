from typing import Tuple, Dict

import torch
from lightly.utils.benchmarking import LinearClassifier as LightningLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torchmetrics.functional import accuracy, f1_score, average_precision


class LinearClassifier(LightningLinearClassifier):
    def __init__(
        self,
        model: nn.Module,
        batch_size_per_device: int,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,
        train_transform: nn.Module = None,
    ):
        super().__init__(model, batch_size_per_device, feature_dim, num_classes, topk, freeze_model)
        self.train_transform = nn.Sequential() if train_transform is None else train_transform

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        with torch.no_grad():
            images = self.train_transform(batch[0])
        return super().training_step((images, *batch[1:]), batch_idx)

    # adding missing test_step function
    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"test_top{k}": acc for k, acc in topk.items()}
        self.log(
            "test_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss


class LinearMultiLabelClassifier(LinearClassifier):
    """Linear classifier for benchmarking binary multilabel classification."""

    def __init__(
        self,
        model: nn.Module,
        batch_size_per_device: int,
        feature_dim: int,
        num_classes: int,
        freeze_model: bool,
        train_transform: nn.Module = None,
    ):
        super().__init__(
            model, batch_size_per_device, feature_dim, num_classes, (-1,), freeze_model, train_transform
        )
        self.num_classes = num_classes
        self.criterion = BCEWithLogitsLoss()

    def shared_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> Tuple[Tensor, Dict[str, float]]:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets.to(predictions.dtype))

        metrics = {}
        acc_glob = accuracy(
            predictions,
            targets,
            task="multilabel",
            num_labels=self.num_classes,
            average="macro",
        )
        # acc_cls = accuracy(
        #     predictions,
        #     targets,
        #     task="multilabel",
        #     num_labels=self.num_classes,
        #     average="none",
        # )
        # metrics = {f"top1 cls-{i}": value.item() for i, value in enumerate(acc_cls)}
        metrics["top1"] = acc_glob.item()

        f1_glob = f1_score(
            predictions,
            targets,
            task="multilabel",
            num_labels=self.num_classes,
            average="macro",
        )
        # f1_cls = f1_score(
        #     predictions,
        #     targets,
        #     task="multilabel",
        #     num_labels=self.num_classes,
        #     average="none",
        # )
        # metrics.update({f"f1 cls-{i}": value.item() for i, value in enumerate(f1_cls)})
        metrics["f1"] = f1_glob.item()

        mAP_glob = average_precision(
            predictions,
            targets,
            task="multilabel",
            num_labels=self.num_classes,
            average="macro",
        )
        # mAP_cls = average_precision(
        #     predictions,
        #     targets,
        #     task="multilabel",
        #     num_labels=self.num_classes,
        #     average="none",
        # )
        # metrics.update({f"mAP cls-{i}": value.item() for i, value in enumerate(mAP_glob)})
        metrics["mAP"] = mAP_glob.item()

        return loss, metrics

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        with torch.no_grad():
            images = self.train_transform(batch[0])
        loss, metrics = self.shared_step(
            batch=(images, *batch[1:]), batch_idx=batch_idx
        )
        batch_size = len(batch[1])
        log_dict = {f"train_{metric}": value for metric, value in metrics.items()}
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, metrics = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"val_{metric}": value for metric, value in metrics.items()}
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, metrics = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"test_{metric}": value for metric, value in metrics.items()}
        self.log(
            "test_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss


class FinetuneEvalClassifier(LinearClassifier):

    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        parameters += self.model.parameters()
        optimizer = SGD(
            parameters,
            lr=0.05 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


class FinetuneMultiLabelClassifier(LinearMultiLabelClassifier):
    """Finetune classifier for benchmarking binary multilabel classification."""

    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        parameters += self.model.parameters()
        optimizer = SGD(
            parameters,
            lr=0.05 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
