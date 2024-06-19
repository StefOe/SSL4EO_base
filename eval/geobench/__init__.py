from modulefinder import Module
from typing import Tuple, Dict

from lightly.utils.benchmarking import LinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD


class LinearMultiLabelClassifier(LinearClassifier):
    """Linear classifier for benchmarking binary multilabel classification."""

    def __init__(
        self,
        model: Module,
        batch_size_per_device: int,
        feature_dim: int,
        num_classes: int,
        freeze_model: bool,
    ):
        super().__init__(
            model, batch_size_per_device, feature_dim, num_classes, (-1,), freeze_model
        )
        self.criterion = BCEWithLogitsLoss()

    def shared_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        predicted_labels = (predictions > 0).int()
        import ipdb; ipdb.set_trace()
        acc = None
        return loss, acc

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, acc = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_acc_cls-{k}": acc for k, acc in acc.items()}
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, acc = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"val_acc_cls-{k}": acc for k, acc in acc.items()}
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss


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
