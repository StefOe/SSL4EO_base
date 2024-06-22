from .finetune import finetune_eval
from .geobench.geobench_clf import geobench_clf_eval
from .knn import knn_eval
from .linear import linear_eval


__all__ = ["knn_eval", "linear_eval", "finetune_eval", "geobench_clf_eval"]
