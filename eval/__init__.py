from eval.finetune import finetune_eval
from eval.geobench.geobench_clf import geobench_clf_eval
from eval.knn import knn_eval
from eval.linear import linear_eval


__all__ = ["knn_eval", "linear_eval", "finetune_eval", "geobench_clf_eval"]
