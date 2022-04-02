from typing import Mapping

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def sym_acc(pred, y, sym_map):
    pred = (pred > 0.5).bool().float()
    y = (y > 0.5).bool().float()
    y_sym = y[..., torch.as_tensor(sym_map).long()]
    acc1 = pred.eq(y).float().mean(-1)
    acc2 = pred.eq(y_sym).float().mean(-1)
    acc = torch.stack([acc1, acc2], -1).max(-1).values
    return acc


def sym_iou(pred, y, sym_map, vertex_areas):
    pred = (pred > 0.5).bool().float()
    y = (y > 0.5).bool().float()
    y_sym = y[..., torch.as_tensor(sym_map).long()]
    iou = (pred * y * vertex_areas).sum(-1) / ((torch.clamp_max(pred + y, 1) * vertex_areas).sum(-1))
    iou_sym = (pred * y_sym * vertex_areas).sum(-1) / ((torch.clamp_max(pred + y_sym, 1) * vertex_areas).sum(-1))
    return torch.stack([iou, iou_sym], -1).max(-1).values


def binary_prfs_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> Mapping[str, float]:
    """
    Return the precision, recall, f-measure. Computed using scikit-learn.

    Useful for the discriminator.

    :param y_true: the ground truth
    :param y_pred: the predictions
    :return: a dictionary containing the accuracy, precision, recall and f-measure
    """
    _, y_pred = torch.max(y_pred, dim=-1)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average="binary")

    return {
        "accuracy": acc_score,
        "precision": precision,
        "recall": recall,
        "f1": fscore,
    }


# # TODO: remove
# class PredictionRank:
#     def __init__(self, dataset_index: FaissIndex):
#         self.index = dataset_index
#
#     def __call__(
#         self,
#         y_pred: torch.Tensor,
#         y_pred_keys: torch.Tensor,
#         max_rank: int = 1000,
#     ) -> torch.Tensor:
#         """
#         The index and y_pred are assumed to be in the same encoding
#         :param dataset_index: faiss index of the dataset
#         :param y_pred: tensor [batch_size, n] of the predicted y
#         :param y_pred_keys: tensor [batch_size, 1] of the ids of the predicted y
#
#         :param max_rank: maximum rank of a prediction in [0, max_rank]
#
#         :return: tensor [batch_size, 1] of ranks for each prediction
#         """
#         device = y_pred.device
#         y_pred = y_pred.detach().cpu().numpy()
#         y_pred_keys = y_pred_keys.detach().cpu().numpy()
#
#         y_true_dists, y_true_keys = self.index.raw_search(y_pred, k_most_similar=max_rank)
#         one_hot_ranking = y_true_keys == y_pred_keys.reshape((y_pred.shape[0], 1))
#         one_hot_ranking_with_max_rank = np.concatenate((one_hot_ranking, np.ones((y_pred.shape[0], 1))), axis=-1)
#         ranks = one_hot_ranking_with_max_rank.argmax(axis=-1)
#         return torch.as_tensor(ranks, device=device)
