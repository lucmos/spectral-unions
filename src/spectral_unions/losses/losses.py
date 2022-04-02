from typing import Mapping

import torch
import torch.nn.functional as F


def mse_vae_loss(
    batch: Mapping[str, torch.Tensor],
    model_output: Mapping[str, torch.Tensor],
    variational_beta: float,
    y_true_key: str = "union_eigenvalues",
) -> torch.Tensor:
    """
    Variational loss with mse reconstruction loss

    :param batch: the current batch, with tensors already uploaded to the correct device
    :param model_output: the output of the model.
                         The predictions are in model_output['y_pred']
    :param y_true_key: which key to use to retrieve y_true from the current batch
    :param variational_beta: the importance of the kldivergence regularizer
    :return: the loss value
    """
    y_true = batch[y_true_key]
    y_pred = model_output["y_pred"]
    mu = model_output["mu"]
    logvar = model_output["logvar"]

    recon_loss = F.mse_loss(input=y_pred, target=y_true)
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence


def squared_relative_error_std(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Squared relative error loss function plus std normalization

    :param batch: the current batch, with tensors already uploaded to the correct device
    :param model_output: the output of the model.
                         The predictions are in model_output['y_pred']
    :param y_true_key: which key to use to retrieve y_true from the current batch
    :return: the loss value
    """
    # y_true = batch[y_true_key]
    # y_pred = model_output["y_pred"]
    error = ((y_pred - y_true) / y_true.clamp(min=1e-5)) ** 2
    tot_error = torch.sum(error, dim=-1)
    std_error = torch.std(error, dim=-1) * 10
    return torch.mean(tot_error + std_error)


def crossentropy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Crossentropy loss

    :param batch: the current batch, with tensors already uploaded to the correct device
    :param model_output: the output of the model.
                         The predictions are in model_output['y_pred']
    :param y_true_key: which key to use to retrieve y_true from the current batch
    :return: the loss value
    """
    return torch.nn.functional.cross_entropy(y_pred, y_true)


def squared_relative_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Squared relative error loss function

    :param batch: the current batch, with tensors already uploaded to the correct device
    :param model_output: the output of the model.
                         The predictions are in model_output['y_pred']
    :param y_true_key: which key to use to retrieve y_true from the current batch

    :return: the loss value
    """
    error = (y_pred - y_true) / y_true.clamp(min=1e-2)
    return torch.mean(torch.sum(error**2, dim=-1))


def relative_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Relative error loss function

    :param batch: the current batch, with tensors already uploaded to the correct device
    :param model_output: the output of the model.
                         The predictions are in model_output['y_pred']
    :param y_true_key: which key to use to retrieve y_true from the current batch

    :return: the loss value
    """
    abs_error = torch.abs(y_pred - y_true)
    return torch.mean(torch.sum(abs_error / y_true.clamp(min=1e-2), dim=-1))


def l1(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    scale_factor: float = 1,
) -> torch.Tensor:
    """
    L1 loss function

    :param batch: the current batch, with tensors already uploaded to the correct device
    :param model_output: the output of the model.
                         The predictions are in model_output['y_pred']
    :param scale_factor: scale factor use to to scale y_true and y_pred before the loss
    :param y_true_key: which key to use to retrieve y_true from the current batch

    :return: the loss value
    """
    y_true, y_pred = y_true * scale_factor, y_pred * scale_factor
    return torch.nn.functional.l1_loss(y_pred, y_true)


def bce(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:

    y_true, y_pred = y_true, y_pred
    return torch.nn.functional.binary_cross_entropy(y_pred, y_true, **kwargs)


def mse(y_pred: torch.Tensor, y_true: torch.Tensor, scale_factor: float = 1, **kwargs) -> torch.Tensor:
    """
    Mean squared error loss function

    :param batch: the current batch, with tensors already uploaded to the correct device
    :param model_output: the output of the model.
                         The predictions are in model_output['y_pred']
    :param scale_factor: scale factor use to to scale y_true and y_pred before the loss
    :param y_true_key: which key to use to retrieve y_true from the current batch

    :return: the loss value
    """
    y_true, y_pred = y_true * scale_factor, y_pred * scale_factor
    return torch.nn.functional.mse_loss(y_pred, y_true, **kwargs)


# def l1_capped(
#     batch: Dict[str, torch.Tensor],
#     model_output: Mapping[str, torch.Tensor],
#     eigs_cap: float,
# ) -> torch.Tensor:
#     """
#     L1 loss function, adjusted to ignore eigenvalues in the union shape greater than
#     the ones in the partial shapes
#
#     :param batch: the current batch, with tensors already uploaded to the correct device
#     :param model_output: the output of the model.
#                          The predictions are in model_output['y_pred']
#     :param y_true_key: which key to use to retrieve y_true from the current batch
#     :param eigs_cap: ignore y_pred and y_true greater than partial eigs times eigs_cap
#
#     :return: the loss value
#     """
#     y_true = batch[y_true_key]
#     y_pred = model_output["y_pred"]
#     x1b = batch["X1_eigenvalues"]
#     x2b = batch["X2_eigenvalues"]
#
#     mask = y_true <= torch.max(x1b.max(), x2b.max()) * eigs_cap
#     return torch.nn.functional.l1_loss(y_pred * mask, y_true * mask)


# def mse_capped(
#     batch: Dict[str, torch.Tensor],
#     model_output: Mapping[str, torch.Tensor],
#     eigs_cap: float,
# ) -> torch.Tensor:
#     """
#     Mean squared error loss function, adjusted to ignore eigenvalues in the union shape
#     greater than the ones in the partial shapes
#
#     :param batch: the current batch, with tensors already uploaded to the correct device
#     :param model_output: the output of the model.
#                          The predictions are in model_output['y_pred']
#     :param y_true_key: which key to use to retrieve y_true from the current batch
#     :param eigs_cap: ignore y_pred and y_true greater than partial eigs times eigs_cap
#
#     :return: the loss value
#     """
#     y_true = batch[y_true_key]
#     y_pred = model_output["y_pred"]
#     x1b = batch["X1_eigenvalues"]
#     x2b = batch["X2_eigenvalues"]
#
#     mask = y_true <= torch.max(x1b.max(), x2b.max()) * eigs_cap
#     return torch.nn.functional.mse_loss(y_pred * mask, y_true * mask)
#
#
# def mse_capped_inorm(
#     batch: Dict[str, torch.Tensor],
#     model_output: Mapping[str, torch.Tensor],
#     eigs_cap: float,
# ) -> torch.Tensor:
#     """
#     Mean squared error loss function, adjusted to:
#     - ignore eigenvalues in the union shape greater than the ones in the partial shapes
#     - Normalize wrt the eigenvalue index
#
#     :param batch: the current batch, with tensors already uploaded to the correct device
#     :param model_output: the output of the model.
#                          The predictions are in model_output['y_pred']
#     :param y_true_key: which key to use to retrieve y_true from the current batch
#     :param eigs_cap: ignore y_pred and y_true greater than partial eigs times eigs_cap
#
#     :return: the loss value
#     """
#     y_true = batch[y_true_key]
#     y_pred = model_output["y_pred"]
#     x1b = batch["X1_eigenvalues"]
#     x2b = batch["X2_eigenvalues"]
#
#     range_indices = torch.arange(
#         1, y_true.shape[-1] + 1, dtype=y_true.dtype, device=y_true.device
#     )
#
#     mask = y_true <= torch.max(x1b.max(), x2b.max()) * eigs_cap
#     return torch.nn.functional.mse_loss(
#         (y_pred * mask) / range_indices, (y_true * mask) / range_indices
#     )
