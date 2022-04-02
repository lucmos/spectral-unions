from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import scipy
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from nn_core.common.utils import get_env

from spectral_unions.data.dataset_utils import CustomDataLoader, CustomDataset, load_mat
from spectral_unions.data.datastructures import Mesh
from spectral_unions.metrics.metrics import sym_acc, sym_iou
from spectral_unions.plot.plot_utils import plot_shapes_comparison

# fixme: broken
register = ...


class MaskModelParametricHorse(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        pl_encoder_class,
        encoder_params,
        encoder_class_weights,
        freeze_encoder,
        pl_decoder_class,
        decoder_params,
        decoder_class_weights,
        freeze_decoder,
    ):
        super().__init__()
        self.hparams = hparams

        self.evals_encoder = None
        self.evals_decoder = None
        self.init_evals_encoders()

        # Select loss function
        self.loss_fun_evals = register.get_loss_function(self.hparams["params"]["loss"]["evals"]["fun"])
        self.evals_alpha = float(self.hparams["params"]["loss"]["evals"]["alpha"])
        self.loss_fun_mask = register.get_loss_function(self.hparams["params"]["loss"]["mask"]["fun"])
        self.mask_alpha = float(self.hparams["params"]["loss"]["mask"]["alpha"])

        dataset_root: Path = Path(get_env(self.hparams["params"]["dataset"]["name"]))
        if "train_datasplit_folder" in self.hparams["params"]["dataset"]:
            self.train_datasplit = Path(self.hparams["params"]["dataset"]["train_datasplit_folder"]) / "train.txt"
        else:
            self.train_datasplit = Path("datasplit_singleshape") / "train.txt"

        self.template_vertices = load_mat(dataset_root / "extras", "VERT.mat")
        self.template_faces = load_mat(dataset_root / "extras", "TRIV.mat").astype("long") - 1

        self.register_buffer(
            name="sym",
            tensor=torch.from_numpy(
                scipy.io.loadmat(dataset_root / "extras" / "SMPLsym.mat")["idxs"].squeeze().astype("long") - 1
            ),
        )

        self.predict_evals = None
        encoder_params = {} if encoder_params is None else encoder_params
        pl_encoder_class: pl.LightningModule = register.get_model_class(pl_encoder_class)
        if encoder_class_weights is None:
            encoder_params = encoder_params if encoder_params else {}
            self.predict_evals = pl_encoder_class(self.hparams, **encoder_params)
        else:
            self.predict_evals = pl_encoder_class.load_from_checkpoint(
                encoder_class_weights,
                hparams=hparams,
                **encoder_params,
            )
            if freeze_encoder:
                self.predict_evals.freeze()

        self.predict_mask = None
        pl_decoder_class: pl.LightningModule = register.get_model_class(pl_decoder_class)
        if decoder_class_weights is None:
            decoder_params = decoder_params if decoder_params else {}
            self.predict_mask = pl_decoder_class(self.hparams, **decoder_params)
        else:
            self.predict_mask = pl_decoder_class.load_from_checkpoint(decoder_class_weights)
            if freeze_decoder:
                self.predict_mask.freeze()

    def _collate(self, batch):
        x_names = {
            "X1_eigenvalues",
            "X2_eigenvalues",
        }

        y_names = {
            "union_eigenvalues",
            "union_indices",
            "X1_indices",
            "X2_indices",
            "complete_shape_areas",
        }

        x = {key: torch.stack([d[key] for d in batch]) for key in x_names}
        y = {key: torch.stack([d[key] for d in batch]) for key in y_names}
        return x, y

    def init_evals_encoders(self):
        # Select evals encoding transformation
        try:
            self.evals_encoder = register.get_data_encoder_function(
                self.hparams["params"]["preprocessing"]["evals_transform"]
            )
            self.evals_decoder = register.get_data_decoder_function(
                self.hparams["params"]["preprocessing"]["evals_transform"]
            )
        except KeyError:
            print("Evals encoder not found: raw eigenvalues will be used!")
            self.evals_encoder = None
            self.evals_decoder = None

    def configure_optimizers(
        self,
    ) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        # Build optimizer
        opt_class = register.get_opt_class(self.hparams["params"]["optimizer"]["class"])
        opt_params = self.hparams["params"]["optimizer"]["params"]
        for x, y in opt_params.items():
            try:
                opt_params[x] = float(y)
            except TypeError:
                pass
        try:
            opt = opt_class(
                self.parameters(),
                **opt_params,
            )
        except ValueError:
            raise ValueError("WARNING: optimizer got an empty parameter list")

        if "scheduler" not in self.hparams["params"]:
            return opt
        else:
            # Build optimizer + scheduler
            scheduler_class = register.get_scheduler_class(self.hparams["params"]["scheduler"]["class"])
            scheduler_params = self.hparams["params"]["scheduler"]["params"]
            scheduler = scheduler_class(opt, **scheduler_params)
            return [opt], [scheduler]

    def prepare_data(self) -> None:
        dataset_name = self.hparams["params"]["dataset"]["name"]
        dataset_folder = Path(get_env(dataset_name))

        self.train_files = sorted((dataset_folder / self.train_datasplit).read_text().splitlines())

        if "train_class" in self.hparams["params"]["dataset"]["train"]:

            train_dataset_class: Callable[..., CustomDataset] = register.get_dataset_class(
                self.hparams["params"]["dataset"]["train"]["train_class"]
            )
        else:
            train_dataset_class: Callable[..., CustomDataset] = register.get_dataset_class(
                self.hparams["params"]["dataset"]["train"]["class"]
            )

        self.train_dataset: CustomDataset = train_dataset_class(
            self.hparams,
            self.hparams["params"]["dataset"]["name"],
            self.train_files,
            evals_encoder=self.evals_encoder,
            augment=True,
        )
        if "class" in self.hparams["params"]["dataset"]["validation"]:

            dataset_class: Callable[..., CustomDataset] = register.get_dataset_class(
                self.hparams["params"]["dataset"]["validation"]["class"]
            )
        else:
            dataset_class = train_dataset_class

        self.val_files = sorted(((dataset_folder / "horse5.txt").read_text().splitlines()))

        self.val_dataset: CustomDataset = dataset_class(
            self.hparams,
            self.hparams["params"]["dataset"]["name"],
            self.val_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.test_files = sorted(((dataset_folder / "horse7.txt").read_text().splitlines()))

        self.test_dataset: CustomDataset = dataset_class(
            self.hparams,
            self.hparams["params"]["dataset"]["name"],
            self.test_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.horse15_files = sorted(((dataset_folder / "horse15.txt").read_text().splitlines()))

        self.horse15_dataset: CustomDataset = dataset_class(
            self.hparams,
            self.hparams["params"]["dataset"]["name"],
            self.horse15_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        # self.val_remeshed_dataset = RemeshDataset(
        #     hparams=self.hparams,
        #     dataset_name="REMESH_DATASET_val",
        #     evals_encoder=self.evals_encoder,
        # )
        # self.test_remeshed_dataset = RemeshDataset(
        #     hparams=self.hparams,
        #     dataset_name="REMESH_DATASET_test",
        #     evals_encoder=self.evals_encoder,
        # )
        #
        # unkid_dataset_folder = Path(get_env("UNK_IDENTITY_DATASET"))
        #
        # self.unkid_val_files = sorted(
        #     ((unkid_dataset_folder / "val.txt").read_text().splitlines())
        # )
        #
        # self.unkid_val_dataset: CustomDataset = dataset_class(
        #     self.hparams,
        #     "UNK_IDENTITY_DATASET",
        #     self.unkid_val_files,
        #     evals_encoder=self.evals_encoder,
        #     augment=False,
        # )
        #
        # self.unkid_test_files = sorted(
        #     ((unkid_dataset_folder / f"test.txt").read_text().splitlines())
        # )
        #
        # self.unkid_test_dataset: CustomDataset = dataset_class(
        #     self.hparams,
        #     "UNK_IDENTITY_DATASET",
        #     self.unkid_test_files,
        #     evals_encoder=self.evals_encoder,
        #     augment=False,
        # )
        #
        # unkid_F_dataset_folder = Path(get_env("UNK_IDENTITY_F_DATASET"))
        #
        # self.unkid_F_val_files = sorted(
        #     ((unkid_F_dataset_folder / f"val.txt").read_text().splitlines())
        # )
        #
        # self.unkid_F_val_dataset: CustomDataset = dataset_class(
        #     self.hparams,
        #     "UNK_IDENTITY_F_DATASET",
        #     self.unkid_F_val_files,
        #     evals_encoder=self.evals_encoder,
        #     augment=False,
        # )
        #
        # self.unkid_F_test_files = sorted(
        #     ((unkid_F_dataset_folder / f"test.txt").read_text().splitlines())
        # )
        #
        # self.unkid_F_test_dataset: CustomDataset = dataset_class(
        #     self.hparams,
        #     "UNK_IDENTITY_F_DATASET",
        #     self.unkid_F_test_files,
        #     evals_encoder=self.evals_encoder,
        #     augment=False,
        # )
        #
        # self.val_unkid_remeshed_dataset = RemeshDataset(
        #     hparams=self.hparams,
        #     dataset_name="UNK_IDENTITY_REMESH_DATASET_val",
        #     evals_encoder=self.evals_encoder,
        # )
        # self.test_unkid_remeshed_dataset = RemeshDataset(
        #     hparams=self.hparams,
        #     dataset_name="UNK_IDENTITY_REMESH_DATASET_test",
        #     evals_encoder=self.evals_encoder,
        # )
        #
        # self.val_unkid_F_remeshed_dataset = RemeshDataset(
        #     hparams=self.hparams,
        #     dataset_name="UNK_IDENTITY_F_REMESH_DATASET_val",
        #     evals_encoder=self.evals_encoder,
        # )
        # self.test_unkid_F_remeshed_dataset = RemeshDataset(
        #     hparams=self.hparams,
        #     dataset_name="UNK_IDENTITY_F_REMESH_DATASET_test",
        #     evals_encoder=self.evals_encoder,
        # )

        self.val_names = [
            "horse5",
            "horse7",
            "horse15"
            # "val_remeshed",
            # "test_remeshed",
            # "val_unkid",
            # "test_unkid",
            # "val_unkid_F",
            # "test_unkid_F",
            # "val_unkid_remeshed",
            # "test_unkid_remeshed",
            # "val_unkid_F_remeshed",
            # "test_unkid_F_remeshed",
        ]

        self.val_datasets = {
            "horse5": self.val_dataset,
            "horse7": self.test_dataset,
            "horse15": self.horse15_dataset,
            # "test": self.test_dataset,
            # "val_remeshed": self.val_remeshed_dataset,
            # "test_remeshed": self.test_remeshed_dataset,
            # "val_unkid": self.unkid_val_dataset,
            # "test_unkid": self.unkid_test_dataset,
            # "val_unkid_F": self.unkid_F_val_dataset,
            # "test_unkid_F": self.unkid_F_test_dataset,
            # "val_unkid_remeshed": self.val_unkid_remeshed_dataset,
            # "test_unkid_remeshed": self.test_unkid_remeshed_dataset,
            # "val_unkid_F_remeshed": self.val_unkid_F_remeshed_dataset,
            # "test_unkid_F_remeshed": self.test_unkid_F_remeshed_dataset,
        }

        if self.logger is not None:
            for logger in self.logger:
                logger.log_data_as_yaml(data=dict(self.hparams), filename="experiment.yml")
                logger.log_file(filename=dataset_folder / self.train_datasplit)
                logger.log_file(filename=dataset_folder / "val.txt")
                logger.log_file(filename=dataset_folder / "test.txt")

    def train_dataloader(self) -> DataLoader:
        train_loader = CustomDataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams["params"]["dataset"]["train"]["loader"]["batch_size"],
            shuffle=self.hparams["params"]["dataset"]["train"]["loader"]["shuffle"],
            num_workers=self.hparams["params"]["dataset"]["train"]["loader"]["workers"],
            pin_memory=self.hparams["params"]["dataset"]["train"]["loader"]["pin_memory"],
            collate_fn=self._collate,
            persistent_workers=self.hparams["params"]["dataset"]["train"]["loader"]["persistent_workers"],
        )
        return train_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return [
            CustomDataLoader(
                dataset=self.val_datasets[val_name],
                batch_size=self.hparams["params"]["dataset"]["validation"]["loader"]["batch_size"],
                shuffle=self.hparams["params"]["dataset"]["validation"]["loader"]["shuffle"],
                num_workers=self.hparams["params"]["dataset"]["validation"]["loader"]["workers"],
                pin_memory=self.hparams["params"]["dataset"]["validation"]["loader"]["pin_memory"],
                collate_fn=self._collate,
                persistent_workers=self.hparams["params"]["dataset"]["validation"]["loader"]["persistent_workers"],
            )
            for val_name in self.val_names
        ]

    def introudce_noise(self, evals):
        return evals * (1 + torch.randn_like(evals, device=evals.device) * 1e-3)

    def forward(
        self,
        X1_eigenvalues,
        X2_eigenvalues,
    ):
        pred_union_evals = self.predict_evals(X1_eigenvalues, X2_eigenvalues)
        pred_union_mask = self.predict_mask(pred_union_evals)
        return pred_union_evals, pred_union_mask

    def compute_evals_loss(self, evals_pred, evals_true):
        if self.evals_decoder is not None and not self.hparams["params"]["loss"]["evals"]["loss_between_encodings"]:
            evals_true = self.evals_decoder(evals_true)
            evals_pred = self.evals_decoder(evals_pred)
        evals_loss = self.loss_fun_evals(
            y_pred=evals_pred,
            y_true=evals_true,
            **self.hparams["params"]["loss"]["evals"]["params"]
            if self.hparams["params"]["loss"]["evals"]["params"] is not None
            else {},
        )
        return evals_loss

    def compute_mask_loss(self, pred_union_mask, union_indices, complete_shape_areas):
        sym_union_indices = union_indices[..., self.sym]

        loss_kwargs = (
            self.hparams["params"]["loss"]["mask"]["params"]
            if self.hparams["params"]["loss"]["mask"]["params"] is not None
            else {}
        )
        mask_loss = (
            self.loss_fun_mask(pred_union_mask, union_indices, reduction="none", **loss_kwargs) * complete_shape_areas
        ).sum(dim=-1)

        # union_dx_overlap = torch.einsum("bj, j -> b", union_indices, self.dx)
        # symunion_dx_overlap = torch.einsum("bj, j -> b", sym_union_indices, self.dx)
        # idxs_union_max_overlap = torch.stack(
        #     [union_dx_overlap, symunion_dx_overlap], -1
        # ).max(-1)[1]

        sym_mask_loss = (
            self.loss_fun_mask(pred_union_mask, sym_union_indices, reduction="none", **loss_kwargs)
            * complete_shape_areas
        ).sum(dim=-1)

        return torch.stack([mask_loss, sym_mask_loss], -1).min(-1)[0].mean()

    def compute_loss(
        self,
        X1_eigenvalues,
        X2_eigenvalues,
        union_eigenvalues,
        union_indices,
        complete_shape_areas,
    ):
        out = self(
            X1_eigenvalues,
            X2_eigenvalues,
        )
        pred_union_evals, pred_union_mask = out

        evals_loss = self.compute_evals_loss(evals_pred=pred_union_evals, evals_true=union_eigenvalues)
        mask_loss = self.compute_mask_loss(
            pred_union_mask=pred_union_mask,
            union_indices=union_indices,
            complete_shape_areas=complete_shape_areas,
        )

        evals_loss = self.evals_alpha * evals_loss
        mask_loss = self.mask_alpha * mask_loss
        loss = evals_loss + mask_loss

        return loss, evals_loss, mask_loss, out

    def compute_losses(self, batch):
        x, y_trues = batch
        x1_eigenvalues = x["X1_eigenvalues"]
        x2_eigenvalues = x["X2_eigenvalues"]

        x1_indices = y_trues["X1_indices"]
        x2_indices = y_trues["X2_indices"]

        complete_shape_areas = y_trues["complete_shape_areas"]
        union_eigenvalues = y_trues["union_eigenvalues"]
        union_indices = y_trues["union_indices"]

        (loss, evals_loss, mask_loss, (pred_union_evals, pred_union_mask),) = self.compute_loss(
            X1_eigenvalues=x1_eigenvalues,
            X2_eigenvalues=x2_eigenvalues,
            union_eigenvalues=union_eigenvalues,
            union_indices=union_indices,
            complete_shape_areas=complete_shape_areas,
        )

        loss_sym, evals_loss_sym, mask_loss_sym, _ = self.compute_loss(
            X1_eigenvalues=x2_eigenvalues,
            X2_eigenvalues=x1_eigenvalues,
            union_eigenvalues=union_eigenvalues,
            union_indices=union_indices,
            complete_shape_areas=complete_shape_areas,
        )

        loss_idempx1, evals_loss_idempx1, mask_loss_idempx1, _ = self.compute_loss(
            X1_eigenvalues=x1_eigenvalues,
            X2_eigenvalues=x1_eigenvalues,
            union_eigenvalues=x1_eigenvalues,
            union_indices=x1_indices,
            complete_shape_areas=complete_shape_areas,
        )

        loss_idempx2, evals_loss_idempx2, mask_loss_idempx2, _ = self.compute_loss(
            X1_eigenvalues=x2_eigenvalues,
            X2_eigenvalues=x2_eigenvalues,
            union_eigenvalues=x2_eigenvalues,
            union_indices=x2_indices,
            complete_shape_areas=complete_shape_areas,
        )

        loss = (loss + loss_sym + loss_idempx1 + loss_idempx2) / 4

        decloss = 0
        if (
            "gt_decoder" in self.hparams["params"]["loss"]["mask"]
            and self.hparams["params"]["loss"]["mask"]["gt_decoder"]
        ):

            x1_pred_mask = self.predict_mask(self.introudce_noise(x1_eigenvalues))
            decloss_x1 = self.compute_mask_loss(
                pred_union_mask=x1_pred_mask,
                union_indices=x1_indices,
                complete_shape_areas=complete_shape_areas,
            )

            x2_pred_mask = self.predict_mask(self.introudce_noise(x2_eigenvalues))
            decloss_x2 = self.compute_mask_loss(
                pred_union_mask=x2_pred_mask,
                union_indices=x2_indices,
                complete_shape_areas=complete_shape_areas,
            )

            union_pred_mask = self.predict_mask(self.introudce_noise(union_eigenvalues))
            decloss_union = self.compute_mask_loss(
                pred_union_mask=union_pred_mask,
                union_indices=union_indices,
                complete_shape_areas=complete_shape_areas,
            )

            decloss = self.mask_alpha * ((decloss_x1 + decloss_x2 + decloss_union) / 3)
            loss = (loss + decloss) / 2

        # comp_loss = 0
        # if (
        #     "compositional" in self.hparams["params"]["loss"]["mask"]
        #     and self.hparams["params"]["loss"]["mask"]["compositional"]
        # ):
        #     comploss_x1 = self.compositional_loss(
        #         union_eigenvalues,
        #         union_indices,
        #         x1_eigenvalues,
        #         x1_indices,
        #         complete_shape_areas,
        #     )
        #     comploss_x2 = self.compositional_loss(
        #         union_eigenvalues,
        #         union_indices,
        #         x2_eigenvalues,
        #         x2_indices,
        #         complete_shape_areas,
        #     )
        #     comp_loss = (comploss_x1 + comploss_x2) / 2
        #     loss = (loss + comp_loss) / 2

        return {
            "loss": loss,
            "decloss": decloss,
            # "comp_loss": comp_loss,
            "evals_loss": (evals_loss + evals_loss_sym + evals_loss_idempx1 + evals_loss_idempx2) / 4,
            "mask_loss": (mask_loss + mask_loss_sym + mask_loss_idempx1 + mask_loss_idempx2) / 4,
            "pred_unions_evals": pred_union_evals,
            "true_union_eigenvalues": union_eigenvalues,
            "pred_union_mask": pred_union_mask,
            "true_union_mask": union_indices,
            "x1_mask": x1_indices,
            "x2_mask": x2_indices,
        }

    to_log = [
        "loss",
        "evals_loss",
        "mask_loss",
        "decloss",
    ]

    # def compositional_loss(
    #     self,
    #     gt_union_evals,
    #     gt_union_mask,
    #     part_evals,
    #     part_mask,
    #     complete_shape_areas,
    # ):
    #     # predict comp evals
    #     pred_comp_evals_notsym = self.predict_evals(gt_union_evals, part_evals)
    #     pred_comp_evals_sym = self.predict_evals(part_evals, gt_union_evals)
    #     pred_comp_evals = (pred_comp_evals_notsym + pred_comp_evals_sym) / 2
    #
    #     # predict comp mask
    #     pred_comp_mask = self.predict_mask(pred_comp_evals)
    #
    #     # predict gt mask with min area union
    #     gt_comp_mask = (gt_union_mask + part_mask).clamp(max=1)
    #     gt_comp_mask_sym = (gt_union_mask + part_mask[..., self.sym]).clamp(max=1)
    #     gt_comp_mask_area = (gt_comp_mask * complete_shape_areas).sum()
    #     gt_comp_mask_sym_area = (gt_comp_mask_sym * complete_shape_areas).sum()
    #     if gt_comp_mask_sym_area < gt_comp_mask_area:
    #         gt_comp_mask = gt_comp_mask_sym
    #
    #     # compute mask loss
    #     return self.compute_mask_loss(
    #         pred_comp_mask, gt_comp_mask, complete_shape_areas
    #     )

    def training_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)

        losses_to_log = {}

        for loss_name in self.to_log:
            losses_to_log[f"train_metrics/{loss_name}"] = losses[loss_name]

        self.log_dict(
            losses_to_log,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return losses["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        losses = self.compute_losses(batch)

        val_name = self.val_names[dataloader_idx]
        losses_to_log = {}

        for loss_name in self.to_log:
            losses_to_log[f"{val_name}_metrics/{loss_name}"] = losses[loss_name]

        self.log_dict(
            losses_to_log,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        return losses, batch

    def validation_epoch_end(self, dataloaders_outputs: List[List[Any]]) -> None:

        to_log_samples = {}

        wandb_logger = self.logger[0]
        # fixme: broken
        # if self.logger is not None:
        #     for x in self.logger:
        #         if isinstance(x, CustomWandbLogger):
        #             wandb_logger = x

        for dataloader_idx, outputs in enumerate(dataloaders_outputs):
            dl_name = self.val_names[dataloader_idx]

            accuracy_metric = 0
            iou_metric = 0
            count = 0.0

            for output, (batch_input, batch_output) in outputs:
                y_pred = output["pred_union_mask"]
                y_true = output["true_union_mask"]
                areas = batch_output["complete_shape_areas"]
                accuracy_metric += sym_acc(y_pred, y_true, self.sym).sum()
                iou_metric += sym_iou(y_pred, y_true, self.sym, areas).sum()
                count += output["true_union_mask"].shape[0]

            accuracy_metric = accuracy_metric / count
            iou_metric = iou_metric / count

            self.log(
                name=f"{dl_name}_metrics/accuracy",
                value=accuracy_metric,
            )
            self.log(name=f"{dl_name}_metrics/iou", value=iou_metric)

            first_batch, _ = outputs[0]
            for i in range(
                min(
                    first_batch["true_union_mask"].shape[0],
                    self.hparams["params"]["logger"]["log_n_samples_x_dl"],
                )
            ):
                f = plot_shapes_comparison(
                    meshes=[
                        Mesh(
                            v=self.template_vertices,
                            f=self.template_faces,
                            mask=first_batch["true_union_mask"].cpu()[i, ...].numpy(),
                        ),
                        Mesh(
                            v=self.template_vertices,
                            f=self.template_faces,
                            mask=first_batch["pred_union_mask"].cpu()[i, ...].numpy(),
                        ),
                    ],
                    names=[
                        "true_union_mask",
                        "pred_union_mask",
                    ],
                    showscales=[True, False],
                )
                to_log_samples[f"{dl_name}_sample/{i}"] = f

        if wandb_logger is not None:
            wandb_logger.experiment.log(to_log_samples)
