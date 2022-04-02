import abc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from nn_core.common.utils import get_env

from spectral_unions.data.dataset_utils import CustomDataLoader, CustomDataset, load_mat
from spectral_unions.data.remesh_dataset import RemeshDataset
from spectral_unions.plot.plot_utils import plot_evals_comparison

# fixme: broken
register = ...


class ParentModelV2(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
    ):
        super().__init__()
        self.hparams = hparams

        self.evals_encoder = None
        self.evals_decoder = None
        self.init_evals_encoders()

        # Select loss function
        self.loss_fun_evals = register.get_loss_function(self.hparams["params"]["loss"]["evals"]["fun"])

        self.loss_fun_evaluation = {
            loss_dict["fun"]: {
                "fun": register.get_loss_function(loss_dict["fun"]),
                "params": loss_dict["params"] if loss_dict["params"] else {},
            }
            for loss_dict in self.hparams["params"]["evaluation"]["batch_eval_funcs"]
        }

        self.idemp_losses = (
            self.hparams["params"]["loss"]["evals"]["idemp_losses"]
            if "idemp_losses" in self.hparams["params"]["loss"]["evals"]
            else True
        )

        self.compositional_loss = (
            self.hparams["params"]["loss"]["evals"]["compositional_loss"]
            if "compositional_loss" in self.hparams["params"]["loss"]["evals"]
            else True
        )

        dataset_root: Path = Path(get_env(self.hparams["params"]["dataset"]["name"]))
        if "train_datasplit_folder" in self.hparams["params"]["dataset"]:
            self.train_datasplit = Path(self.hparams["params"]["dataset"]["train_datasplit_folder"]) / "train.txt"
        else:
            self.train_datasplit = Path("datasplit_singleshape") / "train.txt"

        self.template_vertices = load_mat(dataset_root / "extras", "VERT.mat")
        self.template_faces = load_mat(dataset_root / "extras", "TRIV.mat").astype("long") - 1

    def _collate(self, batch):
        x_names = {
            "X1_eigenvalues",
            "X2_eigenvalues",
        }

        y_names = {
            "union_eigenvalues",
            "union_indices",
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

        self.val_files = sorted(((dataset_folder / "val.txt").read_text().splitlines()))

        self.val_dataset: CustomDataset = dataset_class(
            self.hparams,
            self.hparams["params"]["dataset"]["name"],
            self.val_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.test_files = sorted(((dataset_folder / "test.txt").read_text().splitlines()))

        self.test_dataset: CustomDataset = dataset_class(
            self.hparams,
            self.hparams["params"]["dataset"]["name"],
            self.test_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.val_remeshed_dataset = RemeshDataset(
            hparams=self.hparams,
            dataset_name="REMESH_DATASET_val",
            evals_encoder=self.evals_encoder,
        )
        self.test_remeshed_dataset = RemeshDataset(
            hparams=self.hparams,
            dataset_name="REMESH_DATASET_test",
            evals_encoder=self.evals_encoder,
        )

        unkid_dataset_folder = Path(get_env("UNK_IDENTITY_DATASET"))

        self.unkid_val_files = sorted(((unkid_dataset_folder / "val.txt").read_text().splitlines()))

        self.unkid_val_dataset: CustomDataset = dataset_class(
            self.hparams,
            "UNK_IDENTITY_DATASET",
            self.unkid_val_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.unkid_test_files = sorted(((unkid_dataset_folder / "test.txt").read_text().splitlines()))

        self.unkid_test_dataset: CustomDataset = dataset_class(
            self.hparams,
            "UNK_IDENTITY_DATASET",
            self.unkid_test_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        unkid_F_dataset_folder = Path(get_env("UNK_IDENTITY_F_DATASET"))

        self.unkid_F_val_files = sorted(((unkid_F_dataset_folder / "val.txt").read_text().splitlines()))

        self.unkid_F_val_dataset: CustomDataset = dataset_class(
            self.hparams,
            "UNK_IDENTITY_F_DATASET",
            self.unkid_F_val_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.unkid_F_test_files = sorted(((unkid_F_dataset_folder / "test.txt").read_text().splitlines()))

        self.unkid_F_test_dataset: CustomDataset = dataset_class(
            self.hparams,
            "UNK_IDENTITY_F_DATASET",
            self.unkid_F_test_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.val_unkid_remeshed_dataset = RemeshDataset(
            hparams=self.hparams,
            dataset_name="UNK_IDENTITY_REMESH_DATASET_val",
            evals_encoder=self.evals_encoder,
        )
        self.test_unkid_remeshed_dataset = RemeshDataset(
            hparams=self.hparams,
            dataset_name="UNK_IDENTITY_REMESH_DATASET_test",
            evals_encoder=self.evals_encoder,
        )

        self.val_unkid_F_remeshed_dataset = RemeshDataset(
            hparams=self.hparams,
            dataset_name="UNK_IDENTITY_F_REMESH_DATASET_val",
            evals_encoder=self.evals_encoder,
        )
        self.test_unkid_F_remeshed_dataset = RemeshDataset(
            hparams=self.hparams,
            dataset_name="UNK_IDENTITY_F_REMESH_DATASET_test",
            evals_encoder=self.evals_encoder,
        )

        horses_dataset_folder = Path(get_env("PARTIAL_DATASET_V2_horses"))

        self.horse5_files = sorted(((horses_dataset_folder / "horse5.txt").read_text().splitlines()))
        self.horse5_dataset: CustomDataset = dataset_class(
            self.hparams,
            "PARTIAL_DATASET_V2_horses",
            self.horse5_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.horse15_files = sorted(((horses_dataset_folder / "horse15.txt").read_text().splitlines()))
        self.horse15_dataset: CustomDataset = dataset_class(
            self.hparams,
            "PARTIAL_DATASET_V2_horses",
            self.horse15_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.horse7_files = sorted(((horses_dataset_folder / "horse7.txt").read_text().splitlines()))
        self.horse7_dataset: CustomDataset = dataset_class(
            self.hparams,
            "PARTIAL_DATASET_V2_horses",
            self.horse7_files,
            evals_encoder=self.evals_encoder,
            augment=False,
        )

        self.val_names = [
            "val",
            "test",
            "val_remeshed",
            "test_remeshed",
            "val_unkid",
            "test_unkid",
            "val_unkid_F",
            "test_unkid_F",
            "val_unkid_remeshed",
            "test_unkid_remeshed",
            "val_unkid_F_remeshed",
            "test_unkid_F_remeshed",
            "horse5",
            "horse7",
            "horse15",
        ]

        self.val_datasets = {
            "val": self.val_dataset,
            "test": self.test_dataset,
            "val_remeshed": self.val_remeshed_dataset,
            "test_remeshed": self.test_remeshed_dataset,
            "val_unkid": self.unkid_val_dataset,
            "test_unkid": self.unkid_test_dataset,
            "val_unkid_F": self.unkid_F_val_dataset,
            "test_unkid_F": self.unkid_F_test_dataset,
            "val_unkid_remeshed": self.val_unkid_remeshed_dataset,
            "test_unkid_remeshed": self.test_unkid_remeshed_dataset,
            "val_unkid_F_remeshed": self.val_unkid_F_remeshed_dataset,
            "test_unkid_F_remeshed": self.test_unkid_F_remeshed_dataset,
            "horse5": self.horse5_dataset,
            "horse7": self.horse7_dataset,
            "horse15": self.horse15_dataset,
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

    @abc.abstractmethod
    def forward(
        self,
        X1_eigenvalues,
        X2_eigenvalues,
    ):
        pass

    def compute_evals_loss(self, pred_union_evals, union_eigenvalues):
        if self.evals_decoder is not None and not self.hparams["params"]["loss"]["evals"]["loss_between_encodings"]:
            union_eigenvalues = self.evals_decoder(union_eigenvalues)
            pred_union_evals = self.evals_decoder(pred_union_evals)
        evals_loss = self.loss_fun_evals(
            y_pred=pred_union_evals,
            y_true=union_eigenvalues,
            **self.hparams["params"]["loss"]["evals"]["params"]
            if self.hparams["params"]["loss"]["evals"]["params"] is not None
            else {},
        )
        return evals_loss

    def compute_loss(
        self,
        X1_eigenvalues,
        X2_eigenvalues,
        union_eigenvalues,
    ):
        pred_union_evals = self(
            X1_eigenvalues,
            X2_eigenvalues,
        )
        evals_loss = self.compute_evals_loss(pred_union_evals=pred_union_evals, union_eigenvalues=union_eigenvalues)
        return evals_loss, pred_union_evals

    def compute_losses(self, batch):
        x, y_trues = batch
        x1_eigenvalues = x["X1_eigenvalues"]
        x2_eigenvalues = x["X2_eigenvalues"]

        union_eigenvalues = y_trues["union_eigenvalues"]

        count = 2
        loss_notsym, pred_notsym = self.compute_loss(
            X1_eigenvalues=x1_eigenvalues,
            X2_eigenvalues=x2_eigenvalues,
            union_eigenvalues=union_eigenvalues,
        )

        loss_sym, pred_sym = self.compute_loss(
            X1_eigenvalues=x2_eigenvalues,
            X2_eigenvalues=x1_eigenvalues,
            union_eigenvalues=union_eigenvalues,
        )

        pred_union_evals = (pred_sym + pred_notsym) / 2

        if self.idemp_losses:
            loss_x1, pred_idemp_x1 = self.compute_loss(
                X1_eigenvalues=x1_eigenvalues,
                X2_eigenvalues=x1_eigenvalues,
                union_eigenvalues=x1_eigenvalues,
            )

            loss_x2, pred_idemp_x2 = self.compute_loss(
                X1_eigenvalues=x2_eigenvalues,
                X2_eigenvalues=x2_eigenvalues,
                union_eigenvalues=x2_eigenvalues,
            )
            count += 2
        else:
            loss_x1 = 0
            loss_x2 = 0

        if self.compositional_loss:

            loss_comp_x1, _ = self.compute_loss(
                X1_eigenvalues=pred_union_evals,
                X2_eigenvalues=x1_eigenvalues,
                union_eigenvalues=union_eigenvalues,
            )
            loss_comp_sym_x1, _ = self.compute_loss(
                X1_eigenvalues=x1_eigenvalues,
                X2_eigenvalues=pred_union_evals,
                union_eigenvalues=union_eigenvalues,
            )

            loss_comp_x2, _ = self.compute_loss(
                X1_eigenvalues=pred_union_evals,
                X2_eigenvalues=x2_eigenvalues,
                union_eigenvalues=union_eigenvalues,
            )
            loss_comp_sym_x2, _ = self.compute_loss(
                X1_eigenvalues=x2_eigenvalues,
                X2_eigenvalues=pred_union_evals,
                union_eigenvalues=union_eigenvalues,
            )
            loss_comp = (loss_comp_x1 + loss_comp_x2 + loss_comp_sym_x1 + loss_comp_sym_x2) / 4
            count += 1
        else:
            loss_comp = 0

        loss = (loss_notsym + loss_sym + loss_x1 + loss_x2 + loss_comp) / count

        return {
            "loss": loss,
            "loss_notsym": loss_notsym,
            "loss_sym": loss_sym,
            "loss_x1": loss_x1,
            "loss_x2": loss_x2,
            "loss_comp": loss_comp,
            "pred_unions_evals": pred_union_evals,
            "true_union_evals": union_eigenvalues,
            "X1_eigenvalues": x["X1_eigenvalues"],
            "X2_eigenvalues": x["X2_eigenvalues"],
        }

    to_log = [
        "loss",
        "loss_notsym",
        "loss_sym",
        "loss_x1",
        "loss_x2",
    ]

    def training_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)

        losses_to_log = {}

        for loss_name in self.to_log:
            losses_to_log[f"train/losses/{loss_name}"] = losses[loss_name]

        self.log_dict(
            losses_to_log,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return losses["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        losses_to_log = {}
        losses = self.compute_losses(batch)

        val_name = self.val_names[dataloader_idx]
        for loss_name in self.to_log:
            losses_to_log[f"{val_name}/losses/{loss_name}"] = losses[loss_name]

        raw_pred_unions_evals = losses["pred_unions_evals"]
        raw_true_unions_evals = losses["true_union_evals"]

        if self.evals_decoder is not None:
            raw_pred_unions_evals = self.evals_decoder(raw_pred_unions_evals)
            raw_true_unions_evals = self.evals_decoder(raw_true_unions_evals)

        for eval_loss_name, eval_loss in self.loss_fun_evaluation.items():
            losses_to_log[f"{val_name}/evaluation/{eval_loss_name}"] = (
                eval_loss["fun"](
                    raw_pred_unions_evals,
                    raw_true_unions_evals,
                    **eval_loss["params"],
                ),
            )

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

            first_batch, _ = outputs[0]
            for i in range(
                min(
                    first_batch["true_union_evals"].shape[0],
                    self.hparams["params"]["logger"]["log_n_samples_x_dl"],
                )
            ):
                raw_pred_unions_evals = first_batch["pred_unions_evals"][i, :]
                raw_true_unions_evals = first_batch["true_union_evals"][i, :]
                raw_X1_eigenvalues = first_batch["X1_eigenvalues"][i, :]
                raw_X2_eigenvalues = first_batch["X2_eigenvalues"][i, :]

                if self.evals_decoder is not None:
                    raw_pred_unions_evals = self.evals_decoder(raw_pred_unions_evals)
                    raw_true_unions_evals = self.evals_decoder(raw_true_unions_evals)
                    raw_X1_eigenvalues = self.evals_decoder(raw_X1_eigenvalues)
                    raw_X2_eigenvalues = self.evals_decoder(raw_X2_eigenvalues)

                f = plot_evals_comparison(
                    pred_values=raw_pred_unions_evals.cpu().numpy(),
                    union_values=raw_true_unions_evals.cpu().numpy(),
                    x1_values=raw_X1_eigenvalues.cpu().numpy(),
                    x2_values=raw_X2_eigenvalues.cpu().numpy(),
                )
                to_log_samples[f"{dl_name}/plot/sample{i}"] = f

        if wandb_logger is not None:
            wandb_logger.experiment.log(to_log_samples)
