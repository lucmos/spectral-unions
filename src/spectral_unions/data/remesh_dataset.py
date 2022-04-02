from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from nn_core.common.utils import get_env

from spectral_unions.data.dataset_utils import CustomDataset, load_mat


class RemeshDataset(CustomDataset):
    def get_file_list(self) -> List[Union[str, Path]]:
        return self.sample_key_list

    def __init__(
        self,
        hparams: dict,
        dataset_name: str,
        return_mesh=False,
        evals_encoder: Optional[Callable] = None,
    ):
        self.hparams: dict = hparams
        self.dataset_name = dataset_name
        self.dataset_root: Path = Path(get_env(dataset_name))
        self.samples_file = self.dataset_root / "samples.txt"
        self.sample_key_list: List[str] = self.samples_file.read_text().splitlines()

        self.relative_area = self.hparams["params"]["dataset"]["relative_area"]
        self.union_num_eigenvalues: int = hparams["params"]["global"]["union_num_eigenvalues"]
        self.part_num_eigenvalues: int = hparams["params"]["global"]["part_num_eigenvalues"]

        self.template_vertices = torch.from_numpy(load_mat(self.dataset_root / "extras", "VERT.mat"))
        self.template_faces = torch.from_numpy(load_mat(self.dataset_root / "extras", "TRIV.mat").astype("long")) - 1
        self.num_vertices = self.template_vertices.shape[0]

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"No such database: {self.dataset_root}")

        self.return_mesh = return_mesh
        self.evals_encoder = evals_encoder

        self.default_augmentation_num_basis_x1 = 40
        self.default_augmentation_num_basis_x2 = 40

        self.default_augmentation_threshold_x1 = 0.5
        self.default_augmentation_threshold_x2 = 0.5

    def __len__(self) -> int:
        """
        :return: the size of the dataset
        """
        return len(self.sample_key_list)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        """
        Get the `item`-th sample from this dataset.

        :param item: an integer representing the sample index
        :return: the item-th sample
        """
        sample_folder = self.dataset_root / self.sample_key_list[item]
        shape_id: Path = Path(self.sample_key_list[item])
        identity_id, pose_id, part_id = shape_id.parts
        sample_folder_A = sample_folder / "A"
        sample_folder_B = sample_folder / "B"

        x1_indices = torch.from_numpy(load_mat(sample_folder_A, "indices.mat").astype(np.long).squeeze()).float()
        x2_indices = torch.from_numpy(load_mat(sample_folder_B, "indices.mat").astype(np.long).squeeze()).float()
        union_indices = torch.from_numpy(load_mat(sample_folder, "indices.mat").astype(np.long).squeeze()).float()

        x1_evals = torch.from_numpy(load_mat(sample_folder_A, "EVALS.mat").squeeze())[
            : self.part_num_eigenvalues
        ].float()
        x2_evals = torch.from_numpy(load_mat(sample_folder_B, "EVALS.mat").squeeze())[
            : self.part_num_eigenvalues
        ].float()
        union_evals = torch.from_numpy(load_mat(sample_folder, "union_eigenvalues.mat").squeeze())[
            : self.union_num_eigenvalues
        ].float()

        complete_shape_areas = torch.from_numpy(load_mat(sample_folder.parent, "areas.mat").squeeze())
        if self.relative_area:
            complete_shape_areas = complete_shape_areas / complete_shape_areas.sum()

        sample: Dict[str, torch.Tensor] = {
            "item": torch.as_tensor(item, dtype=torch.long),
            "id": str(shape_id),
            "identity_id": str(identity_id),
            "pose_id": str(pose_id),
            "part_id": str(part_id),
            "union_eigenvalues": union_evals[: self.union_num_eigenvalues],
            "X1_eigenvalues": x1_evals[: self.part_num_eigenvalues],
            "X2_eigenvalues": x2_evals[: self.part_num_eigenvalues],
            "union_indices": union_indices,
            "X1_indices": x1_indices,
            "X2_indices": x2_indices,
            "complete_shape_areas": complete_shape_areas,
        }

        if self.return_mesh:
            complete_shape_vertices = load_mat(sample_folder.parent, "VERT.mat")
            sample.update(
                {
                    "complete_shape_vertices": complete_shape_vertices,
                    "complete_shape_faces": self.template_faces,
                    "x1_vertices": torch.from_numpy(load_mat(sample_folder_A, "VERT.mat")),
                    "x1_faces": torch.from_numpy(load_mat(sample_folder_A, "TRIV.mat").astype("long") - 1),
                    "x2_vertices": torch.from_numpy(load_mat(sample_folder_B, "VERT.mat")),
                    "x2_faces": torch.from_numpy(load_mat(sample_folder_B, "TRIV.mat").astype("long") - 1),
                }
            )

        if self.evals_encoder is not None:
            for t in ["union_eigenvalues", "X1_eigenvalues", "X2_eigenvalues"]:
                sample[t] = self.evals_encoder(sample[t])

        return sample


# fixme: broken
load_config = ...
if __name__ == "__main__":
    seed_everything(0)
    config_name = "experiment.yml"

    cfg = load_config(config_name)

    data = Path(get_env("PARTIAL_DATASET_V2"))
    dataset = RemeshDataset(
        cfg,
        dataset_name="REMESH_DATASET_test",
        return_mesh=True,
    )
    print(dataset.samples_file)
    for i in tqdm(dataset):
        print(i)
        break
    assert False
