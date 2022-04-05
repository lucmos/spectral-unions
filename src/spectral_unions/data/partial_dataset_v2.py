from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import hydra
import numpy as np
import torch
from omegaconf import omegaconf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import get_env

from spectral_unions.data.dataset_utils import CustomDataset, load_mat, multi_one_hot


class PartialDatasetV2(CustomDataset):
    def get_file_list(self) -> List[Union[str, Path]]:
        return self.sample_key_list

    def __init__(
        self,
        dataset_name: str,
        sample_key_list: List[str],
        boundary_conditions: str,
        union_num_eigenvalues: str,
        part_num_eigenvalues: str,
        return_mesh=False,
        evals_encoder: Optional[Callable] = None,
        no_evals_and_area=False,
    ):
        self.boundary_conditions = boundary_conditions.lower()
        assert self.boundary_conditions in {"neumann", "dirichlet"}
        self.evals_filename = f"evals_{self.boundary_conditions}.mat"

        self.sample_key_list: List[str] = sample_key_list
        self.dataset_name = dataset_name
        self.dataset_root: Path = Path(get_env(dataset_name))

        self.union_num_eigenvalues: int = union_num_eigenvalues
        self.part_num_eigenvalues: int = part_num_eigenvalues

        self.template_vertices = load_mat(self.dataset_root / "extras", "VERT.mat")
        self.template_faces = load_mat(self.dataset_root / "extras", "TRIV.mat") - 1
        self.num_vertices = self.template_vertices.shape[0]

        self.no_evals_and_area = no_evals_and_area

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"No such database: {self.dataset_root}")

        self.return_meshes = return_mesh
        self.evals_encoder = evals_encoder

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
        # assert self.evals_filename == "evals_neumann.mat"
        sample_folder = self.dataset_root / self.sample_key_list[item]
        sample_folder_A = sample_folder / "A"
        sample_folder_B = sample_folder / "B"

        union_evals = (
            torch.as_tensor(load_mat(sample_folder, self.evals_filename), dtype=torch.float).squeeze()
            if not self.no_evals_and_area
            else torch.zeros(1)
        )
        x1_evals = (
            torch.as_tensor(load_mat(sample_folder_A, self.evals_filename), dtype=torch.float).squeeze()
            if not self.no_evals_and_area
            else torch.zeros(1)
        )
        x2_evals = (
            torch.as_tensor(load_mat(sample_folder_B, self.evals_filename), dtype=torch.float).squeeze()
            if not self.no_evals_and_area
            else torch.zeros(1)
        )

        union_indices = multi_one_hot(
            load_mat(sample_folder, "indices.mat").astype(np.long),
            self.num_vertices,
        ).squeeze()
        x1_indices = multi_one_hot(
            load_mat(sample_folder_A, "indices.mat").astype(np.long),
            self.num_vertices,
        ).squeeze()
        x2_indices = multi_one_hot(
            load_mat(sample_folder_B, "indices.mat").astype(np.long),
            self.num_vertices,
        ).squeeze()

        union_area = (
            torch.as_tensor(load_mat(sample_folder, "area.mat"), dtype=torch.float).squeeze()
            if not self.no_evals_and_area
            else torch.zeros(1)
        )
        x1_area = (
            torch.as_tensor(load_mat(sample_folder_A, "area.mat"), dtype=torch.float).squeeze()
            if not self.no_evals_and_area
            else torch.zeros(1)
        )
        x2_area = (
            torch.as_tensor(load_mat(sample_folder_B, "area.mat"), dtype=torch.float).squeeze()
            if not self.no_evals_and_area
            else torch.zeros(1)
        )

        shape_id: Path = Path(self.sample_key_list[item])
        identity_id, pose_id, part_id = shape_id.parts

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
            "union_area": union_area,
            "X1_area": x1_area,
            "X2_area": x2_area,
        }

        if self.return_meshes:
            complete_shape_vertices = load_mat(sample_folder.parent, "VERT.mat")
            complete_shape_faces = self.template_faces
            union_vertices = load_mat(sample_folder, "VERT.mat")
            union_faces = load_mat(sample_folder, "TRIV.mat").astype(np.long)
            X1_vertices = load_mat(sample_folder_A, "VERT.mat")
            X1_faces = load_mat(sample_folder_A, "TRIV.mat").astype(np.long)
            X2_vertices = load_mat(sample_folder_B, "VERT.mat")
            X2_faces = load_mat(sample_folder_B, "TRIV.mat").astype(np.long)

            sample.update(
                {
                    "complete_shape_vertices": complete_shape_vertices,
                    "complete_shape_faces": complete_shape_faces,
                    "union_vertices": union_vertices,
                    "union_faces": union_faces,
                    "X1_vertices": X1_vertices,
                    "X1_faces": X1_faces,
                    "X2_vertices": X2_vertices,
                    "X2_faces": X2_faces,
                }
            )

        if self.evals_encoder is not None:
            for t in ["union_eigenvalues", "X1_eigenvalues", "X2_eigenvalues"]:
                sample[t] = self.evals_encoder(sample[t])

        return sample


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """

    data = Path(get_env("PARTIAL_DATASET_V2"))
    trainset = (data / "datasplit_singleshape" / "train.txt").read_text().splitlines()

    dataset: Dataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, sample_key_list=trainset, _recursive_=False)
    loader = DataLoader(dataset, batch_size=32, num_workers=12, persistent_workers=False)
    for x in tqdm(loader):
        print(x["union_indices"].shape)
        break


if __name__ == "__main__":
    main()
