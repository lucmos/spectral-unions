from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import hydra
import numpy as np
import omegaconf
import scipy
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import get_env

from spectral_unions.data.dataset_utils import (
    CustomDataset,
    get_augmented_mask,
    get_mask_evals,
    get_vertex_areas,
    load_mat,
    multi_one_hot,
)
from spectral_unions.spectral.eigendecomposition import ShapeEigendecomposition, ShapeHamiltonianAugmenter


class PartialAugmentedDataset(CustomDataset):
    def get_file_list(self) -> List[Union[str, Path]]:
        return self.sample_key_list

    def __init__(
        self,
        dataset_name: str,
        sample_key_list: List[str],
        boundary_conditions: str,
        union_num_eigenvalues: str,
        part_num_eigenvalues: str,
        min_union_prior: bool,
        independent_augmentation: bool,
        relative_area: bool,
        gpus: int,
        introduce_input_evals_noise: bool = False,
        return_mesh=False,
        evals_encoder: Optional[Callable] = None,
        augment: bool = True,
    ):
        self.augment = augment
        self.boundary_conditions = boundary_conditions.lower()
        assert self.boundary_conditions == "dirichlet"
        self.evals_filename = f"evals_{self.boundary_conditions}.mat"

        self.min_union_prior = min_union_prior
        self.independent_augmentation = independent_augmentation
        self.introduce_input_evals_noise = introduce_input_evals_noise
        self.relative_area = relative_area

        self.sample_key_list: List[str] = sample_key_list
        self.dataset_name = dataset_name
        self.dataset_root: Path = Path(get_env(dataset_name))

        self.union_num_eigenvalues: int = union_num_eigenvalues
        self.part_num_eigenvalues: int = part_num_eigenvalues

        self.template_vertices = torch.from_numpy(load_mat(self.dataset_root / "extras", "VERT.mat"))
        self.template_faces = torch.from_numpy(load_mat(self.dataset_root / "extras", "TRIV.mat").astype("long")) - 1
        self.num_vertices = self.template_vertices.shape[0]

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"No such database: {self.dataset_root}")

        self.return_mesh = return_mesh
        self.evals_encoder = evals_encoder

        self.template_eigen = ShapeEigendecomposition(
            self.template_vertices,
            self.template_faces,
        )

        self.vertex2faces = torch.sparse_coo_tensor(
            indices=torch.stack(
                (
                    self.template_faces.transpose(1, 0).flatten(),
                    torch.arange(self.template_faces.shape[0]).repeat(3),
                ),
                dim=0,
            ),
            values=torch.ones(self.template_faces.shape[0] * 3),
            size=(self.template_vertices.shape[0], self.template_faces.shape[0]),
        )

        self.sym_map = torch.from_numpy(
            scipy.io.loadmat(self.dataset_root / "extras" / "SMPLsym.mat")["idxs"].squeeze().astype("long") - 1
        )

        self.default_augmentation_num_basis_x1 = 40
        self.default_augmentation_num_basis_x2 = 40

        self.default_augmentation_threshold_x1 = 0.5
        self.default_augmentation_threshold_x2 = 0.5

        self.gpus = gpus
        self.device = "cpu" if gpus == 0 else "cuda"

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

        complete_shape_areas = get_vertex_areas(
            self.vertex2faces,
            self.template_faces,
            self.dataset_name,
            identity_id,
            pose_id,
            relative_area=self.relative_area,
        )

        x1_indices = multi_one_hot(
            load_mat(sample_folder_A, "indices.mat").astype(np.long),
            self.num_vertices,
        ).squeeze()
        x2_indices = multi_one_hot(
            load_mat(sample_folder_B, "indices.mat").astype(np.long),
            self.num_vertices,
        ).squeeze()

        if self.independent_augmentation:
            x1_num_basis = ShapeEigendecomposition.get_random_num_basis()
            x1_threshold = ShapeHamiltonianAugmenter.get_random_discretized_threshold()
            x2_num_basis = ShapeEigendecomposition.get_random_num_basis()
            x2_threshold = ShapeHamiltonianAugmenter.get_random_discretized_threshold()
        else:
            x1_num_basis = x2_num_basis = ShapeEigendecomposition.get_random_num_basis()
            x1_threshold = x2_threshold = ShapeHamiltonianAugmenter.get_random_discretized_threshold()

        if self.augment:
            x1_indices = get_augmented_mask(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x1_indices,
                num_basis_vectors=x1_num_basis,
                threshold=x1_threshold,
                device=self.device,
            )
            x2_indices = get_augmented_mask(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x2_indices,
                num_basis_vectors=x2_num_basis,
                threshold=x2_threshold,
                device=self.device,
            )
        else:
            x1_indices = get_augmented_mask(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x1_indices,
                num_basis_vectors=self.default_augmentation_num_basis_x1,
                threshold=self.default_augmentation_threshold_x1,
                device=self.device,
            )
            x2_indices = get_augmented_mask(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x2_indices,
                num_basis_vectors=self.default_augmentation_num_basis_x2,
                threshold=self.default_augmentation_threshold_x2,
                device=self.device,
            )

        union_indices = (x1_indices + x2_indices).bool().float()

        if self.min_union_prior:
            x2_indices_sym = x2_indices[self.sym_map]
            union_indices_sym = (x1_indices + x2_indices_sym).bool().float()

            union_area = (union_indices * complete_shape_areas).sum()
            union_area_sym = (union_indices_sym * complete_shape_areas).sum()
            if union_area_sym < union_area:
                x2_indices = x2_indices_sym
                union_indices = union_indices_sym

        union_evals = get_mask_evals(
            template_eigen=self.template_eigen,
            dataset_name=self.dataset_name,
            identity_id=identity_id,
            pose_id=pose_id,
            mask=union_indices,
            k=self.union_num_eigenvalues,
            device=self.device,
        )
        x1_evals = get_mask_evals(
            template_eigen=self.template_eigen,
            dataset_name=self.dataset_name,
            identity_id=identity_id,
            pose_id=pose_id,
            mask=x1_indices,
            k=self.part_num_eigenvalues,
            device=self.device,
        )
        x2_evals = get_mask_evals(
            template_eigen=self.template_eigen,
            dataset_name=self.dataset_name,
            identity_id=identity_id,
            pose_id=pose_id,
            mask=x2_indices,
            k=self.part_num_eigenvalues,
            device=self.device,
        )

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
            complete_shape_faces = self.template_faces

            sample.update(
                {
                    "complete_shape_vertices": complete_shape_vertices,
                    "complete_shape_faces": complete_shape_faces,
                }
            )

        if self.introduce_input_evals_noise:
            for t in ["X1_eigenvalues", "X2_eigenvalues"]:
                sample[t] = self.introudce_noise(sample[t])

        if self.evals_encoder is not None:
            for t in ["union_eigenvalues", "X1_eigenvalues", "X2_eigenvalues"]:
                sample[t] = self.evals_encoder(sample[t])

        return sample

    def introudce_noise(self, evals):
        return evals * (1 + torch.randn_like(evals, device=evals.device) * 1e-3)


# TODO: remove all this if not needed
# def _generate_sample_cache(
#     template_eigen,
#     dataset_name,
#     identity_id,
#     pose_id,
#     x1_indices,
#     x2_indices,
#     x1_num_basis,
#     x1_threshold,
#     x2_num_basis,
#     x2_threshold,
#     num_eigenvalues,
#     sym_map,
# ):
#     x1_indices = get_augmented_mask(
#         template_eigen=template_eigen,
#         dataset_name=dataset_name,
#         identity_id=identity_id,
#         pose_id=pose_id,
#         mask=x1_indices,
#         num_basis_vectors=x1_num_basis,
#         threshold=x1_threshold,
#     )
#
#     x2_indices = get_augmented_mask(
#         template_eigen=template_eigen,
#         dataset_name=dataset_name,
#         identity_id=identity_id,
#         pose_id=pose_id,
#         mask=x2_indices,
#         num_basis_vectors=x2_num_basis,
#         threshold=x2_threshold,
#     )
#
#     for x2_indices_sym in [x2_indices, x2_indices[sym_map]]:
#
#         union_indices = (x1_indices + x2_indices_sym).bool().float()
#
#         x1_out = get_mask_evals(
#             template_eigen=template_eigen,
#             dataset_name=dataset_name,
#             identity_id=identity_id,
#             pose_id=pose_id,
#             mask=x1_indices,
#             k=num_eigenvalues,
#         )
#         x2_out = get_mask_evals(
#             template_eigen=template_eigen,
#             dataset_name=dataset_name,
#             identity_id=identity_id,
#             pose_id=pose_id,
#             mask=x2_indices_sym,
#             k=num_eigenvalues,
#         )
#         union_out = get_mask_evals(
#             template_eigen=template_eigen,
#             dataset_name=dataset_name,
#             identity_id=identity_id,
#             pose_id=pose_id,
#             mask=union_indices,
#             k=num_eigenvalues,
#         )
#
#
# def _regenerate_cache():
#     from src.mask_prediction.partial_dataset_v2 import PartialDatasetV2
#
#     # todo: genera solo autovalori unione in cui i parametri di augmentation fra x1 e x2
#     # sono uguali (threshold e num_basis)
#     load_envs()
#     config_name = "experiment.yml"
#
#     cfg = load_config(config_name)
#
#     dataset_name = "PARTIAL_DATASET_V2_horses"
#
#     data = Path(safe_get_env(dataset_name))
#     dataset_root = Path(safe_get_env(dataset_name))
#
#     split = Path(cfg["params"]["dataset"]["train_datasplit_folder"]) / "train.txt"
#     split = "samples.txt"
#     trainset = (dataset_root / split).read_text().splitlines()
#
#     validation_dataset = False  # if true don't generate all augmentations
#
#     dataset = PartialDatasetV2(cfg, dataset_name, trainset, return_mesh=False, no_evals_and_area=True)
#
#     template_eigen = ShapeEigendecomposition(
#         torch.from_numpy(dataset.template_vertices),
#         torch.from_numpy(dataset.template_faces.astype("long")),
#     )
#
#     sym_map = torch.from_numpy(
#         scipy.io.loadmat(dataset.dataset_root / "extras" / "SMPLsym.mat")["idxs"].squeeze().astype("long") - 1
#     )
#
#     for sample in tqdm(dataset):
#         identity_id = sample["identity_id"]
#         pose_id = sample["pose_id"]
#         if not validation_dataset:
#             for x1_threshold in ShapeHamiltonianAugmenter.thresholds:
#                 for x2_threshold in ShapeHamiltonianAugmenter.thresholds:
#                     for x1_num_basis in ShapeEigendecomposition.possible_num_basis_vectors:
#                         for x2_num_basis in ShapeEigendecomposition.possible_num_basis_vectors:
#                             _generate_sample_cache(
#                                 template_eigen=template_eigen,
#                                 dataset_name=dataset.dataset_name,
#                                 identity_id=identity_id,
#                                 pose_id=pose_id,
#                                 x1_indices=sample["X1_indices"],
#                                 x2_indices=sample["X2_indices"],
#                                 x1_num_basis=x1_num_basis,
#                                 x1_threshold=x1_threshold,
#                                 x2_num_basis=x2_num_basis,
#                                 x2_threshold=x2_threshold,
#                                 num_eigenvalues=dataset.union_num_eigenvalues,
#                                 sym_map=sym_map,
#                             )
#
#         _generate_sample_cache(
#             template_eigen=template_eigen,
#             dataset_name=dataset.dataset_name,
#             identity_id=identity_id,
#             pose_id=pose_id,
#             x1_indices=sample["X1_indices"],
#             x2_indices=sample["X2_indices"],
#             x1_num_basis=40,
#             x1_threshold=0.5,
#             x2_num_basis=40,
#             x2_threshold=0.5,
#             num_eigenvalues=dataset.union_num_eigenvalues,
#             sym_map=sym_map,
#         )
#
#
# if __name__ == "__main__":
#     seed_everything(0)
#     _regenerate_cache()
#     assert False
#     load_envs()
#     config_name = "experiment.yml"
#
#     seed_everything(0)
#     cfg = load_config(config_name)
#     dataset_name = "PARTIAL_DATASET_V2_horses"
#
#     data = Path(safe_get_env(dataset_name))
#     dataset_root = Path(safe_get_env(dataset_name))
#
#     split = Path(cfg["params"]["dataset"]["train_datasplit_folder"]) / "train.txt"
#     sample_key_list = (dataset_root / split).read_text().splitlines()
#
#     dataset = PartialAugmentedDataset(cfg, dataset_name, sample_key_list, augment=True, return_mesh=False)
#     for i in tqdm(dataset):
#         pass
#     assert False
#     # # #
#     # #
#     # # # loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
#     # # #
#     # for x in tqdm(dataset):
#     #     pass
#     # #
#     # # # todo Dataset[0] non prende da cache?
#     # _generate_sample_cache(
#     #     dataset.template_eigen,
#     #     dataset.dataset_root,
#     #     "shape000024",
#     #     "pose0000235",
#     #     dataset[0]["X1_indices"],
#     #     dataset[0]["X2_indices"],
#     #     num_basis=ShapeEigendecomposition.get_random_num_basis(),
#     #     threshold=ShapeHamiltonianAugmenter.get_random_discretized_threshold(),
#     #     num_eigenvalues=20,
#     # )
#
#     i = np.random.randint(0, len(dataset))
#     print(i)
#     i = 37344
#     sample = dataset[i]
#
#     plot_shapes_comparison(
#         meshes=[
#             mesh(
#                 v=dataset.template_vertices.numpy(),
#                 f=dataset.template_faces.numpy(),
#                 color=sample["X1_indices"],
#             ),
#             mesh(
#                 v=dataset.template_vertices.numpy(),
#                 f=dataset.template_faces.numpy(),
#                 color=sample["X2_indices"],
#             ),
#             mesh(
#                 v=dataset.template_vertices.numpy(),
#                 f=dataset.template_faces.numpy(),
#                 color=sample["union_indices"],
#             ),
#         ],
#         names=["x1", "x2", "union"],
#         showscales=[False, False, False],
#     ).show()
#     assert False
#     # ind = sample["union_indices"]
#     #
#     # print("next")
#     # s = set()
#     #
#     # for i in tqdm(range(10000)):
#     #
#     #     x1_indices = get_augmented_mask(
#     #         template_eigen=dataset.template_eigen,
#     #         dataset_root=dataset.dataset_root,
#     #         identity_id=sample["identity_id"],
#     #         pose_id=sample["pose_id"],
#     #         mask=sample["X1_indices"],
#     #         num_basis_vectors=ShapeEigendecomposition.get_random_num_basis(),
#     #         threshold=ShapeHamiltonianAugmenter.get_random_discretized_threshold(),
#     #     )
#     #     s.add(x1_indices)
#     #     get_mask_evals(
#     #         template_eigen=dataset.template_eigen,
#     #         dataset_root=dataset.dataset_root,
#     #         identity_id=sample["identity_id"],
#     #         pose_id=sample["pose_id"],
#     #         mask=sample["X1_indices"],
#     #     )
#     #
#     # print(len(s))
#     # #
#     #
#     # assert False
#     # mat = scipy.io.loadmat("/home/luca/Desktop/test_3.mat")
#     #
#     # VERT = torch.from_numpy(mat["VERT"]).float().cpu()
#     # TRIV = torch.from_numpy(mat["TRIV"]).long().cpu()
#     # ind = torch.from_numpy(mat["ind"]).float().cpu()[0]
#
#     # vertex2faces = VF_adjacency_matrix(VERT, TRIV)
#     #
#     # assert False
#
#     # VERT = torch.from_numpy(dataset.template_vertices)
#     # TRIV = torch.from_numpy(dataset.template_faces.astype("long"))
#     # ind = dataset[2550]["union_indices"]
#     # sym = (
#     #     scipy.io.loadmat(
#     #         "/home/luca/Repositories/partial-shape-generator/partial_dataset_v2/SMPLsym.mat"
#     #     )["idxs"]
#     #     .squeeze()
#     #     .astype("long")
#     # )
#     # S = (
#     #     torch.sparse_coo_tensor(
#     #         indices=torch.stack(
#     #             (
#     #                 torch.arange(sym.shape[0]),
#     #                 torch.from_numpy(sym) - 1,
#     #             ),
#     #             dim=0,
#     #         ),
#     #         values=torch.ones(sym.shape[0]),
#     #         size=(VERT.shape[0], VERT.shape[0]),
#     #     )
#     #     .to_dense()
#     #     .bool()
#     #     .float()
#     # )
#     # sym_union_indices = torch.einsum("ij, bj -> bi", S, ind[None, :]).bool().float()[0]
#     #
#     # eigen = ShapeEigendecomposition(VERT, TRIV)
#     # s = ShapeHamiltonianAugmenter(eigen, VERT)
#     #
#     # ks = [0 for i in range(1)]
#     # plot_shapes_comparison(
#     #     meshes=[
#     #         mesh(
#     #             v=s.vertices.numpy(),
#     #             f=eigen.faces.numpy(),
#     #             color=ind,
#     #         ),
#     #         mesh(
#     #             v=s.vertices.numpy(),
#     #             f=eigen.faces.numpy(),
#     #             color=sym_union_indices,
#     #         ),
#     #     ],
#     #     names=["sym1", "sym1"],
#     #     showscales=[False, True],
#     # ).show()
#     #
#     # assert False
#     # import torch
#     #
#     # eigen = ShapeEigendecomposition(VERT, TRIV)
#     # s = ShapeHamiltonianAugmenter(eigen, VERT)
#     #
#     # ks = [0 for i in range(1)]
#     # plot_shapes_comparison(
#     #     meshes=[
#     #         mesh(
#     #             v=s.vertices.numpy(),
#     #             f=eigen.faces.numpy(),
#     #             color=ind,
#     #         ),
#     #         mesh(
#     #             v=s.vertices.numpy(),
#     #             f=eigen.faces.numpy(),
#     #             color=s.mask_random_augmentation(
#     #                 ind,
#     #                 eigen.get_random_num_basis(),
#     #                 s.get_random_discretized_threshold(),
#     #                 True,
#     #             ),
#     #         ),
#     #     ]
#     #     + [
#     #         mesh(
#     #             v=s.vertices.numpy(),
#     #             f=eigen.faces.numpy(),
#     #             color=s.mask_random_augmentation(
#     #                 ind,
#     #                 eigen.get_random_num_basis(),
#     #                 s.get_random_discretized_threshold(),
#     #             ).numpy(),
#     #         )
#     #         for x in ks
#     #     ],
#     #     names=["gournd truth", "projected"] + [f"_" for x in ks],
#     #     showscales=[False, True] + [False] * len(ks),
#     # ).show()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """

    data = Path(get_env("PARTIAL_DATASET_V2"))
    trainset = (PROJECT_ROOT / data / "datasplit_singleshape" / "train.txt").read_text().splitlines()

    dataset: Dataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, sample_key_list=trainset, _recursive_=False)
    loader = DataLoader(dataset, batch_size=32, num_workers=12, persistent_workers=False)
    for x in tqdm(loader):
        print(x["union_indices"].shape)
        break


if __name__ == "__main__":
    main()
