import hashlib
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import scipy
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from nn_core.common.utils import get_env

from spectral_unions.data.dataset_utils import (
    CustomDataset,
    cache_path,
    get_augmented_mask,
    get_mask_evals,
    get_vertex_areas,
    load_mat,
    multi_one_hot,
)
from spectral_unions.spectral.eigendecomposition import ShapeEigendecomposition, ShapeHamiltonianAugmenter

# fixme: broken
load_config = ...


class RandomAugmentedDataset(CustomDataset):
    possible_random_thresholds = [0.25, 0.5, 0.75]

    def get_file_list(self) -> List[Union[str, Path]]:
        return self.sample_key_list

    def __init__(
        self,
        hparams: dict,
        dataset_name: str,
        sample_key_list: List[str],
        return_mesh=False,
        evals_encoder: Optional[Callable] = None,
        augment: bool = False,
    ):
        self.hparams: dict = hparams
        self.augment = augment
        self.boundary_conditions = self.hparams["params"]["global"]["boundary_conditions"].lower()
        assert self.boundary_conditions == "dirichlet"
        self.evals_filename = f"evals_{self.boundary_conditions}.mat"

        self.min_union_prior = self.hparams["params"]["dataset"]["min_union_prior"]
        self.independent_augmentation = self.hparams["params"]["dataset"]["independent_augmentation"]
        self.introduce_input_evals_noise = (
            self.hparams["params"]["dataset"]["introduce_input_evals_noise"]
            if "introduce_input_evals_noise" in self.hparams["params"]["dataset"]
            else False
        )
        self.relative_area = self.hparams["params"]["dataset"]["relative_area"]

        self.sample_key_list: List[str] = sample_key_list
        self.dataset_name = dataset_name
        self.dataset_root: Path = Path(get_env(dataset_name))

        self.union_num_eigenvalues: int = hparams["params"]["global"]["union_num_eigenvalues"]
        self.part_num_eigenvalues: int = hparams["params"]["global"]["part_num_eigenvalues"]

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

        split = Path(self.hparams["params"]["dataset"]["train_datasplit_folder"]) / "train.txt"
        self.intersection_classes = self.get_eq_classes_uniq(
            datasplit_name=split,
            n_common_vertices=40,
            overwrite=False,
        )
        self.count = 0

    def get_eq_classes_uniq(self, datasplit_name, n_common_vertices, overwrite=False):
        from spectral_unions.data.partial_dataset_v2 import PartialDatasetV2

        filename = f"eq_uniq_classes_{self.dataset_name}_{datasplit_name}_{n_common_vertices}.pickle"
        filepath = cache_path / "intersections" / filename

        if filepath.exists() and not overwrite:
            with filepath.open("rb") as f:
                return pickle.load(f)
        else:
            dataset = PartialDatasetV2(
                self.hparams,
                self.dataset_name,
                self.sample_key_list,
                return_mesh=False,
                no_evals_and_area=True,
            )

            template_eigen = ShapeEigendecomposition(
                torch.from_numpy(dataset.template_vertices),
                torch.from_numpy(dataset.template_faces.astype("long")),
            )

            mask_hash2name = {}
            name2indices = {}
            for sample in tqdm(dataset):
                for folder_name, name in [
                    ("A", "X1_indices"),
                    ("B", "X2_indices"),
                    ("", "union_indices"),
                ]:
                    for threshold in RandomAugmentedDataset.possible_random_thresholds:
                        mask = sample[name]
                        mask = get_augmented_mask(
                            template_eigen=template_eigen,
                            dataset_name=self.dataset_name,
                            identity_id=sample["identity_id"],
                            pose_id=sample["pose_id"],
                            mask=mask,
                            num_basis_vectors=40,
                            threshold=threshold,
                        )
                        mask_hash = hashlib.md5(mask.numpy().tobytes()).hexdigest()
                        if mask_hash not in mask_hash2name:
                            partname = str(Path(sample["id"]) / folder_name), threshold
                            mask_hash2name[mask_hash] = partname
                            name2indices[partname] = mask

            uniq_masks_part_names = mask_hash2name.values()
            eq_uniq = defaultdict(set)
            for sample1 in tqdm(dataset, desc="Intersections eq classes"):
                x1_part_name = str(Path(sample1["id"]) / "A"), 0.5
                x1_indices = sample1["X1_indices"]
                x1_indices = get_augmented_mask(
                    template_eigen=template_eigen,
                    dataset_name=self.dataset_name,
                    identity_id=sample1["identity_id"],
                    pose_id=sample1["pose_id"],
                    mask=x1_indices,
                    num_basis_vectors=40,
                    threshold=0.5,
                )

                for x2_name in uniq_masks_part_names:
                    x2_indices = name2indices[x2_name]
                    if (x1_indices * x2_indices).sum() > 50:
                        eq_uniq[x1_part_name].add(x2_name)

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open("wb") as f:
                pickle.dump(eq_uniq, f)
                print(f"Wrote eq classes: {filepath}")
                return eq_uniq

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
        sample_key = self.sample_key_list[item]
        shape_id: Path = Path(sample_key)
        identity_id, pose_id, part_id = shape_id.parts

        # if identity_id == "shape000024":
        #     choices = (True, False)
        # else:
        #     choices = (True,)
        choices = (
            True,
            False,
        )

        if random.choice(choices):
            sample_folder_A = self.dataset_root / sample_key / "A"
            x1_indices = multi_one_hot(
                load_mat(sample_folder_A, "indices.mat").astype(np.long),
                self.num_vertices,
            ).squeeze()

            random_folder_B = self.intersection_classes[(str(Path(self.sample_key_list[item]) / "A"), 0.5)]
            sample_folder_B, x2_random_threshold = random.sample(random_folder_B, k=1)[0]
            sample_folder_B = self.dataset_root / sample_folder_B
            x2_indices = multi_one_hot(
                load_mat(sample_folder_B, "indices.mat").astype(np.long),
                self.num_vertices,
            ).squeeze()

            x1_indices = get_augmented_mask(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x1_indices,
                num_basis_vectors=40,
                threshold=0.5,
            )
            x2_indices = get_augmented_mask(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x2_indices,
                num_basis_vectors=40,
                threshold=x2_random_threshold,
            )

            union_indices = (x1_indices + x2_indices).bool().float()

            complete_shape_areas = get_vertex_areas(
                self.vertex2faces,
                self.template_faces,
                self.dataset_name,
                identity_id,
                pose_id,
                relative_area=self.relative_area,
            )

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
            )
            x1_evals = get_mask_evals(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x1_indices,
                k=self.part_num_eigenvalues,
            )
            x2_evals = get_mask_evals(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x2_indices,
                k=self.part_num_eigenvalues,
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

        # augmentation
        else:
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
                )
                x2_indices = get_augmented_mask(
                    template_eigen=self.template_eigen,
                    dataset_name=self.dataset_name,
                    identity_id=identity_id,
                    pose_id=pose_id,
                    mask=x2_indices,
                    num_basis_vectors=x2_num_basis,
                    threshold=x2_threshold,
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
                )
                x2_indices = get_augmented_mask(
                    template_eigen=self.template_eigen,
                    dataset_name=self.dataset_name,
                    identity_id=identity_id,
                    pose_id=pose_id,
                    mask=x2_indices,
                    num_basis_vectors=self.default_augmentation_num_basis_x2,
                    threshold=self.default_augmentation_threshold_x2,
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
            )
            x1_evals = get_mask_evals(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x1_indices,
                k=self.part_num_eigenvalues,
            )
            x2_evals = get_mask_evals(
                template_eigen=self.template_eigen,
                dataset_name=self.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                mask=x2_indices,
                k=self.part_num_eigenvalues,
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

        if self.introduce_input_evals_noise:
            for t in ["X1_eigenvalues", "X2_eigenvalues"]:
                sample[t] = self.introudce_noise(sample[t])

        if self.evals_encoder is not None:
            for t in ["union_eigenvalues", "X1_eigenvalues", "X2_eigenvalues"]:
                sample[t] = self.evals_encoder(sample[t])

        return sample

    def introudce_noise(self, evals):
        return evals * (1 + torch.randn_like(evals, device=evals.device) * 1e-3)


def _generate_sample_cache(
    template_eigen,
    dataset_name,
    identity_id,
    pose_id,
    x1_indices,
    x2_indices,
    x1_num_basis,
    x1_threshold,
    x2_num_basis,
    x2_threshold,
    num_eigenvalues,
    sym_map,
):
    x1_indices = get_augmented_mask(
        template_eigen=template_eigen,
        dataset_name=dataset_name,
        identity_id=identity_id,
        pose_id=pose_id,
        mask=x1_indices,
        num_basis_vectors=x1_num_basis,
        threshold=x1_threshold,
    )

    x2_indices = get_augmented_mask(
        template_eigen=template_eigen,
        dataset_name=dataset_name,
        identity_id=identity_id,
        pose_id=pose_id,
        mask=x2_indices,
        num_basis_vectors=x2_num_basis,
        threshold=x2_threshold,
    )

    for x2_indices_sym in [x2_indices, x2_indices[sym_map]]:

        union_indices = (x1_indices + x2_indices_sym).bool().float()

        get_mask_evals(
            template_eigen=template_eigen,
            dataset_name=dataset_name,
            identity_id=identity_id,
            pose_id=pose_id,
            mask=x1_indices,
            k=num_eigenvalues,
        )
        get_mask_evals(
            template_eigen=template_eigen,
            dataset_name=dataset_name,
            identity_id=identity_id,
            pose_id=pose_id,
            mask=x2_indices_sym,
            k=num_eigenvalues,
        )
        get_mask_evals(
            template_eigen=template_eigen,
            dataset_name=dataset_name,
            identity_id=identity_id,
            pose_id=pose_id,
            mask=union_indices,
            k=num_eigenvalues,
        )


def _regenerate_cache():
    from spectral_unions.data.partial_dataset_v2 import PartialDatasetV2

    # todo: genera solo autovalori unione in cui i parametri di augmentation fra x1 e x2
    # sono uguali (threshold e num_basis)
    # config_name = "experiment.yml"
    # cfg = load_config(config_name)

    dataset_name = "PARTIAL_DATASET_V2_horses"
    datasplit = "train.txt"

    data = Path(get_env(dataset_name))
    split = Path(cfg["params"]["dataset"]["train_datasplit_folder"]) / datasplit
    split = "samples.txt"
    # split = Path(cfg["params"]["dataset"]["train_datasplit_folder"]) / "train.txt"
    trainset = (data / split).read_text().splitlines()
    # trainset = (data / datasplit).read_text().splitlines()

    dataset = PartialDatasetV2(cfg, dataset_name, trainset, return_mesh=False, no_evals_and_area=True)
    random_dataset = RandomAugmentedDataset(cfg, dataset_name, trainset)

    template_eigen = ShapeEigendecomposition(
        torch.from_numpy(dataset.template_vertices),
        torch.from_numpy(dataset.template_faces.astype("long")),
    )

    sym_map = torch.from_numpy(
        scipy.io.loadmat(dataset.dataset_root / "extras" / "SMPLsym.mat")["idxs"].squeeze().astype("long") - 1
    )

    for sample1 in tqdm(dataset):
        identity_id = sample1["identity_id"]
        pose_id = sample1["pose_id"]

        x1_id = sample1["id"]
        x1_indices = sample1["X1_indices"]
        for sample_folder_B, x2_random_threshold in random_dataset.intersection_classes[(str(Path(x1_id) / "A"), 0.5)]:
            x2_indices = multi_one_hot(
                load_mat(random_dataset.dataset_root / sample_folder_B, "indices.mat").astype(np.long),
                dataset.num_vertices,
            ).squeeze()
            _generate_sample_cache(
                template_eigen=template_eigen,
                dataset_name=dataset.dataset_name,
                identity_id=identity_id,
                pose_id=pose_id,
                x1_indices=x1_indices,
                x2_indices=x2_indices,
                x1_num_basis=40,
                x1_threshold=0.5,
                x2_num_basis=40,
                x2_threshold=x2_random_threshold,
                num_eigenvalues=random_dataset.union_num_eigenvalues,
                sym_map=sym_map,
            )


if __name__ == "__main__":
    seed_everything(0)
    _regenerate_cache()
    assert False
    config_name = "experiment.yml"

    seed_everything(0)
    cfg = load_config(config_name)

    dataset_name = "PARTIAL_DATASET_V2_horses"
    dataset_root = Path(get_env(dataset_name))

    split = Path(cfg["params"]["dataset"]["train_datasplit_folder"]) / "train.txt"
    sample_key_list = (dataset_root / split).read_text().splitlines()

    # get_eq_classes_uniq(dataset_name, "train.txt", 40, overwrite=True)

    # print(sample_folder_A)
    dataset = RandomAugmentedDataset(cfg, dataset_name, sample_key_list, augment=True, return_mesh=False)

    dataset[0]
    for sample in tqdm(dataset):
        x = sample["X1_indices"]
        y = sample["X2_indices"]
        z = sample["union_indices"]

        a = sample["union_eigenvalues"]
        b = sample["X1_eigenvalues"]
        c = sample["X2_eigenvalues"]
        k = 0
        pass
    # assert False
    # # #
    # #
    # # # loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    # # #
    # for x in tqdm(dataset):
    #     pass
    # #
    # # # todo Dataset[0] non prende da cache?
    # _generate_sample_cache(
    #     dataset.template_eigen,
    #     dataset.dataset_root,
    #     "shape000024",
    #     "pose0000235",
    #     dataset[0]["X1_indices"],
    #     dataset[0]["X2_indices"],
    #     num_basis=ShapeEigendecomposition.get_random_num_basis(),
    #     threshold=ShapeHamiltonianAugmenter.get_random_discretized_threshold(),
    #     num_eigenvalues=20,
    # )
    #
    # i = np.random.randint(0, len(dataset))
    # print(i)
    # i = 37344
    # sample = dataset[i]
    #
    # plot_shapes_comparison(
    #     meshes=[
    #         mesh(
    #             v=dataset.template_vertices.numpy(),
    #             f=dataset.template_faces.numpy(),
    #             color=sample["X1_indices"],
    #         ),
    #         mesh(
    #             v=dataset.template_vertices.numpy(),
    #             f=dataset.template_faces.numpy(),
    #             color=sample["X2_indices"],
    #         ),
    #         mesh(
    #             v=dataset.template_vertices.numpy(),
    #             f=dataset.template_faces.numpy(),
    #             color=sample["union_indices"],
    #         ),
    #     ],
    #     names=["x1", "x2", "union"],
    #     showscales=[False, False, False],
    # ).show()
    # assert False
    # ind = sample["union_indices"]
    #
    # print("next")
    # s = set()
    #
    # for i in tqdm(range(10000)):
    #
    #     x1_indices = get_augmented_mask(
    #         template_eigen=dataset.template_eigen,
    #         dataset_root=dataset.dataset_root,
    #         identity_id=sample["identity_id"],
    #         pose_id=sample["pose_id"],
    #         mask=sample["X1_indices"],
    #         num_basis_vectors=ShapeEigendecomposition.get_random_num_basis(),
    #         threshold=ShapeHamiltonianAugmenter.get_random_discretized_threshold(),
    #     )
    #     s.add(x1_indices)
    #     get_mask_evals(
    #         template_eigen=dataset.template_eigen,
    #         dataset_root=dataset.dataset_root,
    #         identity_id=sample["identity_id"],
    #         pose_id=sample["pose_id"],
    #         mask=sample["X1_indices"],
    #     )
    #
    # print(len(s))
    # #
    #
    # assert False
    # mat = scipy.io.loadmat("/home/luca/Desktop/test_3.mat")
    #
    # VERT = torch.from_numpy(mat["VERT"]).float().cpu()
    # TRIV = torch.from_numpy(mat["TRIV"]).long().cpu()
    # ind = torch.from_numpy(mat["ind"]).float().cpu()[0]

    # vertex2faces = VF_adjacency_matrix(VERT, TRIV)
    #
    # assert False

    # VERT = torch.from_numpy(dataset.template_vertices)
    # TRIV = torch.from_numpy(dataset.template_faces.astype("long"))
    # ind = dataset[2550]["union_indices"]
    # sym = (
    #     scipy.io.loadmat(
    #         "/home/luca/Repositories/partial-shape-generator/partial_dataset_v2/SMPLsym.mat"
    #     )["idxs"]
    #     .squeeze()
    #     .astype("long")
    # )
    # S = (
    #     torch.sparse_coo_tensor(
    #         indices=torch.stack(
    #             (
    #                 torch.arange(sym.shape[0]),
    #                 torch.from_numpy(sym) - 1,
    #             ),
    #             dim=0,
    #         ),
    #         values=torch.ones(sym.shape[0]),
    #         size=(VERT.shape[0], VERT.shape[0]),
    #     )
    #     .to_dense()
    #     .bool()
    #     .float()
    # )
    # sym_union_indices = torch.einsum("ij, bj -> bi", S, ind[None, :]).bool().float()[0]
    #
    # eigen = ShapeEigendecomposition(VERT, TRIV)
    # s = ShapeHamiltonianAugmenter(eigen, VERT)
    #
    # ks = [0 for i in range(1)]
    # plot_shapes_comparison(
    #     meshes=[
    #         mesh(
    #             v=s.vertices.numpy(),
    #             f=eigen.faces.numpy(),
    #             color=ind,
    #         ),
    #         mesh(
    #             v=s.vertices.numpy(),
    #             f=eigen.faces.numpy(),
    #             color=sym_union_indices,
    #         ),
    #     ],
    #     names=["sym1", "sym1"],
    #     showscales=[False, True],
    # ).show()
    #
    # assert False
    # import torch
    #
    # eigen = ShapeEigendecomposition(VERT, TRIV)
    # s = ShapeHamiltonianAugmenter(eigen, VERT)
    #
    # ks = [0 for i in range(1)]
    # plot_shapes_comparison(
    #     meshes=[
    #         mesh(
    #             v=s.vertices.numpy(),
    #             f=eigen.faces.numpy(),
    #             color=ind,
    #         ),
    #         mesh(
    #             v=s.vertices.numpy(),
    #             f=eigen.faces.numpy(),
    #             color=s.mask_random_augmentation(
    #                 ind,
    #                 eigen.get_random_num_basis(),
    #                 s.get_random_discretized_threshold(),
    #                 True,
    #             ),
    #         ),
    #     ]
    #     + [
    #         mesh(
    #             v=s.vertices.numpy(),
    #             f=eigen.faces.numpy(),
    #             color=s.mask_random_augmentation(
    #                 ind,
    #                 eigen.get_random_num_basis(),
    #                 s.get_random_discretized_threshold(),
    #             ).numpy(),
    #         )
    #         for x in ks
    #     ],
    #     names=["gournd truth", "projected"] + [f"_" for x in ks],
    #     showscales=[False, True] + [False] * len(ks),
    # ).show()
