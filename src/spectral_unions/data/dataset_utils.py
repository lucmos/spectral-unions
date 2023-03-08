import abc
import functools
import glob
import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import scipy.io as io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import get_env

from spectral_unions.spectral.eigendecomposition import ShapeHamiltonianAugmenter
from spectral_unions.spectral.utils_spectral_3 import calc_tri_areas

precision = "float"
device = "cpu"
cache_path = PROJECT_ROOT / "data" / "cache" / "augmented_dataset_v2_cache"  # TODO: check it is ok


@functools.lru_cache(maxsize=None)
def get_shape_augmenter(template_eigen, dataset_name: str, identity_id: str, pose_id: str, device: str):
    dataset_root = Path(get_env(dataset_name))
    vertices = torch.from_numpy(load_mat(dataset_root / identity_id / pose_id, "VERT.mat"))
    return ShapeHamiltonianAugmenter(template_eigen, vertices, device=device)


def get_augmented_mask(
    template_eigen,
    dataset_name: str,
    identity_id: str,
    pose_id: str,
    mask,
    num_basis_vectors,
    threshold,
    device,
):
    original_dataset_name = dataset_name

    if dataset_name == "PARTIAL_DATASET_V2":
        dataset_name = dataset_name.lower()
        # fix to re-use old cache with new naming schema that support multiple val datsets

    maskname = hashlib.md5(mask.numpy().tobytes()).hexdigest()
    filename = f"{dataset_name}_{identity_id}_{pose_id}_{maskname}_basis{num_basis_vectors}_thresh{threshold}.pickle"
    filepath = cache_path / "mask" / maskname[:2] / filename

    if filepath.exists():
        with filepath.open("rb") as f:
            return pickle.load(f)
    else:
        # assert False, filepath

        augmenter = get_shape_augmenter(template_eigen, original_dataset_name, identity_id, pose_id, device=device)
        augmented_mask = augmenter.mask_random_augmentation(
            mask, num_basis_vectors=num_basis_vectors, threshold=threshold, device=device
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
            pickle.dump(augmented_mask, f)
            return augmented_mask


def get_mask_evals(template_eigen, dataset_name: str, identity_id: str, pose_id: str, mask, device, k=20):
    original_dataset_name = dataset_name
    if dataset_name == "PARTIAL_DATASET_V2":
        dataset_name = dataset_name.lower()
        # fix to re-use old cache with new naming schema that support multiple val datsets

    maskname = hashlib.md5(mask.numpy().tobytes()).hexdigest()
    filename = f"{dataset_name}_{identity_id}_{pose_id}_{k}_{maskname}.pickle"
    filepath = cache_path / "evals" / maskname[:2] / filename

    if filepath.exists():
        with filepath.open("rb") as f:
            evals = pickle.load(f)
    else:
        # assert False, filepath

        augmenter = get_shape_augmenter(template_eigen, original_dataset_name, identity_id, pose_id, device=device)
        _, evals = augmenter.get_hamiltonian_evals(mask, k=k)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
            pickle.dump(evals, f)
    return evals


def get_vertex_areas(vertex2faces, faces, dataset_name, identity_id, pose_id, relative_area=True):
    original_dataset_name = dataset_name
    if dataset_name == "PARTIAL_DATASET_V2":
        dataset_name = dataset_name.lower()
        # fix to re-use old cache with new naming schema that support multiple val datsets

    filename = f"{dataset_name}_{identity_id}_{pose_id}_{relative_area}.pickle"
    filepath = cache_path / "areas" / filename

    if filepath.exists():
        with filepath.open("rb") as f:
            return pickle.load(f)
    else:
        # assert False, filepath
        dataset_root: Path = Path(get_env(original_dataset_name))
        union_vertices = torch.from_numpy(load_mat(dataset_root / identity_id / pose_id, "VERT.mat")).float()
        areas = calc_tri_areas(union_vertices, faces)
        areas = (vertex2faces @ areas) / 3

        if relative_area:
            areas = areas / areas.sum()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as f:
            pickle.dump(areas, f)
        return areas


@functools.lru_cache(maxsize=None)
def load_mat(sample_name: Path, mat_name: str) -> np.ndarray:
    filename = sample_name / mat_name
    assert filename.exists(), filename
    m = io.loadmat(filename)
    return m["value"]


def read_dataset_path(dataset_name: str) -> Path:
    """
    Read the system specific dataset path,
    given the dataset name specified in the current configuration

    :param dataset_name: dataset name to use
    :return: the system-specific path of the desired dataset
    """
    return Path(get_env(dataset_name))


def get_relativepath_wrt_dataset(dataset_name: str, database_type: str, full_path: str) -> str:
    """
    Given a full path inside the dataset,
    return the portion of the path starting from the dataset root.

    In this way it is possible to store sample's paths in a system-independent way.

    :param cfg: the current configuration
    :param full_path: the full path of a file inside the system-specific dataset root
    :return: the system-independent path of the file starting from the dataset root
    """
    if database_type == "filesystem":
        return str(Path(full_path).relative_to(read_dataset_path(dataset_name)))
    elif database_type == "leveldb":
        return full_path
    else:
        raise TypeError(f"Unknown database type: {database_type}")


def get_fullpath_wrt_dataset(dataset_name: str, database_type: str, relative: str) -> str:
    """
    Given a system-independent relative path, that starts from the dataset root,
    builds its full path using the system-specific dataset path

    :param cfg: the current configuration
    :param relative: the system-independent path of the file starting from dataset root
    :return: the full path of a file inside the system-specific dataset root
    """
    if database_type == "filesystem":
        return os.path.join(read_dataset_path(dataset_name), relative)
    elif database_type == "leveldb":
        return relative
    else:
        raise TypeError(f"Unknown database type: {database_type}")


def get_sample_name_list(dataset_root: Path, database_type: str, file_pattern: str = "**/*.mat") -> List[str]:
    if database_type == "filesystem":
        return sorted(glob.glob(os.path.join(dataset_root, file_pattern), recursive=True))
    elif database_type == "leveldb":
        if not Path(dataset_root).exists():
            raise FileNotFoundError(f"No such database: {dataset_root}")

        assert False
        # TODO: remove leveldb dependency
        # db = LevelDB(dataset_root, read_only=True)
        # names = list(db.prefixed_iter(prefixes=["Union", "evals"], include_value=False))
        # return names
    else:
        raise TypeError(f"Unknown database type: {database_type}")


def split_train_validation_mesh_files(
    dataset_name: str,
    database_type: str,
    train_size: Optional[int],
    validation_size: Optional[int],
) -> (Sequence[str], Sequence[str]):
    """
    Reads all the files (path only) that match the pattern, recursively from the dataset root.
    Then, performs a train-test split according to the sizes specified in the configuration file

    :param cfg: the configuration dictionary
    :param pattern: consider all the files that match this pattern

    :return: the (train, test) file paths
    """
    assert (
        train_size is None or validation_size is None or (train_size + validation_size == 1)
    ), f"The train_size plus test_size should be one (not {train_size + validation_size})"

    mesh_file_list = get_sample_name_list(read_dataset_path(dataset_name), database_type)

    train_files, validation_files = train_test_split(mesh_file_list, train_size=train_size, test_size=validation_size)

    return train_files, validation_files


class CustomDataset(Dataset):
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {len(self)} elements>"

    @staticmethod
    def get_collate_fn() -> Optional[Callable[[Any], Any]]:
        return None

    @abc.abstractmethod
    def get_file_list(self) -> List[Union[str, Path]]:
        pass


class CustomDataLoader(DataLoader):
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {len(self)} elements>"


def multi_one_hot(targets, num_classes, one_index=True):
    if targets.min() == 0 and one_index:
        raise RuntimeError

    onehot = torch.zeros(num_classes)
    onehot[targets - int(one_index)] = 1
    return onehot
