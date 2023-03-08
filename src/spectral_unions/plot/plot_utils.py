from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy import io

from nn_core.common.utils import get_env

from spectral_unions.data.dataset_utils import load_mat, multi_one_hot
from spectral_unions.data.datastructures import Mesh

# fixme: broken
load_config = ...


def default_layout(fig: go.Figure) -> None:
    """
    Set the default camera parameters for the plotly Mesh3D

    :param fig: the figure to adjust
    :return: the adjusted figure
    """
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-0.25, y=0.25, z=2),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        # title='ex',
        scene_aspectmode="auto",
    )

    return fig


def plot_mesh(m: Mesh, showscale, **kwargs) -> go.Mesh3d:
    """
    Plot the mesh in a plotly graph object

    :param m: the mesh to plot
    :param kwargs: possibly additional parameters for the go.Mesh3D class
    :return: the plotted mesh
    """
    vertices = m.v.astype(np.float64)
    if m.f is not None:

        faces = m.f.astype(np.uint32)
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            colorscale="Viridis",
            opacity=1,
            intensity=m.mask,
            showscale=showscale,
            **kwargs,
        )
    else:
        return go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
        )


def plot_shapes_comparison(meshes, names=None, showscales=None):
    myscene = dict(
        camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-0.25, y=0.25, z=2.75),
        ),
        aspectmode="data",
    )
    fig = make_subplots(
        rows=1,
        cols=len(meshes),
        specs=[[{"is_3d": True}] * len(meshes)],
        subplot_titles=names,
        horizontal_spacing=0,
        vertical_spacing=0,
    )

    for i, mesh in enumerate(meshes):
        fig.add_trace(
            plot_mesh(
                mesh,
                showscale=showscales[i] if showscales is not None else None,
                scene=f"scene{i+1}",
            ),
            row=1,
            col=i + 1,
        )
    for i in range(len(meshes)):
        fig["layout"][f"scene{i+1}"].update(myscene)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


def plot_sample(path):

    config_name = "experiment.yml"

    cfg = load_config(config_name)
    dataset_name = cfg["params"]["dataset"]["name"]

    dataset_folder = Path(get_env(dataset_name))
    template_vertices = torch.from_numpy(load_mat(dataset_folder / "extras", "VERT.mat"))
    template_faces = torch.from_numpy(load_mat(dataset_folder / "extras", "TRIV.mat").astype("long")) - 1
    x1 = multi_one_hot(
        torch.from_numpy(io.loadmat(dataset_folder / path / "A" / "indices.mat")["value"].astype("long")),
        num_classes=6890,
    )
    x2 = multi_one_hot(
        torch.from_numpy(io.loadmat(dataset_folder / path / "B" / "indices.mat")["value"].astype("long")),
        num_classes=6890,
    )
    union = multi_one_hot(
        torch.from_numpy(io.loadmat(dataset_folder / path / "indices.mat")["value"].astype("long")),
        num_classes=6890,
    )

    plot_shapes_comparison(
        meshes=[
            Mesh(
                v=template_vertices.numpy(),
                f=template_faces.numpy(),
                mask=x1,
            ),
            Mesh(
                v=template_vertices.numpy(),
                f=template_faces.numpy(),
                mask=x2,
            ),
            Mesh(
                v=template_vertices.numpy(),
                f=template_faces.numpy(),
                mask=union,
            ),
        ],
        names=["X1", "X2", "Union"],
    ).show()


def plot_by_name(paths):

    config_name = "experiment.yml"

    cfg = load_config(config_name)
    dataset_name = cfg["params"]["dataset"]["name"]

    dataset_folder = Path(get_env(dataset_name))
    template_vertices = torch.from_numpy(load_mat(dataset_folder / "extras", "VERT.mat"))
    template_faces = torch.from_numpy(load_mat(dataset_folder / "extras", "TRIV.mat").astype("long")) - 1
    if isinstance(paths, str) or isinstance(paths, Path):
        paths = [paths]

    inds = [
        multi_one_hot(
            torch.from_numpy(io.loadmat(dataset_folder / z / "indices.mat")["value"].astype("long")),
            num_classes=6890,
        )
        for z in paths
    ]

    plot_shapes_comparison(
        meshes=[
            Mesh(
                v=template_vertices.numpy(),
                f=template_faces.numpy(),
                mask=ind,
            )
            for ind in inds
        ],
        names=[str(b) for b in paths],
    ).show()


def plot_by_indices(indices, vertices=None, faces=None, autoshow=True):
    if vertices is None or faces is None:

        config_name = "experiment.yml"

        # fixme: broken
        cfg = load_config(config_name)

        dataset_name = cfg["params"]["dataset"]["name"]

        dataset_folder = Path(get_env(dataset_name))
        vertices = torch.from_numpy(load_mat(dataset_folder / "extras", "VERT.mat"))
        faces = torch.from_numpy(load_mat(dataset_folder / "extras", "TRIV.mat").astype("long") - 1)

    if not (isinstance(indices, list) or isinstance(indices, tuple)):
        indices = [indices]

    fig = plot_shapes_comparison(
        meshes=[
            Mesh(
                v=vertices.numpy(),
                f=faces.numpy(),
                mask=ind,
            )
            for ind in indices
        ],
    )

    if autoshow:
        fig.show()

    return fig


def plot(vertices, faces, autoshow=True):

    fig = plot_shapes_comparison(
        meshes=[Mesh(v=vertices.numpy(), f=faces.numpy(), mask=vertices.numpy()[:, 0])],
    )

    if autoshow:
        fig.show()

    return fig


def plot_eigs(eigs: np.ndarray, **kwargs) -> go.Scatter:
    """
    Plot the eigenvalues in a plotly graph object

    :param eigs: the eigenvalues to plot
    :param kwargs: possibly additional parameters for the  go.Scatter class
    :return: the plotted eigenvalues
    """
    return go.Scatter(x=np.arange(len(eigs)), y=eigs, mode="lines+markers", **kwargs)


def plot_evals_comparison(
    pred_values,
    union_values,
    x1_values,
    x2_values,
    plot_height=800,
):
    fig = go.Figure()

    fig.add_trace(plot_eigs(pred_values, name="Union predicted eigenvalues"))
    fig.add_trace(plot_eigs(union_values, name="Union ground-truth eigenvalues"))
    fig.add_trace(plot_eigs(x1_values, name="X1 eigenvalues"))
    fig.add_trace(plot_eigs(x2_values, name="X2 eigenvalues"))

    abs_errors = np.abs(pred_values - union_values)
    relative_errors = abs_errors / np.clip(np.abs(union_values), a_min=1e-6, a_max=None)
    relative_errors[0] = 0  # first eigenvalue is always zero
    relative_errors_perc = relative_errors * 100

    fig.add_trace(
        go.Bar(
            name="Relative error",
            x=np.arange(len(pred_values)),
            customdata=np.column_stack((relative_errors_perc, abs_errors)),
            hovertemplate="<b>relative: %{customdata[0]:.2f}%<br>abs: " "%{customdata[1]:.3f}</b><br>eigenvalue: %{x}",
            y=relative_errors,
            marker_color="rgba(100, 100, 100, 0.2)",
        )
    )

    fig.update_layout(
        autosize=True,
        # width=800,
        height=plot_height,
        margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
        # paper_bgcolor="LightSteelBlue",
    )
    fig.update_yaxes(automargin=True)
    return fig
