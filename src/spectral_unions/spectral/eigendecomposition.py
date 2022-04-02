import random

import numpy as np
import scipy
import torch

from spectral_unions.spectral.utils_spectral_3 import Eigendecomposition, calc_LB_FEM_sparse


def compute_hamiltonian_evals(Mvert, Mtriv, optmask, k=10):
    with torch.no_grad():
        L_sym, A, Ainv, lumped = calc_LB_FEM_sparse(Mvert.double(), Mtriv, device="cpu")
        soptmask = 1 - optmask
        Adj = L_sym.bool().double()
        soptmask = ((Adj @ soptmask) > 0).double()
        P = torch.sparse_coo_tensor(
            indices=torch.arange(Mvert.shape[0])[None, :].repeat(2, 1),
            values=soptmask * 1e6,
        ).double()
        phi, lam = Eigendecomposition(
            (L_sym + P).coalesce().values(),
            L_sym.indices(),
            torch.tensor(k),
        )
        return phi.detach().cpu(), lam.detach().cpu()


class ShapeEigendecomposition:

    possible_num_basis_vectors = [
        40
    ]  # todo: cache generata con 40. Magari cambiamo a 30 perché più smoothe quindi threshold variano di più?

    @staticmethod
    def get_random_num_basis():
        return random.sample(ShapeEigendecomposition.possible_num_basis_vectors, 1)[0]

    def __init__(self, vertices, faces, k=30):
        with torch.no_grad():
            self.vertices = vertices.double()
            self.faces = faces
            assert self.faces.min().item() == 0
            (
                self.stiff,
                self.lumped_mass,
                self.lower_inv,
                self.mass,
            ) = calc_LB_FEM_sparse(self.vertices, self.faces, device=vertices.device)
            self.adj = self.stiff.bool().float()
            Mevecs, Mevals = Eigendecomposition(
                self.stiff.values(),
                self.stiff.indices(),
                torch.tensor(k),
            )
            self.evecs = Mevecs * self.lumped_mass.rsqrt()[:, None]

            self.vertices = self.vertices.float()
            self.stiff = self.stiff.float()
            self.lumped_mass = self.lumped_mass.float()
            self.lower_inv = self.lower_inv.float()
            self.mass = self.mass.float()
            self.evecs = self.evecs.float()

        self.projectors = {
            i: self.get_laplacian_basis_projector(self.evecs, self.lumped_mass, i)
            for i in ShapeEigendecomposition.possible_num_basis_vectors
        }

    def get_laplacian_basis_projector(self, evecs, lumped_mass, num_basis_vectors):
        phi = evecs[:, :num_basis_vectors]
        projector = phi @ (phi.t() * lumped_mass[None, :])
        return projector


class ShapeHamiltonianAugmenter:
    num_discretizations = 10

    thresholds = np.linspace(0.15, 0.85, num_discretizations).tolist()

    def __init__(
        self,
        template_eigendecomposition,
        vertices,
        device="cuda",
    ):
        self.template_eigendecomposition = template_eigendecomposition

        with torch.no_grad():
            self.vertices = vertices.double().to(device)
            (self.stiff, self.lumped_mass, self.lower_inv, self.mass,) = calc_LB_FEM_sparse(
                self.vertices.to(device),
                self.template_eigendecomposition.faces.to(device),
                device=device,
            )
            self.vertices = self.vertices.float().cpu()
            self.stiff = self.stiff.float().cpu()
            self.lumped_mass = self.lumped_mass.float().cpu()
            self.lower_inv = self.lower_inv.float().cpu()
            self.mass = self.mass.float().cpu()

    @staticmethod
    def get_random_discretized_threshold():
        return random.sample(ShapeHamiltonianAugmenter.thresholds, 1)[0]

    def get_hamiltonian_evals(self, mask, k):
        with torch.no_grad():
            soptmask = 1 - mask
            soptmask = ((self.template_eigendecomposition.adj @ soptmask) > 0).float()
            P = torch.sparse_coo_tensor(
                indices=torch.arange(self.vertices.shape[0])[None, :].repeat(2, 1),
                values=soptmask * 1e6,
            )
            m = (self.stiff + P).coalesce().double()
            phi, lam = Eigendecomposition(
                m.values(),
                m.indices(),
                torch.tensor(k),
            )
            return phi.detach().cpu().float(), lam.detach().cpu().float()

    def mask_random_augmentation(self, mask, num_basis_vectors, threshold, return_projected=False, device="cuda"):
        with torch.no_grad():
            if num_basis_vectors in self.template_eigendecomposition.projectors:
                projector = self.template_eigendecomposition.projectors[num_basis_vectors].to(device)
            else:
                projector = (
                    self.template_eigendecomposition.get_laplacian_basis_projector(
                        self.template_eigendecomposition.evecs,
                        self.template_eigendecomposition.lumped_mass,
                        num_basis_vectors,
                    )
                ).to(device)
            projected_mask = projector @ mask.float().to(device)

            if return_projected:
                return projected_mask.cpu()

            smoothed_mask = (projected_mask > threshold).float()

            return smoothed_mask.cpu()


if __name__ == "__main__":
    mat = scipy.io.loadmat("/home/luca/Desktop/test_3.mat")

    VERT = torch.from_numpy(mat["VERT"]).double().cpu()
    TRIV = torch.from_numpy(mat["TRIV"]).long().cpu()
    ind = torch.from_numpy(mat["ind"]).double().cpu()[0]

    evecs, evals = compute_hamiltonian_evals(VERT, TRIV, ind)
    print(evals[:14].data.cpu())

    a, e = ShapeEigendecomposition(VERT, TRIV).get_hamiltonian_evals(ind, 10)
    print(e)

    # e, phi = lin.eigsh(
    #     L_sym,
    #     M=A,
    #     k=20,
    #     sigma=-1e-5,
    # )
    # # e1, phi1 = Eigsh_torch.apply(C.to_dense(), C.values(), C.indices())
    # print(e[:21])
    # myscene = dict(
    #     camera=dict(
    #         up=dict(x=0, y=1, z=0),
    #         center=dict(x=0, y=0, z=0),
    #         eye=dict(x=-0.25, y=0.25, z=2.75),
    #     ),
    #     aspectmode="data",
    # )
    # fig = make_subplots(
    #     rows=1,
    #     cols=3,
    #     specs=[[{"is_3d": True}] * 3],
    #     subplot_titles=["partiality", "complete phi", "phi partiality"],
    #     horizontal_spacing=0,
    #     vertical_spacing=0,
    # )
    # import plotly.graph_objects as go
    #
    # vertices, faces = v.numpy().astype(np.float64), f.numpy().astype(np.uint32)
    #
    # for k in range(100):
    #     fig.add_trace(
    #         go.Mesh3d(
    #             x=vertices[:, 0],
    #             y=vertices[:, 1],
    #             z=vertices[:, 2],
    #             i=faces[:, 0],
    #             j=faces[:, 1],
    #             k=faces[:, 2],
    #             colorscale="Viridis",
    #             opacity=1,
    #             intensity=i,
    #             showscale=False,
    #         ),
    #         row=1,
    #         col=1,
    #     )
    #
    #     fig.add_trace(
    #         go.Mesh3d(
    #             x=vertices[:, 0],
    #             y=vertices[:, 1],
    #             z=vertices[:, 2],
    #             i=faces[:, 0],
    #             j=faces[:, 1],
    #             k=faces[:, 2],
    #             colorscale="Viridis",
    #             opacity=1,
    #             intensity=phi1[:, k],
    #             cmid=0,
    #             showscale=True,
    #         ),
    #         row=1,
    #         col=2,
    #     )
    #     fig.add_trace(
    #         go.Mesh3d(
    #             x=vertices[:, 0],
    #             y=vertices[:, 1],
    #             z=vertices[:, 2],
    #             i=faces[:, 0],
    #             j=faces[:, 1],
    #             k=faces[:, 2],
    #             colorscale="Viridis",
    #             opacity=1,
    #             intensity=phi2[:, k],
    #             cauto=True,
    #             cmid=0,
    #             showscale=True,
    #         ),
    #         row=1,
    #         col=3,
    #     )
    #     fig["layout"][f"scene{1}"].update(myscene)
    #     fig["layout"][f"scene{2}"].update(myscene)
    #     fig["layout"][f"scene{3}"].update(myscene)
    #     fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    #     fig.show()
    #     print(k)
    #     input()
