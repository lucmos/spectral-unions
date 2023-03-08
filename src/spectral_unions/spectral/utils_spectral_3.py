import scipy
import torch
from scipy.sparse.linalg import eigsh

USE_PYTORCH_SYMEIG = False  # faster but less stable


class Eigendecomposition(torch.autograd.Function):
    v0 = None

    # Note that both forward and backward are @staticmethods
    @classmethod
    def forward(cls, ctx, input_matrix, K, N):
        #         t = time.time()
        if USE_PYTORCH_SYMEIG:
            eigvals, eigvecs = torch.symeig(input_matrix, eigenvectors=True)
        else:
            input_matrix_np = input_matrix.data.cpu().numpy()
            Knp = K.data.numpy()

            # Full decomposition
            #             eigvals, eigvecs = eigh(input_matrix_np, lower=True)

            # Smalleste eigenvalues with full matrix
            #             eigvals, eigvecs = eigh(input_matrix_np, eigvals=(0, Knp - 1), lower=True)

            # Smallest eigenvalues with sparse matrix
            SM = scipy.sparse.csr_matrix(input_matrix_np)
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(SM, k=Knp, sigma=-1e-6, v0=cls.v0)

            #             cls.v0 = eigvecs[:,0]

            eigvals = torch.from_numpy(eigvals).cuda()
            eigvecs = torch.from_numpy(eigvecs).cuda()

        #         print(time.time()-t)
        ctx.save_for_backward(input_matrix, eigvals, eigvecs, K, N)
        return (eigvecs[:, :K], eigvals[:K])

    @staticmethod
    def backward(ctx, grad_output, grad_output2):
        # grad_output stands for the grad of eigvecs
        # grad_output2 stands for the grad of eigvals

        #         t = time.time()
        input_matrix, eigvals, eigvecs, K, N = ctx.saved_tensors

        Kknp = eigvals.shape[0]
        grad_K = None
        grad_N = None

        # NOTE: if we suffer memory issues we can split the K eigenvectors and iterate using an accumulator
        with torch.no_grad():
            grad_input_matrix = 0
            if grad_output.abs().sum() > 0:
                Ink = torch.eye(Kknp, K, dtype=eigvals.dtype).cuda()
                eig_inv = 1 / (eigvals[None, :K] - eigvals[:, None] + Ink) * (1 - Ink)
                uk = eigvecs[:, :K]
                grad_input_matrix = torch.mm(
                    torch.mm(
                        eigvecs,
                        eig_inv[:, :] * torch.mm(eigvecs.t(), grad_output[:, :K]),
                    ),
                    uk.t(),
                ).t()
            #         print('time -',time.time() - t)

            if grad_output2.abs().sum() > 0:
                for i in range(K):
                    grad_input_matrix += eigvecs[None, :, i] * eigvecs[:, None, i] * grad_output2[i]
        #                 grad_input_matrix += (eigvecs[None,:,:K]*eigvecs[:,None,:K]*grad_output2[None,None,:K]).sum(-1).T
        # df(l)/dX = df/dl * dl/dX
        # dl_i/dX -> evecs_i[] evecs_i'

        return grad_input_matrix, grad_K, grad_N


# Aliasing
# Eigendecomposition = Eigendecomposition.apply


# TODO: delete this
class EigendecompositionSparse(torch.autograd.Function):
    v0 = None

    # Note that both forward and backward are @staticmethods
    @classmethod
    def forward(cls, ctx, values, indices, K):
        device = values.device
        #         t = time.time()
        if USE_PYTORCH_SYMEIG:
            # fixme: broken
            # eigvals, eigvecs = torch.symeig(input_matrix, eigenvectors=True)
            pass
        else:
            # input_matrix_np = input_matrix.data.cpu().numpy()
            Knp = K.data.numpy()

            valuesnp = values.data.cpu().numpy()
            indicesnp = indices.data.cpu().numpy()

            SM = scipy.sparse.coo_matrix((valuesnp, indicesnp)).tocsc()

            eigvals, eigvecs = eigsh(SM, k=Knp, sigma=-1e-6, v0=cls.v0, which="LM")

            eigvals = torch.from_numpy(eigvals).to(device)
            eigvecs = torch.from_numpy(eigvecs).to(device)

        #         print(time.time()-t)
        ctx.save_for_backward(indices, eigvals, eigvecs, K)
        return (eigvecs[:, :K], eigvals[:K])

    @staticmethod
    def backward(ctx, grad_output, grad_output2):
        # grad_output stands for the grad of eigvecs
        # grad_output2 stands for the grad of eigvals

        #         t = time.time()
        indices, eigvals, eigvecs, K = ctx.saved_tensors

        Kknp = eigvals.shape[0]
        grad_K = None
        grad_N = None

        # NOTE: if we suffer memory issues we can split the K eigenvectors and iterate using an accumulator
        with torch.no_grad():
            grad_input_matrix = 0
            if grad_output.abs().sum() > 0:
                Ink = torch.eye(Kknp, K, dtype=eigvals.dtype).cuda()
                eig_inv = 1 / (eigvals[None, :K] - eigvals[:, None] + Ink) * (1 - Ink)
                uk = eigvecs[:, :K]
                grad_input_matrix = torch.mm(
                    torch.mm(
                        eigvecs,
                        eig_inv[:, :] * torch.mm(eigvecs.t(), grad_output[:, :K]),
                    ),
                    uk.t(),
                ).t()
                #         print('time -',time.time() - t)
                grad_input_matrix = grad_input_matrix[indices[0], indices[1]]

            if grad_output2.abs().sum() > 0:
                grad_input_matrix += (eigvecs[indices[0]] * grad_output2[None, :] * eigvecs[indices[1]]).sum(-1)
        return grad_input_matrix, None, grad_K, grad_N


# Aliasing
Eigendecomposition = EigendecompositionSparse.apply  # noqa


def LBO_slim(V, faces):
    """
    Input:
      V: B x N x 3
      faces: B x faces x 3
    Outputs:
      C: B x faces x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    """
    # tensor type and device
    device = V.device
    dtype = V.dtype

    indices_repeat = torch.stack([faces, faces, faces], dim=2)

    # v1 is the list of first triangles B*faces*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    l1 = torch.sqrt(((v2 - v3) ** 2).sum(2))  # distance of edge 2-3 for every face B*faces
    l2 = torch.sqrt(((v3 - v1) ** 2).sum(2))
    l3 = torch.sqrt(((v1 - v2) ** 2).sum(2))

    # semiperimieters
    # sp = (l1 + l2 + l3) * 0.5  # fixme: flake8 says unused

    # Heron's formula for area
    # A = torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3)).cuda()
    A = 0.5 * (torch.sum(torch.cross(v2 - v1, v3 - v2, dim=2) ** 2, dim=2) ** 0.5)  # VALIDATED

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l1**2 - l2**2 - l3**2) / (8 * A)
    cot31 = (l2**2 - l3**2 - l1**2) / (8 * A)
    cot12 = (l3**2 - l1**2 - l2**2) / (8 * A)

    batch_cot23 = cot23.view(-1)
    batch_cot31 = cot31.view(-1)
    batch_cot12 = cot12.view(-1)

    # proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    # C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 8 # dim: [B x faces x 3] cotangent of angle at vertex 1,2,3 correspondingly

    B = V.shape[0]
    num_vertices_full = V.shape[1]
    num_faces = faces.shape[1]

    edges_23 = faces[:, :, [1, 2]]
    edges_31 = faces[:, :, [2, 0]]
    edges_12 = faces[:, :, [0, 1]]

    batch_edges_23 = edges_23.view(-1, 2)
    batch_edges_31 = edges_31.view(-1, 2)
    batch_edges_12 = edges_12.view(-1, 2)

    W = torch.zeros(B, num_vertices_full, num_vertices_full, dtype=dtype, device=device)

    repeated_batch_idx_f = (
        torch.arange(0, B).repeat(num_faces).reshape(num_faces, B).transpose(1, 0).contiguous().view(-1)
    )  # [000...111...BBB...], number of repetitions is: num_faces
    repeated_batch_idx_v = (
        torch.arange(0, B).repeat(num_vertices_full).reshape(num_vertices_full, B).transpose(1, 0).contiguous().view(-1)
    )  # [000...111...BBB...], number of repetitions is: num_vertices_full
    repeated_vertex_idx_b = torch.arange(0, num_vertices_full).repeat(B)

    W[repeated_batch_idx_f, batch_edges_23[:, 0], batch_edges_23[:, 1]] = batch_cot23
    W[repeated_batch_idx_f, batch_edges_31[:, 0], batch_edges_31[:, 1]] = batch_cot31
    W[repeated_batch_idx_f, batch_edges_12[:, 0], batch_edges_12[:, 1]] = batch_cot12

    W = W + W.transpose(2, 1)

    batch_rows_sum_W = torch.sum(W, dim=1).view(-1)
    W[repeated_batch_idx_v, repeated_vertex_idx_b, repeated_vertex_idx_b] = -batch_rows_sum_W
    # W is the contangent matrix VALIDATED
    # Warning: residual error of torch.max(torch.sum(W,dim = 1).view(-1)) is ~ 1e-18

    VF_adj = VF_adjacency_matrix(V[0], faces[0]).unsqueeze(0).expand(B, num_vertices_full, num_faces)  # VALIDATED
    V_area = (torch.bmm(VF_adj, A.unsqueeze(2)) / 3).squeeze()  # VALIDATED

    return W, V_area


def LBO(V, faces):
    W, V_area = LBO_slim(V, faces)
    area_matrix = torch.diag_embed(V_area)
    area_matrix_inv = torch.diag_embed(1 / V_area)
    L = torch.bmm(area_matrix_inv, W)  # VALIDATED
    return L, area_matrix, area_matrix_inv, W


def VF_adjacency_matrix(V, faces):
    """
    Input:
    V: N x 3
    faces: faces x 3
    Outputs:
    C: V x faces adjacency matrix
    """
    # tensor type and device
    device = V.device
    dtype = V.dtype

    VF_adj = torch.zeros((V.shape[0], faces.shape[0]), dtype=dtype, device=device)
    v_idx = faces.view(-1)
    f_idx = (
        torch.arange(faces.shape[0]).repeat(3).reshape(3, faces.shape[0]).transpose(1, 0).contiguous().view(-1)
    )  # [000111...FFF]

    VF_adj[v_idx, f_idx] = 1
    return VF_adj


def calc_tri_areas(vertices, faces):
    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]

    v1 = v1 - v3
    v2 = v2 - v3
    return torch.norm(torch.cross(v1, v2, dim=1), dim=1) * 0.5


def calc_LB_FEM_sparse(vertices, faces, symmetric=True, device="cuda"):
    n = vertices.shape[0]
    # m = faces.shape[0]  # fixme: flake8 says unused

    angles = {}
    for i in (1.0, 2.0, 3.0):
        a = torch.fmod(torch.as_tensor(i - 1), torch.as_tensor(3.0)).long()
        b = torch.fmod(torch.as_tensor(i), torch.as_tensor(3.0)).long()
        c = torch.fmod(torch.as_tensor(i + 1), torch.as_tensor(3.0)).long()

        ab = vertices[faces[:, b], :] - vertices[faces[:, a], :]
        ac = vertices[faces[:, c], :] - vertices[faces[:, a], :]

        ab = torch.nn.functional.normalize(ab, p=2, dim=1)
        ac = torch.nn.functional.normalize(ac, p=2, dim=1)

        o = torch.mul(ab, ac)
        o = torch.sum(o, dim=1)
        o = torch.acos(o)
        o = torch.div(torch.cos(o), torch.sin(o))

        angles[i] = o

    indicesI = torch.cat((faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 1], faces[:, 0]))
    indicesJ = torch.cat((faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 1], faces[:, 0], faces[:, 2]))
    indices = torch.stack((indicesI, indicesJ))

    one_to_n = torch.arange(0, n, dtype=torch.long, device=device)
    eye_indices = torch.stack((one_to_n, one_to_n))

    values = torch.cat((angles[3], angles[1], angles[2], angles[1], angles[3], angles[2])) * 0.5

    stiff = torch.sparse_coo_tensor(
        indices=indices, dtype=values.dtype, values=-values, device=device, size=(n, n)
    ).coalesce()
    stiff = (
        stiff
        + torch.sparse_coo_tensor(
            indices=eye_indices,
            dtype=values.dtype,
            values=-torch.sparse.sum(stiff, dim=0).to_dense(),
            device=device,
            size=(n, n),
        ).coalesce()
    )

    areas = calc_tri_areas(vertices, faces)
    areas = areas.repeat(6) / 12.0

    mass = torch.sparse_coo_tensor(
        indices=indices, dtype=values.dtype, values=areas, device=device, size=(n, n)
    ).coalesce()
    mass = (
        mass
        + torch.sparse_coo_tensor(
            indices=eye_indices,
            dtype=values.dtype,
            values=torch.sparse.sum(mass, dim=0).to_dense(),
            device=device,
            size=(n, n),
        ).coalesce()
    )

    lumped_mass = torch.sparse.sum(mass, dim=1).to_dense()
    lower_inv = None
    if symmetric:
        lower_inv = lumped_mass.rsqrt()
        stiff = sparse_dense_mul(stiff, lower_inv[None, :] * lower_inv[:, None]).coalesce()  # <- INEFFICIENCY

    return stiff, lumped_mass, lower_inv, mass


def sparse_dense_mul(s, d):
    # implements point-wise product sparse * dense
    s = s.coalesce()
    i = s.indices()
    v = s.values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size()).coalesce()


# TODO: remove unused functions!
