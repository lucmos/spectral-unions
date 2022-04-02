import torch
from torch import nn


class EvalsEncodingV2(nn.Module):
    def __init__(self, num_evals, embed_dim):
        super(EvalsEncodingV2, self).__init__()
        assert embed_dim % 2 == 0

        self.num_evals = num_evals
        self.embed_dim = embed_dim
        self.half_embed_dim = embed_dim // 2
        self.embedder = nn.Embedding(num_embeddings=num_evals, embedding_dim=self.half_embed_dim)
        self.eval_proj = nn.Linear(in_features=1, out_features=self.half_embed_dim - 1)

    def forward(self, evals):
        # evals: [seq, batch]
        embeddings = self.embedder(torch.arange(self.num_evals, device=evals.device).repeat((evals.shape[1], 1)))
        embeddings = embeddings.permute(1, 0, 2)
        evals_embed = self.eval_proj(evals[..., None])
        # [seq, batch, embed evals + learnable embed + eval]
        output = torch.cat((embeddings, evals_embed, evals[..., None]), dim=-1)
        return output
