import math

import torch.nn.functional as F
from torch import nn

from spectral_unions.modules.union.evals_encoding import EvalsEncodingV2
from spectral_unions.modules.union.positional_encoding import PositionalEncoding
from spectral_unions.pl_modules.union.parent_model_v2 import ParentModelV2


class UnionTransformerV6(ParentModelV2):
    def __init__(
        self,
        hparams,
        evals_embedding_dim=32,
        encoder_nhead=8,
        encoder_dim_feedforward=2048,
        encoder_nlayers=6,
        encoder_dropout=0.1,
        decoder_nhead=8,
        decoder_dim_feedforward=2048,
        decoder_nlayers=6,
        decoder_dropout=0.1,
        last_relu=False,
    ):
        super().__init__(hparams)
        evals_input = hparams["params"]["global"]["part_num_eigenvalues"]
        # evals_output = hparams["params"]["global"]["union_num_eigenvalues"]

        self.evals_embedding_dim = evals_embedding_dim

        self.evals_embedder = EvalsEncodingV2(evals_input, evals_embedding_dim)
        self.pos_encoder = PositionalEncoding(evals_embedding_dim, encoder_dropout)

        self.d_model = evals_embedding_dim
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=encoder_nhead,
            num_encoder_layers=encoder_nlayers,
            num_decoder_layers=encoder_nlayers,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
        )

        self.union_transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=decoder_nhead,
            num_encoder_layers=decoder_nlayers,
            num_decoder_layers=decoder_nlayers,
            dim_feedforward=decoder_dim_feedforward,
            dropout=decoder_dropout,
        )
        self.final_layer = nn.Linear(in_features=self.d_model, out_features=1)
        self.last_relu = last_relu

    def forward(self, X1_eigenvalues, X2_eigenvalues):
        X1_eigenvalues = X1_eigenvalues.permute(1, 0)
        X2_eigenvalues = X2_eigenvalues.permute(1, 0)

        x2_emb = self.evals_embedder(X2_eigenvalues)
        x2_emb = self.pos_encoder(x2_emb) * math.sqrt(self.evals_embedding_dim)

        x1_emb = self.evals_embedder(X1_eigenvalues)
        x1_emb = self.pos_encoder(x1_emb) * math.sqrt(self.evals_embedding_dim)

        union_emb1 = self.transformer(src=x1_emb, tgt=x2_emb)
        union_emb2 = self.transformer(src=x2_emb, tgt=x1_emb)

        union_emb = union_emb1 + union_emb2
        union_emb = self.union_transformer(src=union_emb, tgt=union_emb)
        output = self.final_layer(union_emb).squeeze(dim=-1).permute(1, 0)

        if self.last_relu:
            output = F.relu(output)

        return output
