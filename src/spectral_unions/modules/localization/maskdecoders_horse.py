import torch
import torch.nn.functional as F
from torch import nn


class DecoderLayerNormDenseHorse(nn.Module):
    def __init__(
        self,
        hparams,
        hidden_dim1,
        hidden_dim2,
        hidden_dim3,
        p_dropout,
        last_sigmoid=False,
    ):
        super(DecoderLayerNormDenseHorse, self).__init__()

        evals_output = hparams["params"]["global"]["union_num_eigenvalues"]
        num_vertices = hparams["params"]["global"]["num_vertices"]

        self.p_dropout = p_dropout
        self.decoder1 = nn.Linear(evals_output, hidden_dim1)

        self.norm2 = nn.LayerNorm(hidden_dim1)
        self.decoder2 = nn.Linear(hidden_dim1, hidden_dim2)

        self.norm3 = nn.LayerNorm(hidden_dim2)
        self.decoder3 = nn.Linear(hidden_dim2, hidden_dim3)

        self.decoder4 = nn.Linear(hidden_dim3, num_vertices)

        self.last_sigmoid = last_sigmoid

        # todo: should be like this
        # not changing to mantain consistency with logged runs
        self.dropout1 = nn.Dropout(p=self.p_dropout)
        self.dropout2 = nn.Dropout(p=self.p_dropout)
        self.dropout3 = nn.Dropout(p=self.p_dropout)

    def forward(self, union_eigenvalues: torch.Tensor, **kwargs):
        """
        :param X1_eigenvalues: tensor with shape [batch_size, seq_len]
        :param X2_eigenvalues: tensor with shape [batch_size, seq_len]
        """

        output = F.relu(self.decoder1(union_eigenvalues))
        output = self.dropout1(F.elu(self.decoder2(self.norm2(output))))
        output = self.dropout2(F.elu(self.decoder3(self.norm3(output))))
        output = self.decoder4(output)

        if self.last_sigmoid:
            output = torch.sigmoid(output)

        return output
