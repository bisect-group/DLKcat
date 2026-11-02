import torch
import torch.nn as nn
import torch.nn.functional as F


class KcatPrediction(nn.Module):
    """The original DLKcat network (Code/model/run_model.py) for one sample: batch size 1, no padding or masks."""

    def __init__(self, n_fingerprint: int, n_word: int, dim: int, layer_gnn: int, window: int, layer_cnn: int, layer_output: int):
        super().__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * window + 1, stride=1, padding=window) for _ in range(layer_cnn)]
        )
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 1)

    def gnn(self, xs, adjacency):
        for layer in self.W_gnn:
            hs = torch.relu(layer(xs))
            xs = xs + torch.matmul(adjacency, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, compound_vector, xs):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for layer in self.W_cnn:
            xs = torch.relu(layer(xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        h = torch.relu(self.W_attention(compound_vector))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, fingerprints, adjacency, words):
        compound_vector = self.gnn(self.embed_fingerprint(fingerprints), adjacency)
        protein_vector = self.attention_cnn(compound_vector, self.embed_word(words))
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for layer in self.W_out:
            cat_vector = torch.relu(layer(cat_vector))
        return self.W_interaction(cat_vector)


def build_model(hparams, n_fingerprint: int, n_word: int) -> KcatPrediction:
    return KcatPrediction(
        n_fingerprint=n_fingerprint,
        n_word=n_word,
        dim=int(hparams["dim"]),
        layer_gnn=int(hparams["layer_gnn"]),
        window=int(hparams["window"]),
        layer_cnn=int(hparams["layer_cnn"]),
        layer_output=int(hparams["layer_output"]),
    )
