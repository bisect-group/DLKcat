import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedKcatPrediction(nn.Module):
    def __init__(
        self,
        n_fingerprint: int,
        n_word: int,
        dim: int,
        layer_gnn: int,
        window: int,
        layer_cnn: int,
        layer_output: int,
    ):
        super().__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=2 * window + 1,
                    stride=1,
                    padding=window,
                )
                for _ in range(layer_cnn)
            ]
        )
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 1)
        self.layer_gnn = int(layer_gnn)
        self.layer_cnn = int(layer_cnn)
        self.layer_output = int(layer_output)

    def gnn(self, xs, adjacency, atom_mask):
        mask = atom_mask.unsqueeze(-1)
        xs = xs * mask
        for layer in self.W_gnn:
            hs = torch.relu(layer(xs)) * mask
            xs = (xs + torch.bmm(adjacency, hs)) * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return xs.sum(dim=1) / denom

    def attention_cnn(self, compound_vector, word_vectors, word_mask):
        xs = word_vectors.unsqueeze(1)
        for layer in self.W_cnn:
            xs = torch.relu(layer(xs))
        xs = xs.squeeze(1) * word_mask.unsqueeze(-1)

        h = torch.relu(self.W_attention(compound_vector))
        hs = torch.relu(self.W_attention(xs)) * word_mask.unsqueeze(-1)
        weights = torch.tanh(torch.bmm(hs, h.unsqueeze(-1)).squeeze(-1))
        weights = weights * word_mask
        ys = weights.unsqueeze(-1) * hs
        denom = word_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return ys.sum(dim=1) / denom

    def forward(self, batch):
        fingerprint_vectors = self.embed_fingerprint(batch["fingerprints"])
        compound_vector = self.gnn(fingerprint_vectors, batch["adjacency"], batch["atom_mask"])

        word_vectors = self.embed_word(batch["words"]) * batch["word_mask"].unsqueeze(-1)
        protein_vector = self.attention_cnn(compound_vector, word_vectors, batch["word_mask"])

        cat_vector = torch.cat((compound_vector, protein_vector), dim=1)
        for layer in self.W_out:
            cat_vector = torch.relu(layer(cat_vector))
        return self.W_interaction(cat_vector).squeeze(-1)


def build_model(hparams, n_fingerprint: int, n_word: int) -> BatchedKcatPrediction:
    return BatchedKcatPrediction(
        n_fingerprint=n_fingerprint,
        n_word=n_word,
        dim=int(hparams["dim"]),
        layer_gnn=int(hparams["layer_gnn"]),
        window=int(hparams["window"]),
        layer_cnn=int(hparams["layer_cnn"]),
        layer_output=int(hparams["layer_output"]),
    )
