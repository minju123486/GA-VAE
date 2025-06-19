import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e          # Number of embeddings
        self.e_dim = e_dim      # Dimension of each embedding vector
        self.beta = beta        # Commitment loss coefficient

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        z.shape: (B, C, H, W)
        Returns:
        - loss
        - z_q: quantized output
        - perplexity
        - min_encodings: one-hot for closest token
        - min_encoding_indices: index of closest token (ei)
        - second_encoding_indices: index of 2nd closest token (ej)
        """

        # Step 1: reshape input
        z = z.permute(0, 2, 3, 1).contiguous()     # (B, H, W, C)
        z_flattened = z.view(-1, self.e_dim)       # (B*H*W, C)

        # Step 2: compute distances to all embeddings
        # (z - e)^2 = z^2 + e^2 - 2ze^T
        z_sq = torch.sum(z_flattened ** 2, dim=1, keepdim=True)       # (N, 1)
        e_sq = torch.sum(self.embedding.weight ** 2, dim=1)           # (K,)
        ze = torch.matmul(z_flattened, self.embedding.weight.t())     # (N, K)
        dists = z_sq + e_sq - 2 * ze                                   # (N, K)

        # Step 3: Top-2 nearest embeddings
        sorted_indices = torch.topk(dists, k=2, dim=1, largest=False)[1]  # (N, 2)
        min_encoding_indices = sorted_indices[:, 0].unsqueeze(1)  # ei (N, 1)
        second_encoding_indices = sorted_indices[:, 1].unsqueeze(1)  # ej (N, 1)

        # Step 4: One-hot encode for the closest
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Step 5: quantize
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Step 6: loss
        loss = torch.mean((z_q.detach() - z) ** 2) + \
               self.beta * torch.mean((z_q - z.detach()) ** 2)

        # Step 7: straight-through estimator
        z_q = z + (z_q - z).detach()

        # Step 8: perplexity
        avg_probs = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Step 9: reshape to original format
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return loss, z_q, perplexity, min_encodings, min_encoding_indices, second_encoding_indices
