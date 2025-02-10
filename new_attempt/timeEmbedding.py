import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
    
    def random_fourier_features(self, t):
        # Generate random fourier features for timestep
        half_dim = self.dim // 2
        freqs = torch.exp(-torch.linspace(0, math.log(10000), half_dim))
        args = t.unsqueeze(-1) * freqs.to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
    
    def forward(self, t):
        fourier_emb = self.random_fourier_features(t)
        return self.mlp(fourier_emb)