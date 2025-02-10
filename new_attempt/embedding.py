import torch
import torch.nn as nn
import math

class CDCDEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=256, sigma=0.001):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Initialize raw embeddings - these will be allowed to vary freely
        self.raw_embedding = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim) * sigma
        )
    
    def normalize_embeddings(self, embeddings):
        """Normalize embeddings before use"""
        # Calculate scale factor based on current embedding dimension
        scale = math.sqrt(embeddings.size(-1))
        normalized = nn.functional.normalize(embeddings, p=2, dim=-1)
        return normalized * scale
    
    def forward(self, x):
        # Get embeddings and normalize before use
        emb = self.raw_embedding[x]
        return self.normalize_embeddings(emb)
    
    @property
    def normalized_embeddings(self):
        """Get the full normalized embedding matrix"""
        return self.normalize_embeddings(self.raw_embedding)