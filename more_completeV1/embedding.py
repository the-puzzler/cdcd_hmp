import torch
import torch.nn as nn
import math

class CDCDEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=256, sigma=0.001, padding_idx=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Initialize raw embeddings - these will be allowed to vary freely
        self.raw_embedding = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim) * sigma
        )

        self.padding_idx = padding_idx

        # Initialize padding embedding to zeros if padding_idx is specified
        if self.padding_idx is not None:
            with torch.no_grad():
                self.raw_embedding[padding_idx].fill_(0)
    
    def normalize_embeddings(self, embeddings):
        """Normalize embeddings before use"""
        # Calculate scale factor based on current embedding dimension
        scale = math.sqrt(embeddings.size(-1))
        
        # Don't normalize padding tokens if they exist
        if self.padding_idx is not None:
            padding_mask = ~(torch.arange(embeddings.size(0)) == self.padding_idx)
            embeddings = embeddings.clone()
            embeddings[padding_mask] = nn.functional.normalize(
                embeddings[padding_mask], p=2, dim=-1
            )
        else:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
            
        return embeddings * scale
    
    def forward(self, x):
        # Get embeddings and normalize before use
        emb = self.raw_embedding[x.long()]
        return self.normalize_embeddings(emb)
    
    @property
    def normalized_embeddings(self):
        """Get the full normalized embedding matrix"""
        return self.normalize_embeddings(self.raw_embedding)