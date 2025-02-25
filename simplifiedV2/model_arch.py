import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

class simplifiedV2(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        transformer_dim,
        num_layers,
        num_heads,
        dropout=0.1,
        embedding_scale=0.01,
        hidden_dim=None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.transformer_dim = transformer_dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        with torch.no_grad():
            self.token_embedding.weight.data.normal_(0, embedding_scale)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(dim=embedding_dim)
        
        # Embedding projection if needed
        self.embedding_projection = (
            nn.Linear(embedding_dim*2, transformer_dim) 
            if embedding_dim*2 != transformer_dim 
            else nn.Identity()
        )
        
        # Custom transformer encoder layer with GELU
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim or 4*transformer_dim,
            dropout=dropout,
            activation=nn.GELU(),
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Final projection layer
        self.final_projection = nn.Linear(transformer_dim, vocab_size)
        
    def forward(self, x, t, padding_mask=None):  # padding_mask: 1 for real tokens, 0 for padding
        """
        Args:
            x: Input tokens (batchsize, seq_length)
            t: Timesteps (batchsize)
            padding_mask: Binary mask (batchsize, seq_length) - 1 for real tokens, 0 for padding
        """
        if padding_mask is None:
            padding_mask = torch.ones_like(x, dtype=torch.bool)
        
        # Create attention mask for transformer (1 = masked/ignore, 0 = attend to)
        # confusing inversion... but docs say 1 means do IGNORE 0 means DONT ignore.
        attention_mask = ~padding_mask
        
        # Get initial token embeddings
        token_emb = self.token_embedding(x)
        
        # L2 normalize and scale by sqrt(dim)
        # Only normalize non-padding tokens
        norm = torch.norm(token_emb, p=2, dim=-1, keepdim=True)
        normalized_emb = torch.where(
            padding_mask.unsqueeze(-1),  # expand mask to embedding dim
            token_emb / (norm + 1e-8),
            token_emb
        )
        scaled_emb = normalized_emb * math.sqrt(self.embedding_dim)
        
        # Add scaled noise based on time (only to non-padding tokens)
        t = t.view(-1, 1, 1)
        noise = torch.randn_like(scaled_emb)
        noised_emb = torch.where(
            padding_mask.unsqueeze(-1),
            scaled_emb + t * noise, #square root here?
            scaled_emb
        )
        
        # Divide by sqrt(t^2 + 1)
        t_squared = t ** 2
        normalized_noised_emb = torch.where(
            padding_mask.unsqueeze(-1),
            noised_emb / torch.sqrt(t_squared + 1),
            noised_emb
        )
        normalized_noised_emb = noised_emb #skip noise
        
        # Process time for time embedding
        log_t = torch.log(t.squeeze(-1).squeeze(-1) + 1e-8) / 4
        time_emb = self.time_embedding(log_t)

        #Trying this.
        # L2 normalize and scale time embeddings to match token embeddings scale
        time_norm = torch.norm(time_emb, p=2, dim=-1, keepdim=True)
        time_emb = (time_emb / (time_norm + 1e-8)) * math.sqrt(self.embedding_dim)
                
        # # Add time embedding (only to non-padding tokens)
        # embedded = torch.where(
        #     padding_mask.unsqueeze(-1),
        #     normalized_noised_emb + time_emb.unsqueeze(1),
        #     normalized_noised_emb
        # )

        expanded_time_emb = time_emb.unsqueeze(1).expand(-1, normalized_noised_emb.size(1), -1)

        # Only apply to non-padding tokens
        expanded_time_emb = torch.where(
            padding_mask.unsqueeze(-1),
            expanded_time_emb,
            torch.zeros_like(expanded_time_emb)
        )

        # Concatenate along the embedding dimension (last dimension)
        embedded = torch.cat([normalized_noised_emb, expanded_time_emb], dim=-1)
        
        # Project to transformer dimension if needed
        embedded = self.embedding_projection(embedded)
        
        # Pass through transformer with attention mask
        transformed = self.transformer(
            embedded,
            src_key_padding_mask=attention_mask  # depends on pytorch version, check docs
        )
        
        # Final projection to vocab size
        output = self.final_projection(transformed)
        
        return output
    
    