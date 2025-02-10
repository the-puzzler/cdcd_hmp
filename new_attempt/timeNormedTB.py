import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, time_embed_dim=128):
        super().__init__()
        # First time-conditional normalization
        self.norm1 = TimeConditionalLayerNorm(hidden_dim, time_embed_dim)
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Second time-conditional normalization
        self.norm2 = TimeConditionalLayerNorm(hidden_dim, time_embed_dim)
        
        # FFN with zero initialization for last layer
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=False)
        )
        nn.init.zeros_(self.ffn[-1].weight)
        
    def forward(self, x, time_emb):
        # First norm and attention
        normed = self.norm1(x, time_emb)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Second norm and FFN
        normed = self.norm2(x, time_emb)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x

class TimeConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_dim, time_embed_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # Project time embedding to shift and scale factors
        self.time_proj = nn.Linear(time_embed_dim, 2 * hidden_dim)
        # Zero initialize
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        
    def forward(self, x, time_emb):
        # Get time-conditional scale and shift
        time_proj = self.time_proj(time_emb)
        scale, shift = time_proj.chunk(2, dim=-1)
        
        # Normalize and apply conditional scale and shift
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x