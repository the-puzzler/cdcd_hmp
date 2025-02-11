import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, time_embed_dim=128):
        super().__init__()
        self.norm1 = TimeConditionalLayerNorm(hidden_dim, time_embed_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm2 = TimeConditionalLayerNorm(hidden_dim, time_embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim, bias=False)
        )
        nn.init.zeros_(self.ffn[-1].weight)
        
    def forward(self, x, time_emb, padding_mask=None):
        # First norm and attention
        normed = self.norm1(x, time_emb, padding_mask)
  
        # Create attention mask from padding mask
        attn_mask = None
        if padding_mask is not None:
            # Change: reshape the mask for multi-head attention
            batch_size = padding_mask.size(0)
            # padding_mask: [batch_size, seq_len] -> attn_mask: [batch_size * num_heads, seq_len, seq_len]
            attn_mask = padding_mask.unsqueeze(1).expand(-1, padding_mask.size(1), -1)
            attn_mask = attn_mask.repeat_interleave(self.attn.num_heads, dim=0)
        
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=None,  # Change: remove key_padding_mask since we're using attn_mask
            attn_mask=attn_mask
        )
        
        # Apply residual connection, masking padded positions
        if padding_mask is not None:
            attn_out = attn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x + attn_out
        
        # Second norm and FFN
        normed = self.norm2(x, time_emb, padding_mask)
        ffn_out = self.ffn(normed)
        
        # Mask FFN output for padding positions
        if padding_mask is not None:
            ffn_out = ffn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x + ffn_out
        
        return x

class TimeConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_dim, time_embed_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.time_proj = nn.Linear(time_embed_dim, 2 * hidden_dim)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        
    def forward(self, x, time_emb, padding_mask=None):
        # Get time-conditional scale and shift
        time_proj = self.time_proj(time_emb)
        scale, shift = time_proj.chunk(2, dim=-1)
        
        # Normalize
        x = self.norm(x)
        
        # Apply conditional scale and shift
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Mask padded positions
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            
        return x