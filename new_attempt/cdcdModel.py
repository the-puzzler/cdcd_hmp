import torch
import torch.nn as nn
from timeEmbedding import TimeEmbedding
from selfConditioning import SelfConditioner
from noise import DiffusionNoise
from embedding import CDCDEmbedding
from timeNormedTB import TransformerBlock


class CDCDModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=1024,
        embedding_dim=256,
        num_layers=8,
        num_heads=8,
        time_embed_dim=128,
        t_min=1.0,
        t_max=300.0,
        pad_token_id=0,
        dropout=0.1
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
    

        # Core components we created
        self.embedding = CDCDEmbedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.noise = DiffusionNoise(t_min, t_max)
        self.self_conditioner = SelfConditioner()
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Project concatenated embeddings to hidden dim
        self.input_proj = nn.Linear(embedding_dim * 2, hidden_dim)  # *2 for concat of noised and self-cond
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, time_embed_dim, dropout = dropout)
            for _ in range(num_layers)
        ])
        
        # Project back to vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        

    
    def forward(self, x, timesteps=None, training=True, padding_mask=None):
        """
        Cannot be used for inference becuase will add noise baed on time step which is not wanted.
        """
        
        batch_size = x.shape[0]
        device = x.device
        
        if timesteps is None:
            timesteps = self.noise.sample_timesteps((batch_size,), device)
    

        # Get self-conditioning embeddings
        p_embeddings = self.self_conditioner.get_self_conditioning(
            x, self, timesteps, training, padding_mask
        )
     
        
        # Embed tokens and add noise
        #embeddings = self.embedding(x)
        embeddings = self.dropout(self.embedding(x))  #embedding dropout
        
        if padding_mask is not None:
            embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            p_embeddings = p_embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        x, _ = self.noise.add_noise(embeddings, p_embeddings, timesteps)
   
        
        #x = self.input_proj(x)
        x = self.dropout(self.input_proj(x)) #hidden dim dropout
        
        c_noise = torch.log(timesteps) / 4
        time_emb = self.time_embedding(c_noise)
     
        
    
        
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, time_emb, padding_mask)
            x = self.dropout(x) #Transformer dropout
        
        x = self.dropout(x) # Final projection dropout
        logits = self.output_proj(x)
       
        
        return logits
    
    @torch.no_grad()
    def sample(self, x, num_steps=200):
        """
        Sample from the model using Euler solver
        Args:
            x: initial tokens [batch_size, seq_len]
            num_steps: number of denoising steps
        Returns:
            tokens: [batch_size, seq_len]
        """
        device = x.device
        timesteps = torch.linspace(self.noise.t_max, self.noise.t_min, num_steps, device=device)
        
        # Initialize self-conditioner with zeros
        self.self_conditioner.prev_preds = None
        
        for t in timesteps:
            # Expand t for batch
            t_batch = t.expand(x.shape[0])
            
            # Get model prediction
            logits = self(x, t_batch, training=False)
            probs = torch.softmax(logits, dim=-1)
            
            # Update self-conditioner
            self.self_conditioner.update_prev_preds(probs)
            
            # Take argmax for next iteration
            x = torch.argmax(probs, dim=-1)
            
        return x
    
  