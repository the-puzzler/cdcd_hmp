import torch
import torch.nn as nn

class SelfConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_preds = None
    
    def get_self_conditioning(self, x, model, timesteps, training=True, padding_mask=None):
        """Get self-conditioning embeddings following Chen et al. 2022"""
        batch_size = x.size(0)
        device = x.device
        embedding_dim = model.embedding.embedding_dim
        
        if training:
            # During training: process half batch
            half_size = batch_size // 2
            
            # Initialize output tensor
            p_embeddings = torch.zeros(batch_size, *x.shape[1:], embedding_dim, device=device)
            
            # Split padding mask if it exists
            half_padding_mask = None
            if padding_mask is not None:
                half_padding_mask = padding_mask[half_size:]
            
            # For second half: get predictions without self-conditioning
            with torch.no_grad():
                # Use the corresponding timesteps for the second half
                half_timesteps = timesteps[half_size:]
                
                # Get predictions using the original indices
                pred_logits = model(
                    x[half_size:],
                    half_timesteps,
                    training=False,
                    padding_mask=half_padding_mask
                )
                
                # Apply padding mask to logits before softmax if needed
                if half_padding_mask is not None:
                    pred_logits = pred_logits.masked_fill(
                        half_padding_mask.unsqueeze(-1), 
                        float('-inf')
                    )
                
                probs = torch.softmax(pred_logits, dim=-1)
                
                # Interpolate embeddings
                p_embeddings[half_size:] = torch.matmul(
                    probs, 
                    model.embedding.normalized_embeddings
                )
                
                # Zero out padding positions in final embeddings
                if half_padding_mask is not None:
                    p_embeddings[half_size:] = p_embeddings[half_size:].masked_fill(
                        half_padding_mask.unsqueeze(-1), 
                        0.0
                    )
            
            return p_embeddings
            
        else:
            # During sampling: use previous predictions if available
            if self.prev_preds is None:
                return torch.zeros((x.size(0), x.size(1), embedding_dim), device=device)
                        
            embeddings = torch.matmul(
                self.prev_preds,
                model.embedding.normalized_embeddings
            )
            
            # Apply padding mask if it exists
            if padding_mask is not None:
                embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                
            return embeddings
    
    def update_prev_preds(self, logits, padding_mask=None):
        """Store softmax probabilities for next sampling step"""
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        self.prev_preds = torch.softmax(logits, dim=-1)
        
    def reset(self):
        """Reset stored predictions (call at start of sampling)"""
        self.prev_preds = None