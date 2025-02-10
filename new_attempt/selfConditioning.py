import torch

class SelfConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_preds = None
    
    def get_self_conditioning(self, x, model, embedding_layer, noise_layer, time_embedding_layer, training=True):
        """Get self-conditioning embeddings following Chen et al. 2022"""
        batch_size = x.size(0)
        device = x.device
        embedding_dim = embedding_layer.embedding_dim
        
        if training:
            # During training: process half batch
            half_size = batch_size // 2
            
            # Initialize output tensor
            p_embeddings = torch.zeros(batch_size, *x.shape[1:], embedding_dim, device=device)
            
            # For second half: get predictions without self-conditioning
            with torch.no_grad():
                # Sample timesteps for second half
                timesteps = noise_layer.sample_timesteps(half_size, device)
                
                # Get embeddings and add noise
                embeddings = embedding_layer(x[half_size:])
                noisy_emb, _ = noise_layer.add_noise(
                    embeddings,
                    torch.zeros_like(embeddings), # No self-conditioning for first pass
                    timesteps
                )
                
                # Get predictions
                pred_logits = model(noisy_emb, timesteps, training=False)
                probs = torch.softmax(pred_logits, dim=-1)
                
                # Interpolate embeddings
                p_embeddings[half_size:] = torch.matmul(
                    probs, 
                    embedding_layer.normalized_embeddings
                )
            
            return p_embeddings
            
        else:
            # During sampling: use previous predictions if available
            if self.prev_preds is None:
                return torch.zeros_like(x[:,:,None].expand(-1,-1,embedding_dim), device=device)
            
            return torch.matmul(
                self.prev_preds,
                embedding_layer.normalized_embeddings
            )
    
    def update_prev_preds(self, logits):
        """Store softmax probabilities for next sampling step"""
        self.prev_preds = torch.softmax(logits, dim=-1)
        
    def reset(self):
        """Reset stored predictions (call at start of sampling)"""
        self.prev_preds = None