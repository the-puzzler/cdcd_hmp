import torch
import torch.nn as nn

class DiffusionNoise:
    def __init__(self, t_min=1.0, t_max=300.0):
        self.t_min = t_min
        self.t_max = t_max
    
    def add_noise(self, embeddings, self_cond, t):
        """
        Add noise and apply preconditioning
        Args:
            embeddings: normalized embeddings
            self_cond: self-conditioning embeddings
            t: timesteps (sigma in Karras notation)
        """
        # Add noise to embeddings (but not self_cond)
        noise = torch.randn_like(embeddings) * t.view(*t.shape, 1)
        noisy_embeddings = embeddings + noise
        
        # Concatenate with self-conditioning
        x = torch.cat([noisy_embeddings, self_cond], dim=-1)
        
        # Apply input preconditioning (cin)
        c_in = 1 / torch.sqrt(t**2 + 1).view(*t.shape, 1)
        x = x * c_in
        
        return x, noise
    
    def sample_timesteps(self, shape, device):
        """Sample random timesteps uniformly between t_min and t_max"""
        return torch.rand(shape, device=device) * (self.t_max - self.t_min) + self.t_min