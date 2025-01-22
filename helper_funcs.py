import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
loaded_df = pd.read_hdf('./data/sample_otu_arrays.h5', key='df')
def generate_sequences(model, num_sequences=1, seq_length=None, num_steps=100, temperature=0, device='cuda'):
    model.eval()
    model = model.to(device)
    
    # Handle sequence lengths
    if seq_length is None:
        real_lengths = [len(x) for x in loaded_df['otu_arrays']]
        sampled_lengths = np.random.choice(real_lengths, size=num_sequences)
        max_sampled_length = max(sampled_lengths)
    else:
        sampled_lengths = [seq_length] * num_sequences
        max_sampled_length = seq_length
    
    sampled_lengths = torch.tensor(sampled_lengths, device=device)
    
    with torch.no_grad():
        # Get warped timesteps
        uniform_steps = torch.linspace(1, 0, num_steps).to(device)
        timesteps = model.time_warping.warp_time(uniform_steps)
        
        # Initialize with scaled noise
        torch.manual_seed(42)
        sigma_T = torch.sqrt(timesteps[0])
        xt = sigma_T * torch.randn(num_sequences, max_sampled_length, model.embed_dim).to(device)
        
        # Create mask
        mask = torch.zeros(num_sequences, max_sampled_length, dtype=torch.bool, device=device)
        for i, length in enumerate(sampled_lengths):
            mask[i, length:] = True
        
        print("\nRunning diffusion...")
        for i in tqdm(range(len(timesteps) - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            t_batch = t.expand(num_sequences)
            logits = model(xt, mask, t_batch)
            expected_x0 = model.get_expected_embedding(logits)
            
            if t > t_next:
                score = model.calculate_score(xt, expected_x0, t_batch)
                dt = t_next - t
                xt = xt -  score * dt #debate as to whether a factor of t should be here
                
                # Add noise
                sigma = torch.sqrt(t_next) * 0 # dot times by 0 for regular ODE as in paper
                noise = torch.randn_like(xt, device=device)
                xt = xt + sigma * noise
        
        # Final prediction
        final_logits = model(xt, mask, torch.zeros(num_sequences, device=device))
        
        # Token selection
        final_tokens = torch.zeros((num_sequences, max_sampled_length), 
                                 dtype=torch.long, device=device)
        
        if temperature == 0:
            # Greedy sampling
            for i in range(num_sequences):
                logits_seq = final_logits[i, :sampled_lengths[i]]
                mask_tokens = torch.ones_like(logits_seq, dtype=torch.bool, device=device)
                for j in range(sampled_lengths[i]):
                    masked_logits = logits_seq[j].clone()
                    masked_logits[~mask_tokens[j]] = float('-inf')
                    token = torch.argmax(masked_logits)
                    final_tokens[i, j] = token
                    mask_tokens[j:, token] = False
        else:
            # Temperature sampling
            for i in range(num_sequences):
                logits_seq = final_logits[i, :sampled_lengths[i]]
                probs = F.softmax(logits_seq / temperature, dim=-1)
                mask_tokens = torch.ones_like(probs, dtype=torch.bool, device=device)
                for j in range(sampled_lengths[i]):
                    masked_probs = probs[j] * mask_tokens[j]
                    if masked_probs.sum() > 0:
                        masked_probs = masked_probs / masked_probs.sum()
                        token = torch.multinomial(masked_probs, 1)
                        final_tokens[i, j] = token
                        mask_tokens[j:, token] = False
                    else:
                        break
        
        return final_tokens.cpu().numpy()
