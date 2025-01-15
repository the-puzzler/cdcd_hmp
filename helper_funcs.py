import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
loaded_df = pd.read_hdf('./data/sample_otu_arrays.h5', key='df')
def generate_sequences(model, num_sequences=1, seq_length=None, num_steps=100, temperature=0, device='cpu'):
    model.eval()
    
    # Handle sequence lengths
    if seq_length is None:
        real_lengths = [len(x) for x in loaded_df['otu_arrays']]
        sampled_lengths = np.random.choice(real_lengths, size=num_sequences)
        max_sampled_length = max(sampled_lengths)
    else:
        sampled_lengths = [seq_length] * num_sequences
        max_sampled_length = seq_length

    with torch.no_grad():
        T = 1.0
        min_t = 0.001
        timesteps = torch.linspace(T, min_t, num_steps).to(device)
        dt = -(T-min_t)/num_steps
        t_batches = timesteps.unsqueeze(1).expand(-1, num_sequences)
        
        torch.manual_seed(42)
        xt = torch.randn(num_sequences, max_sampled_length, model.embed_dim).to(device)
        
        # Create proper padding mask
        mask = torch.zeros(num_sequences, max_sampled_length, dtype=bool).to(device)
        for i, length in enumerate(sampled_lengths):
            mask[i, length:] = True  # Mark padding positions as True
        
        print("Starting denoising process...")
        for t, t_batch in tqdm(zip(timesteps, t_batches)):
            logits = model(xt, mask, t_batch)
            expected_x0 = model.get_expected_embedding(logits)
            score = (expected_x0 - xt)/(t**2)
            
            if t > min_t:
                update = -t * score * dt
                xt = xt + update

        final_logits = model(xt, mask, torch.zeros(num_sequences).to(device))
        
        
        # Vectorized token selection
        final_tokens = torch.zeros((num_sequences, max_sampled_length), dtype=torch.long, device=device)
        
        if temperature == 0:
            # Parallel argmax processing
            for i in range(num_sequences):
                logits_seq = final_logits[i, :sampled_lengths[i]]
                
                mask = torch.ones_like(logits_seq, dtype=torch.bool)
                for j in range(sampled_lengths[i]):
                    # Apply mask to prevent repeated tokens
                    masked_logits = logits_seq[j].clone()
                    masked_logits[~mask[j]] = float('-inf')
                    token = torch.argmax(masked_logits)
                    final_tokens[i, j] = token
                    mask[j:, token] = False
        else:
            # Parallel temperature-based sampling
            for i in range(num_sequences):
                logits_seq = final_logits[i, :sampled_lengths[i]]
                probs = F.softmax(logits_seq / temperature, dim=-1)
                mask = torch.ones_like(probs, dtype=torch.bool)
                for j in range(sampled_lengths[i]):
                    masked_probs = probs[j] * mask[j]
                    if masked_probs.sum() > 0:
                        masked_probs = masked_probs / masked_probs.sum()
                        token = torch.multinomial(masked_probs, 1)
                        final_tokens[i, j] = token
                        mask[j:, token] = False
                    else:
                        break
        
        return final_tokens.cpu().numpy()
