from pathlib import Path

import torch
import torch.nn as nn
from protgen.data import ProteinDataset, special_tokens

ROOT = Path(__file__).resolve().parent

train_ds = ProteinDataset(ROOT / "data/train.csv")
val_ds = ProteinDataset(ROOT / "data/val.csv")

PAD_ID = special_tokens.index("<PAD>")
BOS_ID = special_tokens.index("<BOS>")
EOS_ID = special_tokens.index("<EOS>")

def context_window(tokens,token_shifted, context_size=3):
    #finder antal sequences, og deres længde
    N = tokens.shape[0]
    
    #laver en padding tensor med størrelsen (N, context_size - 1) som kun består af padding indexet
    padding = torch.full((N, context_size - 1), PAD_ID, dtype=tokens.dtype)
    
    # smækker padding tensoren i de to koloner foran tokensne
    padded_sec = torch.cat([padding,tokens],dim=1)
    
    # gennemgår 1. akse, med window størrelse "context window", stepsize
    windows=padded_sec.unfold(1,context_size,1)
    
    # mask til at fjerne padding, 
    mask = token_shifted != -1
    
    # returnere alle windows hvor der er et target, padding er ikke et target,
    # og target for hvert window
    return windows[mask], token_shifted[mask]



class ProteinMLP(nn.Module):
    def __init__(self, vocab_size = 23,context_size = 3, embed_dim = 32, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(context_size * embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(start_dim=1)
        x = self.hidden(x)
        x = torch.tanh(x)
        return self.output(x)
    
    
    def train(self, lr = 1e-3):
        