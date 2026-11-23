import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset, special_tokens

class ProteinRNN(nn.Module):
    def __init__(self, vocab_size = 23, embed_dim = 32, hidden_dim=128, layers = 1):
        super(ProteinRNN,self).__init__()
        self.rnn = nn.RNN(input_size = embed_dim,
                          hidden_size = hidden_dim,
                          num_layers = layers,
                          batch_first = True)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x, h_prev=None):
        
        x = self.embedding(x)
        
        rnn_out, h_n = self.rnn(x,h_prev)
        
        last_output = rnn_out[:, -1, :]
        
        logits = self.fc(rnn_out)
        
        return logits, h_n

ROOT = Path(__file__).resolve().parents[3]   # models -> protgen -> src -> project root

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, x_shifted, y, length in loader:
        x, x_shifted = x.to(device), x_shifted.to(device)
        logits, _ = model(x)
        total_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_shifted.reshape(-1),
                                      ignore_index=-1, reduction="sum").item()
        mask = x_shifted != -1
        correct += (logits.argmax(dim=-1)[mask] == x_shifted[mask]).sum().item()
        count += mask.sum().item()
    loss = total_loss / count
    return {"loss": loss, "ppl": math.exp(loss), "acc": correct / count}

if __name__ == "__main__":
    train_ds = ProteinDataset(ROOT / "data/train.csv")
    val_ds = ProteinDataset(ROOT / "data/val.csv")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = ProteinRNN().to(device)
    print(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_epochs = 10

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_count = 0.0, 0
        for x, x_shifted, y, length in train_loader:
            x, x_shifted = x.to(device), x_shifted.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 23), x_shifted.reshape(-1), ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # loss is a per-token mean, so weight it by the batch's token count for an epoch average
            n_tokens = (x_shifted != -1).sum().item()
            train_loss += loss.item() * n_tokens
            train_count += n_tokens

        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1:2d}  train {train_loss / train_count:.4f}  "
              f"val {val['loss']:.4f} , acc {val['acc']:.3f})")
    
    save_dir = ROOT / "src/protgen/model_weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "rnn.pt")
