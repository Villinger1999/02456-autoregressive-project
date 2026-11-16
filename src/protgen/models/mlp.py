import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

class ProteinMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(23, 32)
        self.hidden = nn.Linear(3*32, 128)
        self.output = nn.Linear(128, 23)

    def forward(self, x):
        x = self.embedding(x)
        x = x.flatten(start_dim=1)
        x = self.hidden(x)
        x = torch.tanh(x)
        return self.output(x)

# Makes windows of 3 tokens, adds 2 x padding in the beginning and removes paddings from the token sequences in the training data
def prepare_batch(x, x_shifted):
    x_padded = F.pad(x, (2, 0), value=0)
    X = x_padded.unfold(dimension=1, size=3, step=1)
    X = X.reshape(-1, 3)
    y = x_shifted.reshape(-1)
    mask = y != -1
    return X[mask], y[mask]


# Evaluates the model loss based on the average loss per valid token in the dataset
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for x, x_shifted, labels, lengths in dataloader:
        x = x.to(device)
        x_shifted = x_shifted.to(device)
        X, y = prepare_batch(x, x_shifted)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / total_tokens

# get data sets
train_data = ProteinDataset(path_to_data="data/train.csv")
val_data = ProteinDataset(path_to_data="data/val.csv")
test_data = ProteinDataset(path_to_data="data/test.csv")

# To use for testing
small_train_data = Subset(train_data, range(1024))
small_test_data = Subset(test_data, range(128))
small_val_data = Subset(val_data, range(128))


# training the model using crossentropy loss and adam optimizer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProteinMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

num_epochs = 20
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_tokens = 0
    for x, x_shifted, labels, lengths in train_loader:
        x = x.to(device)
        x_shifted = x_shifted.to(device)
        X, y = prepare_batch(x, x_shifted)
        forward = model(X)
        loss = F.cross_entropy(forward, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    train_losses.append(avg_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}: "f"Train Loss = {avg_loss:.4f}")
    
val_loss = evaluate(model, val_loader)
print(f"Epoch {epoch+1}/{num_epochs}: "f"Train Loss = {avg_loss:.4f}, "f"Val Loss = {val_loss:.4f}")
    
# evaluate model on test data
test_loader = DataLoader(test_data,batch_size=32,shuffle=False)
test_loss = evaluate(model, test_loader)
print(f"Test loss: {test_loss:.4f}")

# generates protein sequences that never generates PAD or BOS 
@torch.no_grad()
def generate_protein(model, dataset, max_length=127):
    model.eval()
    device = next(model.parameters()).device
    vocab = dataset.vocab
    bos = vocab["<BOS>"]
    eos = vocab["<EOS>"]
    pad = vocab["<PAD>"]
    context = [pad, pad, bos]
    generated = []
    
    for _ in range(max_length):
        x = torch.tensor([context[-3:]], device=device)
        logits = model(x)[0]
        logits[pad] = -float("inf")
        logits[bos] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == eos:
            break
        generated.append(next_token)
        context.append(next_token)
    return dataset.decode(generated)

# generate and save sequences
sequences = [generate_protein(model, train_data) for _ in range(256)]

df = pd.DataFrame({"sequence": sequences})
df.to_csv("results/sequences/mlp_generated.csv", index=False)