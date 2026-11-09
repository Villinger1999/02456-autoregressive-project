import torch
import torch.nn.functional as F

class ProteinMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = F.Embedding(23, 32)
        self.hidden = F.Linear(3 * 32, 128)
        self.output = F.Linear(128, 23)

    def forward(self, x):

        x = self.embedding(x)
        x = x.flatten(start_dim=1)
        x = self.hidden(x)
        x = torch.tanh(x)
        x = self.output(x)

        return F.softmax(x)
    
    