import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset, special_tokens
from protgen.models.RNN_v import ProteinRNN

device = torch.device('cpu')
model = ProteinRNN()
model.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))