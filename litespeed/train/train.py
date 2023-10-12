import torch
from typing import List
import wandb
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from litespeed.models.phi import Phi
from litespeed.data.make_dataset import make_train_dl

# cpu offload
# https://discuss.pytorch.org/t/modifying-forward-backward-pass/169687/2

model = Phi(hidden_size=2048, n_heads=32, dropout=0.0, n_blocks=24, vocab_size=51200)

train_dataloader = make_train_dl()
