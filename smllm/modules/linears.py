from torch import nn
import torch
from smllm.modules.activation import Gelu


class GeluMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size, bias=True)
        self.gelu = Gelu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.linear(x)
        return x
