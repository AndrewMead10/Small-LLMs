import torch
from torch import nn


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size,
        max_seq_len,
        hidden_size,
        n_heads,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_len, hidden_size // n_heads)
        self.k = torch.zeros(cache_shape, device=device, dtype=dtype)

        self.v = torch.zeros(cache_shape, device=device, dtype=dtype)

    def update(self, k_val, v_val, index):
        assert (
            index.shape[0] == k_val.shape[2]
        ), "index shape {} does not match k_val shape {}".format(
            index.shape, k_val.shape
        )
        self.k[:, :, index] = k_val
        self.v[:, :, index] = v_val

        return self.k, self.v


class ModelKVCache(nn.Module):
    def __init__(
        self,
        num_layers,
        max_batch_size,
        max_seq_len,
        hidden_size,
        n_heads,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.device = device
        self.dtype = dtype
        self.cache = nn.ModuleList([])

    def initialize_cache(self):
        self.cache = nn.ModuleList(
            [
                KVCache(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.hidden_size,
                    self.n_heads,
                    self.device,
                    self.dtype,
                )
                for _ in range(self.num_layers)
            ]
        )

    def __getitem__(self, index):
        return self.cache[index]

    def reset_cache(self):
        self.cache = nn.ModuleList([])
