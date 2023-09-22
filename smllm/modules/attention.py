import torch
from torch import nn
import torch.nn.functional as F
from smllm.modules.rope import apply_rope
from smllm.inference.KVCache import KVCache


class MHA(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.n_heads = n_heads

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        rope: torch.Tensor = None,
        index: torch.Tensor = None,
        kv_cache: KVCache = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.Wqkv(x)

        q, k, v = qkv.split(C, dim=-1)

        head_size = C // self.n_heads
        q = q.view(B, T, self.n_heads, head_size)
        k = k.view(B, T, self.n_heads, head_size)
        v = v.view(B, T, self.n_heads, head_size)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, index)
            k, v = k[:, :, : index[-1] + 1], v[:, :, : index[-1] + 1]

        # print(q.shape, k.shape, v.shape)

        context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        context = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context)

        return out
