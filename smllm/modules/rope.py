import torch


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cpu",
    base: int = 10000,
) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    x_pass = x[:, :, :, 32:]
    x_rot = x[:, :, :, :32]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rope_cache = rope_cache.unsqueeze(1)

    cos = rope_cache[..., 0]
    sin = rope_cache[..., 1]

    x_out2 = torch.cat(
        [
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ],
        -1,
    )

    return torch.cat((x_out2, x_pass), dim=-1).type_as(x)
