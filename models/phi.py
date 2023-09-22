from torch import nn
from torch.nn import functional as F
import torch
import math


class Gelu(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
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
        kv_cache: KVCache = None,
        index: torch.Tensor = None,
        rope: torch.Tensor = None,
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

        k, v = kv_cache.update(k, v, index)

        k, v = k[:, :, : index[-1] + 1], v[:, :, : index[-1] + 1]

        context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        context = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context)

        return out


class Block(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.resid_dropout = nn.Dropout(dropout)
        self.mha = MHA(hidden_size, n_heads, dropout)
        self.mlp = MLP(hidden_size, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        kv_cache: KVCache = None,
        index: torch.Tensor = None,
        rope: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        attn_out = self.mha(x, mask, kv_cache, index, rope)
        attn_out = self.resid_dropout(attn_out)
        ff = self.resid_dropout(self.mlp(x))

        return residual + attn_out + ff


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.linear(x)
        return x


class Phi(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.0,
        n_blocks: int = 24,
        vocab_size: int = 51200,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_size, dropout)
        self.blocks = nn.Sequential(
            *[Block(hidden_size, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.lm_head = LMHead(hidden_size, vocab_size)

        self.kv_cache = ModelKVCache(n_blocks, 1, 1024, hidden_size, n_heads)

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        index: torch.Tensor = None,
        rope: torch.Tensor = None,
    ) -> torch.Tensor:
        print(x)
        x = self.embedding(x)
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=self.kv_cache[i], mask=mask, index=index, rope=rope)
        x = self.lm_head(x)
        return x

    def init_cache(self):
        self.kv_cache.initialize_cache()

        ones = torch.ones((2048, 2048), device=self.device, dtype=self.dtype)
        self.mask_cache = torch.tril(ones).unsqueeze(0)

        self.rope_cache = build_rope_cache(
            seq_len=self.hidden_size,
            n_elem=32,
            dtype=self.dtype,
            device=self.device,
        )

    def generate(self, input_ids, max_length=1):
        # input_ids (BS, seq_len)
        bs, seq_len = input_ids.shape
        out_seq_len = seq_len + max_length
        out_vec = torch.zeros((bs, out_seq_len), dtype=torch.long, device=self.device)
        out_vec[:, :seq_len] = input_ids

        first_token = self.process_input(input_ids)
        out_vec[:, seq_len] = first_token

        index = torch.tensor([seq_len], device=self.device)

        for i in range(max_length - 1):
            # update rope
            rope = self.rope_cache.index_select(0, index)
            # make a 1,1 tensor from the value at out_vec[0, index]
            cur_token = out_vec[0, index].unsqueeze(0).to(torch.int64)

            output = self(cur_token, index=index, rope=rope)

            token = self.sample_output(output)

            out_vec[:, index + 1] = token
            index = index + 1

        return out_vec

    def process_input(self, input_ids):
        # input_ids (BS, seq_len)
        bs, seq_len = input_ids.shape

        index = torch.arange(0, seq_len, device=self.device)
        mask = self.mask_cache.index_select(2, index)
        mask = mask[:, :seq_len, :seq_len]
        rope = self.rope_cache.index_select(0, index)
        # process input_ids and get first output
        out = self(input_ids, mask.bool(), index, rope)

        token = self.sample_output(out)

        return token

    def fast_multinomial_sample_one(self, probs_sort):
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True)

    def sample_output(self, logits):
        # get the last token preds
        next_token_logits = logits[0, -1]
        # sample from our output logits
        probs = F.softmax(next_token_logits, dim=-1)
        # next_token = self.fast_multinomial_sample_one(probs)
        next_token = probs.argmax(-1).unsqueeze(0)

        return next_token


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


def map_hf_state_dict(state_dict_keys, weights):
    key_mapping = {}

    for key in state_dict_keys:
        # Extract block number
        block_num = (
            int(key.split(".")[1]) - 1
        )  # Adjusting for the difference in starting index

        if "wte" in key:
            mapped_key = key.replace("layers.0.wte", "embedding.embedding")
        else:
            if block_num == 24:
                mapped_key = key.replace("layers.25", "lm_head")

            # Map layer norm
            elif ".ln." in key:
                mapped_key = key.replace(
                    f"layers.{block_num + 1}.ln", f"blocks.{block_num}.ln"
                )

            # Map multi-head attention / rotary embedding
            elif ".mixer." in key:
                if "rotary_emb" in key:
                    mapped_key = key.replace(
                        f"layers.{block_num + 1}.mixer", f"blocks.{block_num}.mha"
                    )
                else:
                    mapped_key = key.replace(
                        f"layers.{block_num + 1}.mixer", f"blocks.{block_num}.mha"
                    )

            # Map MLP layers
            elif ".mlp." in key:
                mapped_key = key.replace(
                    f"layers.{block_num + 1}.mlp.fc1", f"blocks.{block_num}.mlp.linear1"
                )
                mapped_key = mapped_key.replace(
                    f"layers.{block_num + 1}.mlp.fc2", f"blocks.{block_num}.mlp.linear2"
                )

            else:
                raise ValueError(f"Unexpected key: {key}")

        key_mapping[key] = mapped_key

    return key_mapping
