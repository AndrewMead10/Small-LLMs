import torch
from torch import nn
from smllm.modules.linears import GeluMLP, Embedding, LMHead
from smllm.modules.attention import MHA
from smllm.inference.KVCache import KVCache, ModelKVCache


class PhiTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.resid_dropout = nn.Dropout(dropout)
        self.mha = MHA(hidden_size, n_heads, dropout)
        self.mlp = GeluMLP(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        rope: torch.Tensor = None,
        index: torch.Tensor = None,
        kv_cache: KVCache = None,
    ) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        attn_out = self.mha(x, mask, rope, index, kv_cache)
        attn_out = self.resid_dropout(attn_out)
        ff = self.resid_dropout(self.mlp(x))

        return residual + attn_out + ff


class Phi(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.0,
        n_blocks: int = 24,
        vocab_size: int = 51200,
        max_seq_len: int = 2048,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, hidden_size, dropout)
        self.blocks = nn.Sequential(
            *[
                PhiTransformerBlock(hidden_size, n_heads, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.lm_head = LMHead(hidden_size, vocab_size)

        self.batch_size = 1

        if self.eval:
            self.kv_cache = ModelKVCache(n_blocks, 1, max_seq_len, hidden_size, n_heads)
        else:
            self.kv_cache = None

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        index: torch.Tensor = None,
        rope: torch.Tensor = None,
    ) -> torch.Tensor:
        bs, seq_len = x.shape

        if bs != self.batch_size and self.eval:
            self.batch_size = bs
            self.kv_cache = ModelKVCache(
                self.n_blocks, bs, self.max_seq_len, self.hidden_size, self.n_heads
            )

        x = self.embedding(x)

        for i, block in enumerate(self.blocks):
            if self.training:
                x = block(x, mask=mask, rope=rope)
            else:
                x = block(
                    x, kv_cache=self.kv_cache[i], mask=mask, index=index, rope=rope
                )

        x = self.lm_head(x)
        return x

    def from_pretrained(self, path):
        weights = torch.load(path, map_location=self.device)

        key_mapping = self.map_hf_state_dict(weights.keys())
        weights = {key_mapping[k]: v for k, v in weights.items()}

        # get rid of all keys that have inv_freq in them
        weights = {k: v for k, v in weights.items() if "inv_freq" not in k}

        self.load_state_dict(weights)

    def map_hf_state_dict(self, state_dict_keys):
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
                        f"layers.{block_num + 1}.mlp.fc1",
                        f"blocks.{block_num}.mlp.linear1",
                    )
                    mapped_key = mapped_key.replace(
                        f"layers.{block_num + 1}.mlp.fc2",
                        f"blocks.{block_num}.mlp.linear2",
                    )

                else:
                    raise ValueError(f"Unexpected key: {key}")

            key_mapping[key] = mapped_key

        return key_mapping
