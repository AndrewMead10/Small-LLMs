import torch
import torch.nn.functional as F
from litespeed.modules.rope import build_rope_cache
from litespeed.inference.calm import CalmModel


class Generate:
    def __init__(
        self, model, model_max_length=2048, use_calm=False, calm_lambda=1, calm_temp=2
    ):
        self.model = model
        self.hidden_size = model.hidden_size
        self.device = model.device
        self.dtype = model.dtype
        self.kv_cache = model.kv_cache
        self.model_max_length = model_max_length

        self.model = self.model.eval()

        if use_calm:
            self.model = CalmModel(model, calm_lambda=calm_lambda, calm_temp=calm_temp)

        self.init_cache()

    def init_cache(self):
        self.kv_cache.initialize_cache()

        ones = torch.ones(
            (self.model_max_length, self.model_max_length),
            device=self.device,
            dtype=self.dtype,
        )
        self.mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0).repeat(8, 1, 1, 1)

        self.rope_cache = build_rope_cache(
            seq_len=self.hidden_size,
            n_elem=32,
            dtype=self.dtype,
            device=self.device,
        )

    def generate(self, input_ids, max_length=1):
        bs, seq_len = input_ids.shape
        if max_length < seq_len:
            raise ValueError(
                "max_length should be greater than input_ids length. Got {} and {}"
                "respectively".format(max_length, seq_len)
            )

        # preallocate output vector
        out_vec = torch.zeros((bs, max_length), dtype=torch.long, device=self.device)
        out_vec[:, :seq_len] = input_ids

        with torch.no_grad():
            # process prompt and get first token
            first_token = self.process_input(input_ids)
            out_vec[:, seq_len : seq_len + 1] = first_token

            index = torch.tensor([seq_len], device=self.device)

            token = first_token

            for _ in range(max_length - seq_len - 1):
                rope = self.rope_cache.index_select(0, index)

                output = self.model(token, index=index, rope=rope)
                token = self.sample_output(output)

                out_vec[:, index + 1 : index + 2] = token
                index = index + 1

            return out_vec

    def process_input(self, input_ids):
        # input_ids (BS, seq_len)
        bs, seq_len = input_ids.shape

        index = torch.arange(0, seq_len, device=self.device)
        mask = self.mask_cache.index_select(2, index)
        mask = mask[:bs, :, :seq_len, :seq_len]
        rope = self.rope_cache.index_select(0, index)
        # process input_ids and get first output
        out = self.model(input_ids, mask.bool(), index, rope)
        token = self.sample_output(out)

        return token

    def fast_multinomial_sample_one(self, probs_sort):
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True)

    def sample_output(self, logits):
        # get the last token preds
        next_token_logits = logits[:, -1]
        # sample from our output logits
        probs = F.softmax(next_token_logits, dim=-1)
        # next_token = self.fast_multinomial_sample_one(probs)
        next_token = probs.argmax(-1).unsqueeze(1)

        return next_token
