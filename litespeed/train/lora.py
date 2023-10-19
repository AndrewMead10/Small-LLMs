import torch.nn as nn
import torch
import math


class LoraLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        orig_weight=None,
        orig_bias=None,
        r=16,
        dropout=0.0,
        alpha=1.0,
    ):
        super(LoraLinear, self).__init__()
        self.orig_linear = nn.Linear(
            in_features,
            out_features,
            bias=orig_bias is not None,
            device="cuda:0",
            dtype=orig_weight.dtype,
        )

        if orig_weight is not None:
            self.orig_linear.weight.data = orig_weight.data
        if orig_bias is not None:
            self.orig_linear.bias.data = orig_bias.data

        self.lora_A = nn.Linear(
            out_features, r, bias=False, device="cuda:0", dtype=orig_weight.dtype
        )
        self.lora_B = nn.Linear(
            r, out_features, bias=False, device="cuda:0", dtype=orig_weight.dtype
        )

        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = alpha / r

        self.merged = False

    def forward(self, x):
        x = self.orig_linear(x)
        if not self.merged:
            x = self.dropout(x)
            x = self.lora_A(x)
            x = self.lora_B(x)
            x = x * self.scaling

        return x

    def merge(self):
        if not self.merged:
            self.orig_linear.weight.data += (
                self.lora_A.weight.data @ self.lora_B.weight.data
            ) * self.scaling
            self.merged = True


# add lora to all nn.Linear layers in model
def add_lora_layers(model, r, dropout, alpha):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            lora_layer = LoraLinear(
                module.in_features,
                module.out_features,
                module.weight,
                module.bias,
                r,
                dropout,
                alpha,
            )
            setattr(model, name, lora_layer)
        else:
            add_lora_layers(module, r, dropout, alpha)


def save_lora_params(model, name):
    lora_params = {
        name: param
        for name, param in model.state_dict().items()
        if model.named_parameters()[name].requires_grad
    }

    torch.save(lora_params, name)


def load_lora_params(model, name):
    lora_params = torch.load(name)
    model.load_state_dict(lora_params, strict=False)
