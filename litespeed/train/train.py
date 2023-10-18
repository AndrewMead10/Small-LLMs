import torch
from typing import List
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from litespeed.models.phi import Phi

from litespeed.data.make_dataset import make_train_dl
from litespeed.train.lora import add_lora_layers

from transformers import AutoTokenizer, AutoModelForCausalLM

# cpu offload
# https://discuss.pytorch.org/t/modifying-forward-backward-pass/169687/2

if __name__ == "__main__":
    use_lora = True
    r = 16
    alpha = 1
    lora_dropout = 0.1
    model_name = "mistralai/Mistral-7B-v0.1"
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # use_flash_attention=True,
        device_map="auto",
        trust_remote_code=True,
    )

    print(model)
    exit()

    model = model.eval()

    if use_lora:
        for param in model.parameters():
            param.requires_grad = False

        add_lora_layers(model, r, lora_dropout, alpha)

        # print num trainable params
        print("trainable params:")
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    else:
        raise NotImplementedError("only lora training is currently supported")

    train_dataloader = make_train_dl(
        "C:\\Users\\andre\\OneDrive\\Documents\\coding projects\\small LLMs\\litespeed\\data",
        tokenizer,
        ["system_prompt", "question", "response"],
        batch_size=32,
        num_workers=10,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5
    )

    for i, batch in enumerate(train_dataloader):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, input_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print(loss)
            # wandb.log({"loss": loss})
