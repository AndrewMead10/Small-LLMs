import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import time

from litespeed.data.make_dataset import make_train_dl
from litespeed.train.lora import add_lora_layers
from litespeed.train.train_utils import CheckpointWrapper

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
    grad_accumulation_steps = 2
    epochs = 4
    gradient_checkpointing = True
    use_torch_compile = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # use_flash_attention=True,
        device_map="auto",
        trust_remote_code=True,
    )

    model = model.train()

    if use_lora:
        add_lora_layers(model, r, lora_dropout, alpha)
        for name, param in model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    else:
        raise NotImplementedError("only lora training is currently supported")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5
    )

    loss_fn = nn.CrossEntropyLoss()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if use_torch_compile:
        model = torch.compile(model)

    train_dataloader = make_train_dl(
        "C:\\Users\\andre\\OneDrive\\Documents\\coding projects\\small LLMs\\litespeed\\data",
        tokenizer,
        ["system_prompt", "question", "response"],
        batch_size=batch_size,
        num_workers=10,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print num trainable params
    print("trainable params:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(epochs):
        for i, batch in tqdm(enumerate(train_dataloader)):
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask).logits

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), input_ids.view(-1))

            loss.backward()

            if (i + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                print(loss)
                # wandb.log({"loss": loss})
