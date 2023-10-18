import torch
from datasets import load_dataset


def make_train_dl(dataset_name, tokenizer, data_columns, batch_size=32, num_workers=0):
    dataset = load_dataset(dataset_name)

    dataset["train"] = dataset["train"].select(range(10000))

    def tokenize_function(examples):
        # combine together the data_columns, seperated by newlines
        examples["text"] = []
        for i in range(len(examples[data_columns[0]])):
            examples["text"].append(
                "\n\n".join([examples[col][i] for col in data_columns])
            )
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=2048,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=data_columns,
        load_from_cache_file=True,
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader
