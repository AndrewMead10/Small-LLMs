from smllm.models.phi import Phi
from smllm.inference.generate import Generate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# Load Phi model
phi_model = Phi(
    hidden_size=2048, n_heads=32, dropout=0.0, n_blocks=24, vocab_size=51200
)
phi_model.from_pretrained(
    "C:\\Users\\andre\\OneDrive\\Documents\\coding projects\\small LLMs\\models\\model_cache\\phi-1.5\\models--microsoft--phi-1_5\\snapshots\\4a426d8015bef5a0cb3acff8d4474ee9ab4071d5\\pytorch_model.bin"
)

phi_generator = Generate(
    phi_model, model_max_length=2048, use_calm=True, calm_lambda=0.95, calm_temp=1
)

# Compare outputs
text = ["Hello, my name is"]

hf_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
hf_tokenizer.padding_side = "left"
input = hf_tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=False,
    pad_to_multiple_of=8,
)


phi_output = phi_generator.generate(input["input_ids"], max_length=15)

print("Phi output: {}".format(phi_output))
print(hf_tokenizer.batch_decode(phi_output))

phi_generator.model.show_calm()
