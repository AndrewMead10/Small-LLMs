from phi import Phi, map_hf_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
hf_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5", trust_remote_code=True, cache_dir="./model_cache/phi-1.5"
)
hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# Load Phi model
phi_model = Phi(
    hidden_size=2048, n_heads=32, dropout=0.0, n_blocks=24, vocab_size=51200
)
phi_model.init_cache()

weights = torch.load(
    ".\\model_cache\\phi-1.5\\models--microsoft--phi-1_5\\snapshots\\4a426d8015bef5a0cb3acff8d4474ee9ab4071d5\\pytorch_model.bin"
)

key_mapping = map_hf_state_dict(weights.keys(), weights)
weights = {key_mapping[k]: v for k, v in weights.items()}

# get rid of all keys that have inv_freq in them
weights = {k: v for k, v in weights.items() if "inv_freq" not in k}

phi_model.load_state_dict(weights)

# Compare outputs
text = "Hello, my dog is cute"
input = hf_tokenizer(text, return_tensors="pt")

hf_output = hf_model(**input)
phi_output = phi_model.generate(input["input_ids"])

print(torch.allclose(hf_output.logits, phi_output))
# print out the total error bewteen the models
print(torch.sum(torch.abs(hf_output.logits - phi_output)))
