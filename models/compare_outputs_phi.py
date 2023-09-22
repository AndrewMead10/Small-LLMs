from phi import Phi, map_hf_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
hf_model = AutoModelForCausalLM.from_pretrained(
    ".\\model_cache\\phi-1.5\\models--microsoft--phi-1_5\\snapshots\\4a426d8015bef5a0cb3acff8d4474ee9ab4071d5\\",
    trust_remote_code=True,
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
print(input["input_ids"])


# hf_out = hf_model(input["input_ids"])

# print(hf_out)
# # phi_out = phi_model(input["input_ids"])

# print(torch.allclose(hf_out.logits, phi_out.logits, atol=1e-5))

hf_output = hf_model.generate(input["input_ids"], max_length=9, min_length=9)
phi_output = phi_model.generate(input["input_ids"], max_length=3)

print("HF output: {}".format(hf_output))
print("Phi output: {}".format(phi_output))
print(hf_tokenizer.decode(hf_output[0]))
print(hf_tokenizer.decode(phi_output[0]))
