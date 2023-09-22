from smllm.models.phi import Phi
from smllm.inference.generate import Generate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
hf_model = AutoModelForCausalLM.from_pretrained(
    ".\models\\model_cache\\phi-1.5\\models--microsoft--phi-1_5\\snapshots\\4a426d8015bef5a0cb3acff8d4474ee9ab4071d5",
    trust_remote_code=True,
)
hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# Load Phi model
phi_model = Phi(
    hidden_size=2048, n_heads=32, dropout=0.0, n_blocks=24, vocab_size=51200
)
phi_model.from_pretrained(
    "C:\\Users\\andre\\OneDrive\\Documents\\coding projects\\small LLMs\\models\\model_cache\\phi-1.5\\models--microsoft--phi-1_5\\snapshots\\4a426d8015bef5a0cb3acff8d4474ee9ab4071d5\\pytorch_model.bin"
)

phi_generator = Generate(phi_model, model_max_length=2048)

# Compare outputs
# text = ["Hello, my name is", "what is for dinner"]
text = ["Hello, my dog is cute", "Hello, my dog is cute"]

hf_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
input = hf_tokenizer(text, return_tensors="pt", padding=True, truncation=False)
print(input["input_ids"])

hf_output = hf_model.generate(input["input_ids"], max_length=9, min_length=9)
phi_output = phi_generator.generate(input["input_ids"], max_length=9)

print("HF output: {}".format(hf_output))
print("Phi output: {}".format(phi_output))
print(hf_tokenizer.decode(hf_output[0]))
print(hf_tokenizer.decode(phi_output[0]))

print(torch.allclose(hf_output, phi_output, atol=1e-5))
