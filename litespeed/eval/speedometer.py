from litespeed.models.phi import Phi
from litespeed.inference.generate import Generate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


def benchmark_generate(input_ids, max_length):
    # warmup
    phi_generator.generate(input_ids, max_length=max_length + max_length)

    # benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(3):
        phi_generator.generate(input_ids, max_length=max_length)
        torch.cuda.synchronize()
    end = time.time()
    print("{} tokens: {}".format(max_length, (end - start) / 3))


# Load Phi model
phi_model = Phi(
    hidden_size=2048, n_heads=32, dropout=0.0, n_blocks=24, vocab_size=51200
)
phi_model.from_pretrained(
    "C:\\Users\\andre\\OneDrive\\Documents\\coding projects\\small LLMs\\models\\model_cache\\phi-1.5\\models--microsoft--phi-1_5\\snapshots\\4a426d8015bef5a0cb3acff8d4474ee9ab4071d5\\pytorch_model.bin"
)

input_ids_128 = torch.randint(0, 51200, (1, 128))

input_ids_1024 = torch.randint(0, 51200, (1, 1024))


phi_generator = Generate(phi_model, model_max_length=2048)

print("No torch compile")
benchmark_generate(input_ids_128, 128 * 2)
benchmark_generate(input_ids_1024, 1024 * 2)


phi_model = torch.compile(phi_model)

phi_generator = Generate(phi_model, model_max_length=2048)

print("With torch compile")
benchmark_generate(input_ids_128, 128 * 2)
benchmark_generate(input_ids_1024, 1024 * 2)
