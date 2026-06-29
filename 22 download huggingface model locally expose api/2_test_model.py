from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path = "./models/qwen3.5-0.8b"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

prompt = "Explain AI in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=150
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))