#pip uninstall -y tokenizers transformers
# pip install transformers git+https://github.com/huggingface/transformers.git
# pip install tokenizers==0.22.1
# pip install accelerate safetensors

'''
deactivate
source qwen-env/bin/activate   # if using venv
'''
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./models/qwen3.5-0.8b"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

app = FastAPI()

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 150

@app.post("/generate")
def generate(req: Req):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens)

    return {
        "response": tokenizer.decode(out[0], skip_special_tokens=True)
    }