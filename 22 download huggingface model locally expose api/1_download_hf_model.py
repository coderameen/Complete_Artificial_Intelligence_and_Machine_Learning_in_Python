#I Can direcly inference with local using VLLM, but If VLLM is not available in the huggingface then this step need to be done.

"""
python -m venv qwen-env
source qwen-env/bin/activate  # Windows: qwen-env\Scripts\activate
pip install torch transformers accelerate fastapi uvicorn

pip install bitsandbytes
"""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    local_dir="./models/qwen3.5-0.8b",
    local_dir_use_symlinks=False
)