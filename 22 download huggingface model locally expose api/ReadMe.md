You can absolutely run Hugging Face models locally and expose them as a service so any app on your machine (or even your network) can use them like an API. I’ll walk you through a clean, practical setup.


NOTE: I Can direcly inference with local using VLLM, but If VLLM is not available in the huggingface then this step need to be done.

#=========STEP 1 (Download model)============
URL: https://huggingface.co/Qwen/Qwen3.5-0.8B



>>python -m venv qwen-env

>>source qwen-env/bin/activate 

>>pip install torch transformers accelerate fastapi uvicorn

>>pip install bitsandbytes

RUN:
>>python 1_download_hf_model.py

#=========STEP 2 (Model Testing)============
RUN:
>>python 2_test_model.py

#=========STEP 3 (Expose API)=============
>>pip uninstall -y tokenizers transformers

>>pip install transformers git+https://github.com/huggingface/transformers.git

>> pip install tokenizers==0.22.1

>>pip install accelerate safetensors

RUN:
>>uvicorn app:app --host 0.0.0.0 --port 8009



#========TESTING (Postman)=======
curl --location 'http://localhost:8009/generate' \
--header 'Content-Type: application/json' \
--data '{"prompt": "Explain python in simple terms", "max_new_tokens": 150}'
