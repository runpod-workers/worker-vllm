import os
from huggingface_hub import snapshot_download
import json

if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        raise ValueError("Must specify model name by adding --build-arg MODEL_NAME=<your model's repo>")
    revision = os.getenv("MODEL_REVISION") or None
    snapshot_download(model_name, revision=revision, cache_dir=os.getenv("HF_HOME"))
    
    tokenizer_name = os.getenv("TOKENIZER_NAME") or None
    tokenizer_revision = os.getenv("TOKENIZER_REVISION") or None
    if tokenizer_name:  
        snapshot_download(tokenizer_name, revision=tokenizer_revision, cache_dir=os.getenv("HF_HOME"))
        
    # Create file with metadata of baked in model and/or tokenizer
    
    with open("/local_metadata.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "revision": revision,
            "tokenizer_name": tokenizer_name or model_name,
            "tokenizer_revision": tokenizer_revision or revision,
            "quantization": os.getenv("QUANTIZATION")
        }, f)

