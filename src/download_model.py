import os
import json
import logging
import glob
from shutil import rmtree
from huggingface_hub import snapshot_download
from utils import timer_decorator

BASE_DIR = "/" 
TOKENIZER_PATTERNS = [["*.json", "tokenizer*"]]
MODEL_PATTERNS = [["*.safetensors"], ["*.bin"], ["*.pt"]]

def setup_env():
    if os.getenv("TESTING_DOWNLOAD") == "1":
        BASE_DIR = "tmp"
        os.makedirs(BASE_DIR, exist_ok=True)
        os.environ.update({
            "HF_HOME": f"{BASE_DIR}/hf_cache",
            "MODEL_NAME": "openchat/openchat-3.5-0106",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TENSORIZE": "1",
            "TENSORIZER_NUM_GPUS": "1",
            "DTYPE": "auto"
        })

@timer_decorator
def download(name, revision, type, cache_dir):
    if type == "model":
        pattern_sets = [model_pattern + TOKENIZER_PATTERNS[0] for model_pattern in MODEL_PATTERNS]
    elif type == "tokenizer":
        pattern_sets = TOKENIZER_PATTERNS
    else:
        raise ValueError(f"Invalid type: {type}")
    try:
        for pattern_set in pattern_sets:
            path = snapshot_download(name, revision=revision, cache_dir=cache_dir, 
                                    allow_patterns=pattern_set)
            for pattern in pattern_set:
                if glob.glob(os.path.join(path, pattern)):
                    logging.info(f"Successfully downloaded {pattern} model files.")
                    return path
    except ValueError:
        raise ValueError(f"No patterns matching {pattern_sets} found for download.")
        
          
# @timer_decorator
# def tensorize_model(model_path): TODO: Add back once tensorizer is ready
#     from vllm.engine.arg_utils import EngineArgs
#     from vllm.model_executor.model_loader.tensorizer import TensorizerConfig, tensorize_vllm_model
#     from torch.cuda import device_count

#     tensorizer_num_gpus = int(os.getenv("TENSORIZER_NUM_GPUS", "1"))
#     if tensorizer_num_gpus > device_count():
#         raise ValueError(f"TENSORIZER_NUM_GPUS ({tensorizer_num_gpus}) exceeds available GPUs ({device_count()})")

#     dtype = os.getenv("DTYPE", "auto")
#     serialized_dir = f"{BASE_DIR}/serialized_model"
#     os.makedirs(serialized_dir, exist_ok=True)
#     serialized_uri = f"{serialized_dir}/model{'-%03d' if tensorizer_num_gpus > 1 else ''}.tensors"
    
#     tensorize_vllm_model(
#         EngineArgs(model=model_path, tensor_parallel_size=tensorizer_num_gpus, dtype=dtype),
#         TensorizerConfig(tensorizer_uri=serialized_uri)
#     )
#     logging.info("Successfully serialized model to %s", str(serialized_uri))
#     logging.info("Removing HF Model files after serialization")
#     rmtree("/".join(model_path.split("/")[:-2]))
#     return serialized_uri, tensorizer_num_gpus, dtype

if __name__ == "__main__":
    setup_env()
    cache_dir = os.getenv("HF_HOME")
    model_name, model_revision = os.getenv("MODEL_NAME"), os.getenv("MODEL_REVISION") or None
    tokenizer_name, tokenizer_revision = os.getenv("TOKENIZER_NAME") or model_name, os.getenv("TOKENIZER_REVISION") or model_revision
   
    model_path = download(model_name, model_revision, "model", cache_dir)   
  
    metadata = {
        "MODEL_NAME": model_path,
        "MODEL_REVISION": os.getenv("MODEL_REVISION"),
        "QUANTIZATION": os.getenv("QUANTIZATION"),
    }   
    
    # if os.getenv("TENSORIZE") == "1": TODO: Add back once tensorizer is ready
    #     serialized_uri, tensorizer_num_gpus, dtype = tensorize_model(model_path)
    #     metadata.update({
    #         "MODEL_NAME": serialized_uri,
    #         "TENSORIZER_URI": serialized_uri,
    #         "TENSOR_PARALLEL_SIZE": tensorizer_num_gpus,
    #         "DTYPE": dtype
    #     })
        
    tokenizer_path = download(tokenizer_name, tokenizer_revision, "tokenizer", cache_dir)
    metadata.update({
        "TOKENIZER_NAME": tokenizer_path,
        "TOKENIZER_REVISION": tokenizer_revision
    })
    
    with open(f"{BASE_DIR}/local_model_args.json", "w") as f:
        json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)