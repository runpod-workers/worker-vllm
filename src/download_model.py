import os
import logging
from vllm.model_executor.weight_utils import prepare_hf_model_weights

if __name__ == "__main__":
    model = os.getenv("MODEL_NAME")
    download_dir = os.getenv("HF_HOME")
    if not model or not download_dir:
        raise ValueError(f"Must specify model and download_dir. Model: {model}, download_dir: {download_dir}")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    logging.info(f"Downloading model {model} to {download_dir}")
        
    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path=model,
        cache_dir=download_dir,
    )
    
    logging.info(f"Finished downloading model {model} to {download_dir}")
    
    # Wrie hf_folder to file
    with open("/local_model_path.txt", "w") as f:
        f.write(hf_folder) 