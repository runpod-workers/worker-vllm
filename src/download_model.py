import os
import shutil
import logging
from huggingface_hub import snapshot_download
from vllm.model_executor.weight_utils import Disabledtqdm, prepare_hf_model_weights

if __name__ == "__main__":
    model = os.getenv("MODEL_NAME")
    download_dir = os.getenv("HF_HOME")
    tokenizer = os.getenv("TOKENIZER_NAME", model)
    model_revision = os.getenv("MODEL_REVISION", "main")
    tokenizer_revision = os.getenv("TOKENIZER_REVISION", "main")
    
    if not model or not download_dir:
        raise ValueError(f"Must specify model and download_dir. Model: {model}, download_dir: {download_dir}")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    logging.info(f"Downloading model {model} to {download_dir}")
        
    model_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path=model,
        cache_dir=download_dir,
        revision=model_revision,
    )
    
    model_extras_folder = snapshot_download(model,
                      allow_patterns=[
                          "*token*",
                          "*.json"
                          ],
                      cache_dir=download_dir + "/extras",
                      tqdm_class=Disabledtqdm,
                      revision=model_revision)
    
    # Move extras to hf_folder
    for f in os.listdir(model_extras_folder):
        shutil.move(model_extras_folder + "/" + f, model_folder + "/" + f)
    
    # Wrie hf_folder to file
    with open("/local_model_path.txt", "w") as f:
        f.write(model_folder) 
        
    logging.info(f"Finished downloading model {model} to {download_dir}")
    
    if tokenizer != model:
        logging.info(f"Downloading tokenizer {tokenizer} to {download_dir}")
        tokenizer_folder = snapshot_download(tokenizer,
                          cache_dir=download_dir,
                          tqdm_class=Disabledtqdm,
                          revision=tokenizer_revision)
        with open("/local_tokenizer_path.txt", "w") as f:
            f.write(tokenizer_folder) 
        
        logging.info(f"Finished downloading tokenizer {tokenizer} to {download_dir}")
    
  
    
