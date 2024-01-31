import os
import logging
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    model = os.getenv("MODEL_NAME")
    download_dir = os.getenv("HF_HOME")
    if not model or not download_dir:
        raise ValueError(f"Must specify model and download_dir. Model: {model}, download_dir: {download_dir}")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    logging.info(f"Downloading model {model} to {download_dir}")

    hf_folder = snapshot_download(
        model,
        local_dir=download_dir,
        cache_dir=download_dir,
        local_dir_use_symlinks=False,
    )
    
    logging.info(f"Finished downloading model {model} to {download_dir}")
    
    # Wrie hf_folder to file
    with open("/local_model_path.txt", "w") as f:
        f.write(hf_folder) 