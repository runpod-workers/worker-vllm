import os
import shutil
from huggingface_hub import snapshot_download
from vllm.model_executor.weight_utils import prepare_hf_model_weights, Disabledtqdm

def download_extras_or_tokenizer(model_name, cache_dir, revision, extras=False):
    """Download model or tokenizer and prepare its weights, returning the local folder path."""
    pattern = ["*token*", "*.json"] if extras else None
    extra_dir = "/extras" if extras else ""
    folder = snapshot_download(
        model_name,
        cache_dir=cache_dir + extra_dir,
        revision=revision,
        tqdm_class=Disabledtqdm,
        allow_patterns=pattern if extras else None,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt"] if not extras else None
    )
    return folder

def move_files(src_dir, dest_dir):
    """Move files from source to destination directory."""
    for f in os.listdir(src_dir):
        src_path = os.path.join(src_dir, f)  
        dst_path = os.path.join(dest_dir, f)         
        shutil.copy2(src_path, dst_path)
        os.remove(src_path)
        
if __name__ == "__main__":
    model, download_dir = os.getenv("MODEL_NAME"), os.getenv("HF_HOME")
    tokenizer = os.getenv("TOKENIZER_NAME", model)
    revisions = {
        "model": os.getenv("MODEL_REVISION") or None,
        "tokenizer": os.getenv("TOKENIZER_REVISION") or None
    }

    if not model or not download_dir:
        raise ValueError(f"Must specify model and download_dir. Model: {model}, download_dir: {download_dir}")

    os.makedirs(download_dir, exist_ok=True)
    model_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(model_name_or_path=model, revision=revisions["model"], cache_dir=download_dir)
    model_extras_folder = download_extras_or_tokenizer(model, download_dir, revisions["model"], extras=True)
    move_files(model_extras_folder, model_folder)

    with open("/local_model_path.txt", "w") as f:
        f.write(model_folder)

    if tokenizer != model:
        tokenizer_folder = download_extras_or_tokenizer(tokenizer, download_dir, revisions["tokenizer"])
        with open("/local_tokenizer_path.txt", "w") as f:
            f.write(tokenizer_folder)
