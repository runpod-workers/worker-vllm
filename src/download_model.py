import os
import argparse
from huggingface_hub import snapshot_download


MODEL_NAME = os.environ.get('MODEL_NAME')
MODEL_REVISION = os.environ.get('MODEL_REVISION', "main")
MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/runpod-volume/')
HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=MODEL_NAME)
    parser.add_argument('--model_revision', type=str, default=MODEL_REVISION)
    parser.add_argument('--model_base_path', type=str, default=MODEL_BASE_PATH)

    args = parser.parse_args()

    snapshot_download(
        args.model_name,
        revision=args.model_revision,
        local_dir=f"{args.model_base_path}{args.model_name.split('/')[1]}",
    )
