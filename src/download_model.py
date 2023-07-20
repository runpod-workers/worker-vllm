import os
from huggingface_hub import snapshot_download

# Get the hugging face token
HUGGING_FACE_HUB_TOKEN = os.environ['HUGGING_FACE_HUB_TOKEN']
DOWNLOAD_7B_MODEL = os.environ.get('DOWNLOAD_7B_MODEL', None)
DOWNLOAD_13B_MODEL = os.environ.get('DOWNLOAD_13B_MODEL', None)

# Download the 7B
if HUGGING_FACE_HUB_TOKEN and DOWNLOAD_7B_MODEL:
    snapshot_download(
            "meta-llama/Llama-2-7b-chat-hf",
            local_dir="/model/Llama-2-7b-chat-hf",
            token=HUGGING_FACE_HUB_TOKEN
    )

# Download the 13B
if HUGGING_FACE_HUB_TOKEN and DOWNLOAD_13B_MODEL:
    snapshot_download(
            "meta-llama/Llama-2-13b-chat-hf",
            local_dir="/model/Llama-2-13b-chat-hf",
            token=HUGGING_FACE_HUB_TOKEN
    )
