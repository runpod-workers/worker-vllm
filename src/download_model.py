import os
from huggingface_hub import snapshot_download

# Get the hugging face token
HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN', None)
MODEL = os.environ.get('MODEL', None)

# Download the model from hugging face
if HUGGING_FACE_HUB_TOKEN and MODEL:
    snapshot_download(
        MODEL,
        local_dir="/model/{}".format(MODEL),
        token=HUGGING_FACE_HUB_TOKEN
    )
