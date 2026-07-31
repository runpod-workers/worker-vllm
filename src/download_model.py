"""Download a model (and tokenizer) into the image at build time.

Used by the Dockerfile "Option 2" flow: when the MODEL_NAME build ARG is set,
the weights are fetched into the HF cache and the resolved snapshot paths are
written to LOCAL_MODEL_ARGS_PATH, which main.py reads at container start.
"""

import json
import logging
import os

from huggingface_hub import snapshot_download

LOCAL_MODEL_ARGS_PATH = "/local_model_args.json"


def snapshot(repo_id: str, revision: str | None = None) -> str:
    path = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=os.getenv("HF_HOME"),
        token=os.getenv("HF_TOKEN"),
    )
    logging.info("Downloaded %s to %s", repo_id, path)
    return path


def main() -> None:
    model_name = os.environ["MODEL_NAME"]
    model_revision = os.getenv("MODEL_REVISION") or None
    tokenizer_name = os.getenv("TOKENIZER_NAME") or model_name
    tokenizer_revision = os.getenv("TOKENIZER_REVISION") or model_revision

    metadata = {
        "MODEL_NAME": snapshot(model_name, model_revision),
        "TOKENIZER_NAME": snapshot(tokenizer_name, tokenizer_revision),
        "QUANTIZATION": os.getenv("QUANTIZATION"),
    }

    with open(LOCAL_MODEL_ARGS_PATH, "w") as f:
        json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)
    logging.info("Wrote %s", LOCAL_MODEL_ARGS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
