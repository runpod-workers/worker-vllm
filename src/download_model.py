import argparse
import os
from vllm.model_executor.weight_utils import prepare_hf_model_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--download_dir", type=str)

    args = parser.parse_args()
    if not args.model or not args.download_dir:
        raise ValueError("Must specify model and download_dir")

    if not os.path.exists(args.download_dir):
        os.makedirs(args.download_dir)

    prepare_hf_model_weights(
        model_name_or_path = args.model,
        cache_dir=args.download_dir,
    )