import argparse
from vllm import LLMEngine, SamplingParams, AsyncEngineArgs, utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--download_dir", type=str)
    parser.add_argument("--tokenizer", type=str, default=None)

    args = parser.parse_args()
    if not args.model or not args.download_dir:
        raise ValueError("Must specify model and download_dir")


    engine_args = AsyncEngineArgs(
    model=args.model,
    download_dir=args.download_dir,
    tokenizer=args.tokenizer,
    dtype="auto"
    )
    
    llm = LLMEngine.from_engine_args(engine_args)
