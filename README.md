<div align="center">

<h1> vLLM Serverless Endpoint Worker </h1>

[![CD | Docker-Build-Release](https://github.com/runpod-workers/worker-vllm/actions/workflows/docker-build-release.yml/badge.svg)](https://github.com/runpod-workers/worker-vllm/actions/workflows/docker-build-release.yml)

Deploy Blazing-fast LLMs powered by [vLLM](https://github.com/vllm-project/vllm) on RunPod Serverless in a few clicks.
</div>

### Worker vLLM 0.2.0 - What's New
- You no longer need a linux-based machine or NVIDIA GPUs to build the worker.
- Over 3x lighter Docker image size.
- OpenAI Chat Completion output format (optional to use).
- Extremely fast image build time.
- Docker Secrets-protected Hugging Face token support for building the image with a model baked in without exposing your token.
- Support for `n` and `best_of` sampling parameters, which allow you to generate multiple responses from a single prompt.
- New environment variables for various configuration.
- vLLM Version: 0.2.7

## Table of Contents
- [Setting up the Serverless Worker](#setting-up-the-serverless-worker)
  - [Option 1: Deploy Any Model Using Pre-Built Docker Image [**RECOMMENDED**]](#option-1-deploy-any-model-using-pre-built-docker-image-recommended)
    - [Prerequisites](#prerequisites)
    - [Environment Variables](#environment-variables)
  - [Option 2: Build Docker Image with Model Inside](#option-2-build-docker-image-with-model-inside)
    - [Arguments](#arguments)
    - [Example: Building an image with OpenChat-3.5](#example-building-an-image-with-openchat-35)
      - [(Optional) Including Huggingface Token](#optional-including-huggingface-token)
  - [Compatible Models](#compatible-models)
- [Usage](#usage)
    - [Endpoint Model Inputs](#endpoint-model-inputs)
    - [Text Input Formats](#text-input-formats)
      - [1. `prompt`](#1-prompt)
      - [2. `messages`](#2-messages)
    - [Sampling Parameters](#sampling-parameters)

## Setting up the Serverless Worker

### Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]

We now offer a pre-built Docker Image for the vLLM Worker that you can configure entirely with Environment Variables when creating the RunPod Serverless Endpoint:

<div align="center">

Stable Image: ```runpod/worker-vllm:0.2.0```

Development Image: ```runpod/worker-vllm:dev```

</div>

#### Prerequisites
- RunPod Account

#### Environment Variables

**Required**:
   - `MODEL_NAME`: Hugging Face Model Repository (e.g., `openchat/openchat-3.5-1210`).

**Optional**:
- Model Settings:
  - `MAX_MODEL_LENGTH`: Maximum number of tokens for the engine to be able to handle. (default: maximum supported by the model)
  - `MODEL_BASE_PATH`: Model storage directory (default: `/runpod-volume`).
  - `LOAD_FORMAT`: Format to load model in (default: `auto`).
  - `HF_TOKEN`: Hugging Face token for private and gated models (e.g., Llama, Falcon).
  - `QUANTIZATION`: AWQ (`awq`), SqueezeLLM (`squeezellm`) or GPTQ (`gptq`) Quantization. The specified Model Repo must be of a quantized model. (default: `None`)
  - `TRUST_REMOTE_CODE`: Trust remote code for Hugging Face (default: `0`)
  
- Tensor Parallelism:

  Note that the more GPUs you split a model's weights accross, the slower it will be due to inter-GPU communication overhead. If you can fit the model on a single GPU, it is recommended to do so. 
  - `USE_TENSOR_PARALLEL`: Enable (`1`) or disable (`0`) Tensor Parallelism. (default: `0`)
  - `TENSOR_PARALLEL_SIZE`: Number of GPUs to shard the model across (default: `1`).
  
- System Settings:
  - `GPU_MEMORY_UTILIZATION`: GPU VRAM utilization (default: `0.98`).
  - `MAX_PARALLEL_LOADING_WORKERS`: Maximum number of parallel workers for loading models (default: `number of available CPU cores`).


- Serverless Settings:
  - `MAX_CONCURRENCY`: Max concurrent requests. (default: `100`)
  - `DEFAULT_BATCH_SIZE`: Token streaming batch size (default: `30`). This reduces the number of HTTP calls, increasing speed 8-10x vs non-batching, matching non-streaming performance.
  - `ALLOW_OPENAI_FORMAT`: Whether to allow users to specify `use_openai_format` to get output in OpenAI format. (default: `1`)
  - `DISABLE_LOG_STATS`: Enable (`0`) or disable (`1`) vLLM stats logging.
  - `DISABLE_LOG_REQUESTS`: Enable (`0`) or disable (`1`) request logging.

### Option 2: Build Docker Image with Model Inside
To build an image with the model baked in, you must specify the following docker arguments when building the image.

#### Prerequisites
- RunPod Account
- Docker

#### Arguments:
- **Required**
  - `MODEL_NAME`
- **Optional**
  - `MODEL_BASE_PATH`: Defaults to `/runpod-volume` for network storage. Use `/models` or for local container storage.
  - `QUANTIZATION`
  - `WORKER_CUDA_VERSION`: `11.8.0` or `12.1.0` (default: `11.8.0` due to a small amount of workers not having CUDA 12.1 support yet. `12.1.0` is recommended for optimal performance).

For the remaining settings, you may apply them as environment variables when running the container. Supported environment variables are listed in the [Environment Variables](#environment-variables) section.

#### Example: Building an image with OpenChat-3.5
```bash
sudo docker build -t username/image:tag --build-arg MODEL_NAME="openchat/openchat_3.5" --build-arg MODEL_BASE_PATH="/models" .
```

##### (Optional) Including Huggingface Token
If the model you would like to deploy is private or gated, you will need to include it during build time as a Docker secret, which will protect it from being exposed in the image and on DockerHub.
1. Enable Docker BuildKit (required for secrets).
```bash
export DOCKER_BUILDKIT=1
```
2. Export your Hugging Face token as an environment variable
```bash
export HF_TOKEN="your_token_here"
```
2. Add the token as a secret when building
```bash
docker build -t username/image:tag --secret id=HF_TOKEN --build-arg MODEL_NAME="openchat/openchat_3.5" .
```

### Compatible Model Architectures
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- Mixtral (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, etc.)
- Phi (`microsoft/phi-1_5`, `microsoft/phi-2`, etc.)
- LLaMA & LLaMA-2 (`meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- Qwen2 (`Qwen/Qwen2-7B-beta`, `Qwen/Qwen-7B-Chat-beta`, etc.)
- StableLM(`stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc.)
- Yi (`01-ai/Yi-6B`, `01-ai/Yi-34B`, etc.)
- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)
- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- DeciLM (`Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.)
- Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)


## Usage
### Endpoint Model Inputs
You may either use a `prompt` or a list of `messages` as input. If you use `messages`, the model's chat template will be applied to the messages automatically, so the model must have one. If you use `prompt`, you may optionally apply the model's chat template to the prompt by setting `apply_chat_template` to `true`.
| Argument              | Type                 | Default            | Description                                                                                            |
|-----------------------|----------------------|--------------------|--------------------------------------------------------------------------------------------------------|
| `prompt`              | str                  |                    | Prompt string to generate text based on.                                                               |
| `messages`            | list[dict[str, str]] |                    | List of messages, which will automatically have the model's chat template applied. Overrides `prompt`. |
| `use_openai_format`   | bool                 | False              | Whether to return output in OpenAI format. `ALLOW_OPENAI_FORMAT` environment variable must be `1`, the input must be a `messages` list, and `stream` enabled.                                                              |
| `apply_chat_template` | bool                 | False              | Whether to apply the model's chat template to the `prompt`.                                            |
| `sampling_params`     | dict                 | {}                 | Sampling parameters to control the generation, like temperature, top_p, etc.                           |
| `stream`              | bool                 | False              | Whether to enable streaming of output. If True, responses are streamed as they are generated.          |
| `batch_size`          | int                  | DEFAULT_BATCH_SIZE | The number of tokens to stream every HTTP POST call.                                                   |

### Text Input Formats 
You may either use a `prompt` or a list of `messages` as input.
#### 1. `prompt` 
The prompt string can be any string, and the model's chat template will not be applied to it unless `apply_chat_template` is set to `true`, in which case it will be treated as a user message.

Example:
```json
"prompt": "..."
```
#### 2. `messages`
Your list can contain any number of messages, and each message can have any role from the following list:
- `user`
- `assistant`
- `system`

The model's chat template will be applied to the messages automatically, so the model must have one.

Example:
```json
"messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
```

### Sampling Parameters
| Argument                        | Type                        | Default | Description                                                                                                                                                                                   |
|---------------------------------|-----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `n`                             | int                         | 1       | Number of output sequences generated from the prompt. The top `n` sequences are returned.                                                                                                      |
| `best_of`                       | Optional[int]               | `n`    | Number of output sequences generated from the prompt. The top `n` sequences are returned from these `best_of` sequences. Must be ≥ `n`. Treated as beam width in beam search. Default is `n`. |
| `presence_penalty`              | float                       | 0.0     | Penalizes new tokens based on their presence in the generated text so far. Values > 0 encourage new tokens, values < 0 encourage repetition.                                                  |
| `frequency_penalty`             | float                       | 0.0     | Penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage new tokens, values < 0 encourage repetition.                                                 |
| `repetition_penalty`            | float                       | 1.0     | Penalizes new tokens based on their appearance in the prompt and generated text. Values > 1 encourage new tokens, values < 1 encourage repetition.                                            |
| `temperature`                   | float                       | 1.0     | Controls the randomness of sampling. Lower values make it more deterministic, higher values make it more random. Zero means greedy sampling.                                                  |
| `top_p`                         | float                       | 1.0     | Controls the cumulative probability of top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.                                                                            |
| `top_k`                         | int                         | -1      | Controls the number of top tokens to consider. Set to -1 to consider all tokens.                                                                                                              |
| `min_p`                         | float                       | 0.0     | Represents the minimum probability for a token to be considered, relative to the most likely token. Must be in [0, 1]. Set to 0 to disable.                                                   |
| `use_beam_search`               | bool                        | False   | Whether to use beam search instead of sampling.                                                                                                                                               |
| `length_penalty`                | float                       | 1.0     | Penalizes sequences based on their length. Used in beam search.                                                                                                                               |
| `early_stopping`                | Union[bool, str]            | False   | Controls stopping condition in beam search. Can be `True`, `False`, or `"never"`.                                                                                                             |
| `stop`                          | Union[None, str, List[str]] | None    | List of strings that stop generation when produced. Output will not contain these strings.                                                                                                    |
| `stop_token_ids`                | Optional[List[int]]         | None    | List of token IDs that stop generation when produced. Output contains these tokens unless they are special tokens.                                                                            |
| `ignore_eos`                    | bool                        | False   | Whether to ignore the End-Of-Sequence token and continue generating tokens after its generation.                                                                                              |
| `max_tokens`                    | int                         | 16      | Maximum number of tokens to generate per output sequence.                                                                                                                                     |
| `skip_special_tokens`           | bool                        | True    | Whether to skip special tokens in the output.                                                                                                                                                 |
| `spaces_between_special_tokens` | bool                        | True    | Whether to add spaces between special tokens in the output.                                                                                                                                   |
