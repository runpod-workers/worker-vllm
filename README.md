<div align="center">

<h1> vLLM Serverless Endpoint Worker </h1>

[![CD | Docker-Build-Release](https://github.com/runpod-workers/worker-vllm/actions/workflows/docker-build-release.yml/badge.svg)](https://github.com/runpod-workers/worker-vllm/actions/workflows/docker-build-release.yml)

Deploy Blazing-fast LLMs powered by [vLLM](https://github.com/vllm-project/vllm) on RunPod Serverless in a few clicks.
</div>

### Worker vLLM 0.3.0: What's New since 0.2.0:
- **🚀 Full OpenAI Compatibility 🚀**

  You may now use your deployment with any OpenAI Codebase by changing **only 3 lines** in total. The supported routes are <ins>Chat Completions</ins>, <ins>Completions</ins>, and <ins>Models</ins> - with both streaming and non-streaming.
- **Dynamic Batch Size** - time-to-first token as fast no batching, while maintaining the performance of batched token streaming throughout the request.
- **Custom chat templates** that you can specify as an environment variable.
- Fixed Tensor Parallelism, baking model into images, and more bugs.

## Table of Contents
- [Setting up the Serverless Worker](#setting-up-the-serverless-worker)
  - [Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]](#option-1-deploy-any-model-using-pre-built-docker-image-recommended)
    - [Prerequisites](#prerequisites)
    - [Environment Variables](#environment-variables)
  - [Option 2: Build Docker Image with Model Inside](#option-2-build-docker-image-with-model-inside)
    - [Prerequisites](#prerequisites-1)
    - [Arguments](#arguments)
    - [Example: Building an image with OpenChat-3.5](#example-building-an-image-with-openchat-35)
      - [(Optional) Including Huggingface Token](#optional-including-huggingface-token)
  - [Compatible Model Architectures](#compatible-model-architectures)
- [Usage: OpenAI Compatibility](#usage-openai-compatibility)
    - [Modifying your OpenAI Codebase to use your deployed vLLM Worker](#modifying-your-openai-codebase-to-use-your-deployed-vllm-worker)
    - [OpenAI Request Input Parameters](#openai-request-input-parameters)
      - [Chat Completions](#chat-completions)
      - [Completions](#completions)
    - [Examples: Using your RunPod endpoint with OpenAI](#examples-using-your-runpod-endpoint-with-openai)
- [Usage: standard](#non-openai-usage)
  - [Input Request Parameters](#input-request-parameters)
    - [Text Input Formats](#text-input-formats)
    - [Sampling Parameters](#sampling-parameters)

# Setting up the Serverless Worker

### Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]
> [!TIP]
> This is the recommended way to deploy your model, as it does not require you to build a Docker image, upload heavy models to DockerHub and wait for workers to download them. Instead, use this option to deploy your model in a few clicks. For even more convenience, attach a network storage volume to your Endpoint, which will download the model once and share it across all workers.

We now offer a pre-built Docker Image for the vLLM Worker that you can configure entirely with Environment Variables when creating the RunPod Serverless Endpoint:

<div align="center">

Stable Image: ```runpod/worker-vllm:0.2.2```

Development Image: ```runpod/worker-vllm:dev```

</div>

#### Prerequisites
- RunPod Account

#### Environment Variables

**Required**:
   - `MODEL_NAME`: Hugging Face Model Repository (e.g., `openchat/openchat-3.5-1210`).

**Optional**:
- LLM Settings:
  - `MODEL_REVISION`: Model revision to load (default: `None`).
  - `MAX_MODEL_LENGTH`: Maximum number of tokens for the engine to be able to handle. (default: maximum supported by the model)
  - `BASE_PATH`: Storage directory where huggingface cache and model will be located. (default: `/runpod-volume`, which will utilize network storage if you attach it or create a local directory within the image if you don't)
  - `LOAD_FORMAT`: Format to load model in (default: `auto`).
  - `HF_TOKEN`: Hugging Face token for private and gated models (e.g., Llama, Falcon).
  - `QUANTIZATION`: AWQ (`awq`), SqueezeLLM (`squeezellm`) or GPTQ (`gptq`) Quantization. The specified Model Repo must be of a quantized model. (default: `None`)
  - `TRUST_REMOTE_CODE`: Trust remote code for Hugging Face (default: `0`)
  
- Tokenizer Settings:
  - `TOKENIZER_NAME`: Tokenizer repository if you would like to use a different tokenizer than the one that comes with the model. (default: `None`, which uses the model's tokenizer)
  - `TOKENIZER_REVISION`: Tokenizer revision to load (default: `None`).
  - `CUSTOM_CHAT_TEMPLATE`: Custom chat jinja template, read more about Hugging Face chat templates [here](https://huggingface.co/docs/transformers/chat_templating). (default: `None`) 

- Tensor Parallelism:
  Note that the more GPUs you split a model's weights across, the slower it will be due to inter-GPU communication overhead. If you can fit the model on a single GPU, it is recommended to do so. 
  - `TENSOR_PARALLEL_SIZE`: Number of GPUs to shard the model across (default: `1`).
  - If you are having issues loading your model with Tensor Parallelism, try decreasing `VLLM_CPU_FRACTION` (default: `1`).
  
- System Settings:
  - `GPU_MEMORY_UTILIZATION`: GPU VRAM utilization (default: `0.98`).
  - `MAX_PARALLEL_LOADING_WORKERS`: Maximum number of parallel workers for loading models, for non-Tensor Parallel only. (default: `number of available CPU cores` if `TENSOR_PARALLEL_SIZE` is `1`, otherwise `None`).

- Streaming Batch Size:
  - `DEFAULT_BATCH_SIZE`: Token streaming batch size (default: `50`). This reduces the number of HTTP calls, increasing speed 8-10x vs non-batching, matching non-streaming performance.
  - Dynamic Batch Size:
    - `DEFAULT_MIN_BATCH_SIZE`: the batch size for the first request (default: `1`).
    - `DEFAULT_BATCH_SIZE_GROWTH_FACTOR`: the growth factor for the dynamic batch size (default: `3`). 
    
    > The way this works is that the first request will have a batch size of `DEFAULT_MIN_BATCH_SIZE`, and each subsequent request will have a batch size of `previous_batch_size * DEFAULT_BATCH_SIZE_GROWTH_FACTOR`. This will continue until the batch size reaches `DEFAULT_BATCH_SIZE`. E.g. for the default values, the batch sizes will be `1, 3, 9, 27, 50, 50, 50, ...`. You can also specify this per request, with inputs `max_batch_size`, `min_batch_size`, and `batch_size_growth_factor`. 

- OpenAI Settings:
  - `RAW_OPENAI_OUTPUT`: Enable (`1`) or disable (`0`) raw OpenAI SSE format string output when streaming(default: `1`). **Required** to be enabled (default) for OpenAI compatibility.

- Serverless Settings:
  - `MAX_CONCURRENCY`: Max concurrent requests. (default: `100`)
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
  - `MODEL_REVISION`: Model revision to load (default: `main`).
  - `BASE_PATH`: Storage directory where huggingface cache and model will be located. (default: `/runpod-volume`, which will utilize network storage if you attach it or create a local directory within the image if you don't. If your intention is to bake the model into the image, you should set this to something like `/models` to make sure there are no issues if you were to accidentally attach network storage.)
  - `QUANTIZATION`
  - `WORKER_CUDA_VERSION`: `11.8.0` or `12.1.0` (default: `11.8.0` due to a small number of workers not having CUDA 12.1 support yet. `12.1.0` is recommended for optimal performance).
  - `TOKENIZER_NAME`: Tokenizer repository if you would like to use a different tokenizer than the one that comes with the model. (default: `None`, which uses the model's tokenizer)
  - `TOKENIZER_REVISION`: Tokenizer revision to load (default: `main`).

For the remaining settings, you may apply them as environment variables when running the container. Supported environment variables are listed in the [Environment Variables](#environment-variables) section.

#### Example: Building an image with OpenChat-3.5
```bash
sudo docker build -t username/image:tag --build-arg MODEL_NAME="openchat/openchat_3.5" --build-arg BASE_PATH="/models" .
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


# Usage: OpenAI Compatibility
The vLLM Worker is fully compatible with OpenAI's API, and you can use it with any OpenAI Codebase by changing only 3 lines in total. The supported routes are <ins>Chat Completions</ins>, <ins>Completions</ins> and <ins>Models</ins> - with both streaming and non-streaming.

## Modifying your OpenAI Codebase to use your deployed vLLM Worker 
**Python** (similar to Node.js, etc.):
1. When initializing the OpenAI Client in your code, change the `api_key` to your RunPod API Key and the `base_url` to your RunPod Serverless Endpoint URL in the following format: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1`, filling in your deployed endpoint ID.
    
    - Before:
    ```python
    import openai
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    ```
    - After:
    ```python
    import openai
    client = OpenAI(api_key=os.environ.get("RUNPOD_API_KEY"), base_url="https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1")
    ```
2. Change the `model` parameter to your deployed model's name whenever using Completions or Chat Completions.
    - Before:
    ```python
    response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': 'Why is RunPod the best platform?'}
    ],
    temperature=0,
    max_tokens=100
    )
    ```
    - After:
    ```python
    response = client.chat.completions.create(
    model="<YOUR DEPLOYED MODEL REPO/NAME>",
    messages=[
        {'role': 'user', 'content': 'Why is RunPod the best platform?'}
    ],
    temperature=0,
    max_tokens=100
    )
    ```

**Using http requests**:
1. Change the `Authorization` header to your RunPod API Key and the `url` to your RunPod Serverless Endpoint URL in the following format: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1`
    - Before:
    ```bash
    curl https://api.openai.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
    "model": "gpt-4",
    "messages": [
      {
        "role": "user",
        "content": "Why is RunPod the best platform?"
      }
    ],
    "temperature": 0,
    "max_tokens": 100
    }'
    ```
    - After:
    ```bash
    curl https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <YOUR OPENAI API KEY>" \
    -d '{
    "model": "<YOUR DEPLOYED MODEL REPO/NAME>",
    "messages": [
      {
        "role": "user",
        "content": "Why is RunPod the best platform?"
      }
    ],
    "temperature": 0,
    "max_tokens": 100
    }'
    ```

## OpenAI Request Input Parameters:

When using the chat completion feature of the vLLM Serverless Endpoint Worker, you can customize your requests with the following parameters:

### Chat Completions
<details>
  <summary>Click to expand table</summary>

  | Parameter                      | Type                             | Default Value | Description                                                                                                                                                                                                                                                                                           |
  |--------------------------------|----------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | `messages`                     | Union[str, List[Dict[str, str]]] |               | List of messages, where each message is a dictionary with a `role` and `content`. The model's chat template will be applied to the messages automatically, so the model must have one or it should be specified as `CUSTOM_CHAT_TEMPLATE` env var.                                                                                                                        |
  | `model`                        | str                              |               | The model repo that you've deployed on your RunPod Serverless Endpoint. If you are unsure what the name is or are baking the model in, use the guide to get the list of available models in the **Examples: Using your RunPod endpoint with OpenAI** section                                                                                                                                                                                                                                                    |
  | `temperature`                  | Optional[float]                  | 0.7           | Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.                                                                                                                                                                                                                               |
  | `top_p`                        | Optional[float]                  | 1.0           | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. |
  | `n`                            | Optional[int]                    | 1             | Number of output sequences to return for the given prompt. |
  | `max_tokens`                   | Optional[int]                    | None          | Maximum number of tokens to generate per output sequence. |
  | `seed`                         | Optional[int]                    | None          | Random seed to use for the generation. |
  | `stop`                         | Optional[Union[str, List[str]]]  | list          | List of strings that stop the generation when they are generated. The returned output will not contain the stop strings. |
  | `stream`                       | Optional[bool]                   | False         |           Whether to stream or not                                                                                                                                                                                                                                                                                            |
  | `presence_penalty`             | Optional[float]                  | 0.0           | Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens. |
  | `frequency_penalty`            | Optional[float]                  | 0.0           | Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens. |
  | `logit_bias`                   | Optional[Dict[str, float]]       | None          |           Unsupported by vLLM                                                                                                                                                                                                                                                                                         |
  | `user`                         | Optional[str]                    | None          |               Unsupported by vLLM                                                                                                                                                                                                                                                                                                                                                                         |
  Additional parameters supported by vLLM:
  | `best_of`                      | Optional[int]                    | None          | Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`. |
  | `top_k`                        | Optional[int]                    | -1            | Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens. |
  | `ignore_eos`                   | Optional[bool]                   | False         | Whether to ignore the EOS token and continue generating tokens after the EOS token is generated. |
  | `use_beam_search`              | Optional[bool]                   | False         | Whether to use beam search instead of sampling. |
  | `stop_token_ids`               | Optional[List[int]]              | list          | List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens. |
  | `skip_special_tokens`          | Optional[bool]                   | True          | Whether to skip special tokens in the output. |
  | `spaces_between_special_tokens`| Optional[bool]                   | True          | Whether to add spaces between special tokens in the output. Defaults to True. |
  | `add_generation_prompt`        | Optional[bool]                   | True          |       Read more [here](https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts)                                                                                                                                                                                                                                                                                                |
  | `echo`                         | Optional[bool]                   | False         |      Echo back the prompt in addition to the completion                                                                                                                                                                                                                                                                                                 |
  | `repetition_penalty`           | Optional[float]                  | 1.0           | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens. |
  | `min_p`                        | Optional[float]                  | 0.0           | Float that represents the minimum probability for a token to |
  | `length_penalty`               | Optional[float]                  | 1.0           | Float that penalizes sequences based on their length. Used in beam search.. |
  | `include_stop_str_in_output`   | Optional[bool]                   | False         | Whether to include the stop strings in output text. Defaults to False.|
</details>

### Completions
<details>
  <summary>Click to expand table</summary>

  | Parameter                      | Type                             | Default Value | Description                                                                                                                                                                                                                                                                                           |
  |--------------------------------|----------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | `model`                        | str                              |               | The model repo that you've deployed on your RunPod Serverless Endpoint. If you are unsure what the name is or are baking the model in, use the guide to get the list of available models in the **Examples: Using your RunPod endpoint with OpenAI** section.                                                                                                                                                                                                                                                    |
  | `prompt`                       | Union[List[int], List[List[int]], str, List[str]] | | A string, array of strings, array of tokens, or array of token arrays to be used as the input for the model. |
  | `suffix`                       | Optional[str]                    | None          | A string to be appended to the end of the generated text. |
  | `max_tokens`                   | Optional[int]                    | 16            | Maximum number of tokens to generate per output sequence. |
  | `temperature`                  | Optional[float]                  | 1.0           | Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.                                                                                                                                                                                                                               |
  | `top_p`                        | Optional[float]                  | 1.0           | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens. |
  | `n`                            | Optional[int]                    | 1             | Number of output sequences to return for the given prompt. |
  | `stream`                       | Optional[bool]                   | False         | Whether to stream the output. |
  | `logprobs`                     | Optional[int]                    | None          | Number of log probabilities to return per output token. |
  | `echo`                         | Optional[bool]                   | False         | Whether to echo back the prompt in addition to the completion. |
  | `stop`                         | Optional[Union[str, List[str]]]  | list          | List of strings that stop the generation when they are generated. The returned output will not contain the stop strings. |
  | `seed`                         | Optional[int]                    | None          | Random seed to use for the generation. |
  | `presence_penalty`             | Optional[float]                  | 0.0           | Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens. |
  | `frequency_penalty`            | Optional[float]                  | 0.0           | Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens. |
  | `best_of`                      | Optional[int]                    | None          | Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This parameter influences the diversity of the output. |
  | `logit_bias`                   | Optional[Dict[str, float]]       | None          | Dictionary of token IDs to biases. |
  | `user`                         | Optional[str]                    | None          | User identifier for personalizing responses. (Unsupported by vLLM) |
  Additional parameters supported by vLLM:
  | `top_k`                        | Optional[int]                    | -1            | Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens. |
  | `ignore_eos`                   | Optional[bool]                   | False         | Whether to ignore the End Of Sentence token and continue generating tokens after the EOS token is generated. |
  | `use_beam_search`              | Optional[bool]                   | False         | Whether to use beam search instead of sampling for generating outputs. |
  | `stop_token_ids`               | Optional[List[int]]              | list          | List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens. |
  | `skip_special_tokens`          | Optional[bool]                   | True          | Whether to skip special tokens in the output. |
  | `spaces_between_special_tokens`| Optional[bool]                   | True          | Whether to add spaces between special tokens in the output. Defaults to True. |
  | `repetition_penalty`           | Optional[float]                  | 1.0           | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens. |
  | `min_p`                        | Optional[float]                  | 0.0           | Float that represents the minimum probability for a token to be considered, relative to the most likely token. Must be in [0, 1]. Set to 0 to disable. |
  | `length_penalty`               | Optional[float]                  | 1.0           | Float that penalizes sequences based on their length. Used in beam search. |
  | `include_stop_str_in_output`   | Optional[bool]                   | False         | Whether to include the stop strings in output text. Defaults to False. |
</details>

## Examples: Using your RunPod endpoint with OpenAI 

First, initialize the OpenAI Client with your RunPod API Key and Endpoint URL:
```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your RunPod API Key and Endpoint URL
client = OpenAI(api_key=os.environ.get("RUNPOD_API_KEY"), base_url="https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1")
```

### Chat Completions:
This is the format used for GPT-4 and focused on instruction-following and chat. Examples of Open Source chat/instruct models include `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `openchat/openchat-3.5-0106`, `NousResearch/Nous-Hermes-2-Mistral-7B-DPO` and more. However, if your model is a completion-style model with no chat/instruct fine-tune and/or does not have a chat template, you can still use this if you provide a chat template with the environment variable `CUSTOM_CHAT_TEMPLATE`.
- **Streaming**:
  ```python
  # Create a chat completion stream
  response_stream = client.chat.completions.create(
      model="<YOUR DEPLOYED MODEL REPO/NAME>",
      messages=[
          {'role': 'user', 'content': 'Why is RunPod the best platform?'}
      ],
      temperature=0,
      max_tokens=100,
      stream=True
  )
  # Stream the response
  for response in response_stream:
      print(chunk.choices[0].delta.content or "", end="", flush=True)
  ```
- **Non-Streaming**:
  ```python
  # Create a chat completion
  response = client.chat.completions.create(
      model="<YOUR DEPLOYED MODEL REPO/NAME>",
      messages=[
          {'role': 'user', 'content': 'Why is RunPod the best platform?'}
      ],
      temperature=0,
      max_tokens=100
  )
  # Print the response
  print(response.choices[0].message.content)
  ```


### Completions:
This is the format used for models like GPT-3 and is meant for completing the text you provide. Instead of responding to your message, it will try to complete it. Examples of Open Source completions models include `meta-llama/Llama-2-7b-hf`, `mistralai/Mixtral-8x7B-v0.1`, `Qwen/Qwen-72B`, and more. However, you can use any model with this format.
- **Streaming**:
  ```python
  # Create a completion stream
  response_stream = client.completions.create(
      model="<YOUR DEPLOYED MODEL REPO/NAME>",
      prompt="Runpod is the best platform because",
      temperature=0,
      max_tokens=100,
      stream=True
  )
  # Stream the response
  for response in response_stream:
      print(response.choices[0].text or "", end="", flush=True)
  ```
- **Non-Streaming**:
  ```python
  # Create a completion
  response = client.completions.create(
      model="<YOUR DEPLOYED MODEL REPO/NAME>",
      prompt="Runpod is the best platform because",
      temperature=0,
      max_tokens=100
  )
  # Print the response
  print(response.choices[0].text)
  ```

### Getting a list of names for available models:
In the case of baking the model into the image, sometimes the repo may not be accepted as the `model` in the request. In this case, you can list the available models as shown below and use that name. 
```python
models_response = client.models.list()
list_of_models = [model.id for model in models_response]
print(list_of_models)
```

## Non-OpenAI Usage
### Input Request Parameters

<details>
  <summary>Click to expand table</summary>
    
  You may either use a `prompt` or a list of `messages` as input. If you use `messages`, the model's chat template will be applied to the messages automatically, so the model must have one. If you use `prompt`, you may optionally apply the model's chat template to the prompt by setting `apply_chat_template` to `true`.
  | Argument              | Type                 | Default            | Description                                                                                            |
  |-----------------------|----------------------|--------------------|--------------------------------------------------------------------------------------------------------|
  | `prompt`              | str                  |                    | Prompt string to generate text based on.                                                               |
  | `messages`            | list[dict[str, str]] |                    | List of messages, which will automatically have the model's chat template applied. Overrides `prompt`. |
  | `apply_chat_template` | bool                 | False              | Whether to apply the model's chat template to the `prompt`.                                            |
  | `sampling_params`     | dict                 | {}                 | Sampling parameters to control the generation, like temperature, top_p, etc.                           |
  | `stream`              | bool                 | False              | Whether to enable streaming of output. If True, responses are streamed as they are generated.          |
  | `max_batch_size`          | int                  | env var `DEFAULT_BATCH_SIZE` | The maximum number of tokens to stream every HTTP POST call.                                                   |
  | `min_batch_size`          | int                  | env var `DEFAULT_MIN_BATCH_SIZE` | The minimum number of tokens to stream every HTTP POST call.                                           |
  | `batch_size_growth_factor` | int                  | env var `DEFAULT_BATCH_SIZE_GROWTH_FACTOR` | The growth factor by which `min_batch_size` will be multiplied for each call until `max_batch_size` is reached.           |
</details>

#### Text Input Formats 
You may either use a `prompt` or a list of `messages` as input.
 1. `prompt` 
The prompt string can be any string, and the model's chat template will not be applied to it unless `apply_chat_template` is set to `true`, in which case it will be treated as a user message.

    Example:
    ```json
    "prompt": "..."
    ```
2. `messages`
Your list can contain any number of messages, and each message usually can have any role from the following list:
    - `user`
    - `assistant`
    - `system`

    However, some models may have different roles, so you should check the model's chat template to see which roles are required.

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

#### Sampling Parameters

Below are all available sampling parameters that you can specify in the `sampling_params` dictionary. If you do not specify any of these parameters, the default values will be used.
<details>
  <summary>Click to expand table</summary>

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
  | `stop`                          | Union[None, str, List[str]] | None    | List of strings that stop generation when produced. The output will not contain these strings.                                                                                                    |
  | `stop_token_ids`                | Optional[List[int]]         | None    | List of token IDs that stop generation when produced. Output contains these tokens unless they are special tokens.                                                                            |
  | `ignore_eos`                    | bool                        | False   | Whether to ignore the End-Of-Sequence token and continue generating tokens after its generation.                                                                                              |
  | `max_tokens`                    | int                         | 16      | Maximum number of tokens to generate per output sequence.                                                                                                                                     |
  | `skip_special_tokens`           | bool                        | True    | Whether to skip special tokens in the output.                                                                                                                                                 |
  | `spaces_between_special_tokens` | bool                        | True    | Whether to add spaces between special tokens in the output.                                                                                                                                   |

