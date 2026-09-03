<div align="center">

# OpenAI-Compatible vLLM Serverless Endpoint Worker

Deploy OpenAI-Compatible Blazing-Fast LLM Endpoints powered by the [vLLM](https://github.com/vllm-project/vllm) Inference Engine on Runpod Serverless with just a few clicks.

</div>

![vLLM worker banner](https://image.runpod.ai/preview/vllm/vllm-banner.png)

Current vLLM version: [0.28.0](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)


> Want a **load balancing** endpoint (direct HTTP, no job queue)? You don't need this worker — deploy the official vLLM image as-is. See [Option 3: Load Balancing with the vLLM Image](#option-3-load-balancing-with-the-vllm-image).

## Table of Contents

- [Setting up the Serverless Worker](#setting-up-the-serverless-worker)
  - [Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]](#option-1-deploy-any-model-using-pre-built-docker-image-recommended)
    - [Configuration](#configuration)
  - [Option 2: Build Docker Image with Model Inside](#option-2-build-docker-image-with-model-inside)
    - [Prerequisites](#prerequisites)
    - [Arguments](#arguments)
    - [Example: Building an image with OpenChat-3.5](#example-building-an-image-with-openchat-35)
      - [(Optional) Including Huggingface Token](#optional-including-huggingface-token)
  - [Option 3: Load Balancing with the vLLM Image](#option-3-load-balancing-with-the-vllm-image)
  - [Compatible Model Architectures](#compatible-model-architectures)
- [Usage: OpenAI Compatibility](#usage-openai-compatibility)
  - [Modifying your OpenAI Codebase to use your deployed vLLM Worker](#modifying-your-openai-codebase-to-use-your-deployed-vllm-worker)
  - [OpenAI Request Input Parameters](#openai-request-input-parameters)
  - [Chat Completions [RECOMMENDED]](#chat-completions-recommended)
  - [Examples: Using your Runpod endpoint with OpenAI](#examples-using-your-runpod-endpoint-with-openai)
    - [Chat Completions](#chat-completions)
    - [Getting a list of names for available models](#getting-a-list-of-names-for-available-models)
    - [OpenAI Responses API](#openai-responses-api)
    - [Anthropic Messages API](#anthropic-messages-api)
- [Usage: Standard (Non-OpenAI)](#usage-standard-non-openai)
  - [Request Input Parameters](#request-input-parameters)
  - [Sampling Parameters](#sampling-parameters)
    - [Text Input Formats](#text-input-formats)

# Setting up the Serverless Worker

## Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]

**🚀 Deploy Guide**: Follow our [step-by-step deployment guide](https://docs.runpod.io/serverless/vllm/get-started) to deploy using the Runpod Console.

**📦 Docker Image**: `runpod/worker-v1-vllm:<version>`

- **Available Versions**: See [GitHub Releases](https://github.com/runpod-workers/worker-vllm/releases)
- **CUDA Compatibility**: Inherits the CUDA runtime of the [`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai) base image used at build time.

### Configuration

Configure worker-vllm using environment variables:

| Environment Variable                | Description                                       | Default             | Options                                                            |
| ----------------------------------- | ------------------------------------------------- | ------------------- | ------------------------------------------------------------------ |
| `MODEL_NAME`                        | Path of the model weights                         |                     | Local folder or Hugging Face repo ID                               |
| `HF_TOKEN`                          | HuggingFace access token for gated/private models |                     | Your HuggingFace access token                                      |
| `MAX_MODEL_LEN`                     | Model's maximum context length                    |                     | Integer (e.g., 4096)                                               |
| `QUANTIZATION`                      | Quantization method                               |                     | "awq", "gptq", "squeezellm", "bitsandbytes"                        |
| `TENSOR_PARALLEL_SIZE`              | Number of GPUs                                    | 1                   | Integer                                                            |
| `GPU_MEMORY_UTILIZATION`            | Fraction of GPU memory to use                     | 0.9                 | Float between 0.0 and 1.0                                          |
| `MAX_NUM_SEQS`                      | Maximum number of sequences per iteration         | 256                 | Integer                                                            |
| `CUSTOM_CHAT_TEMPLATE`              | Custom chat template override                     |                     | Jinja2 template string                                             |
| `ENABLE_AUTO_TOOL_CHOICE`           | Enable automatic tool selection                   | false               | boolean (true or false)                                            |
| `TOOL_CALL_PARSER`                  | Parser for tool calls                             |                     | "mistral", "hermes", "llama3_json", "granite", "deepseek_v3", etc. |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | Override served model name in API                 |                     | String                                                             |
| `MAX_CONCURRENCY`                   | Maximum concurrent requests                       | 30                  | Integer                                                            |

**Pass any vLLM engine arg** as an environment variable: the worker translates env vars into `vllm serve` CLI flags. Any env var whose name is the UPPERCASED form of a `vllm serve` flag is applied automatically:

| Environment Variable      | vLLM CLI Flag            | Example Value |
| ------------------------- | ------------------------ | ------------- |
| `MAX_MODEL_LEN`           | `--max-model-len`        | `4096`        |
| `ENFORCE_EAGER`           | `--enforce-eager`        | `true`        |
| `ENABLE_CHUNKED_PREFILL`  | `--enable-chunked-prefill` | `true`      |
| `SPECULATIVE_CONFIG`      | `--speculative-config`   | `'{"model": "org/draft", "num_speculative_tokens": 3}'` |

Backward-compat aliases are also honored: `MODEL_NAME` (`--model`), `MODEL_REVISION` (`--revision`), `TOKENIZER_NAME` (`--tokenizer`), `OPENAI_SERVED_MODEL_NAME_OVERRIDE` (`--served-model-name`), `CUSTOM_CHAT_TEMPLATE` (`--chat-template`).

Values are passed straight through to vLLM (ints, floats, JSON blobs all work). For anything not recognized — e.g. a flag newer than the worker's list — use the ultimate escape hatch, whose contents are appended verbatim to the `vllm serve` command (and override earlier settings):

```bash
VLLM_EXTRA_ARGS="--override-generation-config '{\"max_new_tokens\": 512}' --uvicorn-log-level warning"
```

### Configuration File (config.yaml)

As an alternative to environment variables, you can supply a complete `vllm serve` `--config` file using the CLI key names (hyphens or underscores both work):

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
max-model-len: 8192
gpu-memory-utilization: 0.90
quantization: awq
tensor-parallel-size: 2
```

Mount the file anywhere into the container and point the `VLLM_CONFIG_FILE` env var at it. CLI flags built from environment variables take precedence over config file values (standard `vllm serve` behavior).

For the complete list of all available environment variables, examples, and detailed descriptions: **[Configuration](docs/configuration.md)**

## Option 2: Build Docker Image with Model Inside

To build an image with the model baked in, you must specify the following docker arguments when building the image.

### Prerequisites

- Docker

### Arguments

- **Required**
  - `MODEL_NAME`
- **Optional**
  - `MODEL_REVISION`: Model revision to load (default: `main`).
  - `VLLM_VERSION`: Tag of the official [`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai) base image to use (default: `v0.23.0`).
  - `BASE_PATH`: Storage directory where huggingface cache and model will be located. (default: `/runpod-volume`, which will utilize network storage if you attach it or create a local directory within the image if you don't. If your intention is to bake the model into the image, you should set this to something like `/models` to make sure there are no issues if you were to accidentally attach network storage.)
  - `QUANTIZATION`
  - `TOKENIZER_NAME`: Tokenizer repository if you would like to use a different tokenizer than the one that comes with the model. (default: `None`, which uses the model's tokenizer)
  - `TOKENIZER_REVISION`: Tokenizer revision to load (default: `main`).

For the remaining settings, you may apply them as environment variables when running the container. Supported environment variables are listed in the [Environment Variables](#environment-variables) section.

### Example: Building an image with OpenChat-3.5

```bash
docker build -t username/image:tag --build-arg MODEL_NAME="openchat/openchat_3.5" --build-arg BASE_PATH="/models" .
```

### Example: Choosing the vLLM Version

The image is based on the official `vllm/vllm-openai` image; switching vLLM versions is a single build arg:

```bash
docker build -t username/image:tag --build-arg VLLM_VERSION=v0.10.0 --build-arg MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" .
```

To run a nightly vLLM build, point `VLLM_VERSION` at a nightly tag or use `latest` at your own risk.

### (Optional) Including Huggingface Token

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

## Option 3: Load Balancing with the vLLM Image

If you want a [load balancing endpoint](https://docs.runpod.io/serverless/load-balancing/overview) — requests routed directly to workers over HTTP, with no job queue — you don't need this worker. The official [`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai) image already serves every vLLM route (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, embeddings, etc.) plus health checks, so you can deploy it as-is:

1. Create a new Serverless endpoint and set the **Endpoint Type** to **Load Balancer**.
2. Use `vllm/vllm-openai:<version>` as the container image (pin a version rather than `latest`).
3. Set the container start args to `vllm serve` arguments — at minimum `--model`, plus any other vLLM flags:

   ```bash
   --model meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192
   ```

4. Configure the endpoint:
   - Set `PORT=8000` as an environment variable and expose `8000/http` under **Expose HTTP Ports** (vLLM serves on 8000 by default).
   - Set `HF_TOKEN` if your model is gated/private.
   - vLLM already serves the load balancer's default `/ping` health check (and `/health`); no extra configuration is needed. To use a different path, set `HEALTH_CHECK_PATH`.
5. Once a worker is healthy, call any vLLM route directly at `https://<ENDPOINT_ID>.api.runpod.ai/<PATH>`. The `model` field is the repo passed to `--model` (unless you also set `--served-model-name`):

   ```bash
   curl https://<ENDPOINT_ID>.api.runpod.ai/v1/chat/completions \
     -H "Authorization: Bearer <RUNPOD_API_KEY>" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "meta-llama/Llama-3.1-8B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
   ```

Note that load balancing endpoints have no queue: a request that arrives before any worker is ready returns an error instead of waiting, so clients should retry through cold starts. See the [load balancing docs](https://docs.runpod.io/serverless/load-balancing/overview) for health check semantics, cold-start handling, and scaling behavior.

# Compatible Model Architectures

You can deploy **any model on Hugging Face** that is supported by vLLM. For the complete and up-to-date list of supported model architectures, see the [vLLM Supported Models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models).

# Usage: OpenAI Compatibility

The vLLM Worker is fully compatible with OpenAI's API, and you can use it with any OpenAI Codebase by changing only 3 lines in total. The supported routes are <ins>Chat Completions</ins>, <ins>Models</ins>, <ins>Responses</ins>, and <ins>Messages</ins> - with both streaming and non-streaming.

## Modifying your OpenAI Codebase to use your deployed vLLM Worker

**Python** (similar to Node.js, etc.):

1. When initializing the OpenAI Client in your code, change the `api_key` to your Runpod API Key and the `base_url` to your Runpod Serverless Endpoint URL in the following format: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1`, filling in your deployed endpoint ID. For example, if your Endpoint ID is `abc1234`, the URL would be `https://api.runpod.ai/v2/abc1234/openai/v1`.

   - Before:

   ```python
   from openai import OpenAI

   client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
   ```

   - After:

   ```python
   from openai import OpenAI

   client = OpenAI(
       api_key=os.environ.get("RUNPOD_API_KEY"),
       base_url="https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1",
   )
   ```

2. Change the `model` parameter to your deployed model's name whenever using Completions or Chat Completions.
   - Before:
   ```python
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Why is Runpod the best platform?"}],
       temperature=0,
       max_tokens=100,
   )
   ```
   - After:
   ```python
   response = client.chat.completions.create(
       model="<YOUR DEPLOYED MODEL REPO/NAME>",
       messages=[{"role": "user", "content": "Why is Runpod the best platform?"}],
       temperature=0,
       max_tokens=100,
   )
   ```

**Using http requests**:

1. Change the `Authorization` header to your Runpod API Key and the `url` to your Runpod Serverless Endpoint URL in the following format: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1`
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
       "content": "Why is Runpod the best platform?"
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
       "content": "Why is Runpod the best platform?"
     }
   ],
   "temperature": 0,
   "max_tokens": 100
   }'
   ```

## OpenAI Request Input Parameters:

When using the chat completion feature of the vLLM Serverless Endpoint Worker, you can customize your requests with the following parameters:

### Chat Completions [RECOMMENDED]

<details>
  <summary>Supported Chat Completions Inputs and Descriptions</summary>

| Parameter           | Type                             | Default Value | Description                                                                                                                                                                                                                                                  |
| ------------------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `messages`          | Union[str, List[Dict[str, str]]] |               | List of messages, where each message is a dictionary with a `role` and `content`. The model's chat template will be applied to the messages automatically, so the model must have one or it should be specified as `CUSTOM_CHAT_TEMPLATE` env var.           |
| `model`             | str                              |               | The model repo that you've deployed on your Runpod Serverless Endpoint. If you are unsure what the name is or are baking the model in, use the guide to get the list of available models in the **Examples: Using your Runpod endpoint with OpenAI** section |
| `temperature`       | Optional[float]                  | 0.7           | Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.                                                                              |
| `top_p`             | Optional[float]                  | 1.0           | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.                                                                                                                            |
| `n`                 | Optional[int]                    | 1             | Number of output sequences to return for the given prompt.                                                                                                                                                                                                   |
| `max_tokens`        | Optional[int]                    | None          | Maximum number of tokens to generate per output sequence.                                                                                                                                                                                                    |
| `seed`              | Optional[int]                    | None          | Random seed to use for the generation.                                                                                                                                                                                                                       |
| `stop`              | Optional[Union[str, List[str]]]  | list          | List of strings that stop the generation when they are generated. The returned output will not contain the stop strings.                                                                                                                                     |
| `stream`            | Optional[bool]                   | False         | Whether to stream or not                                                                                                                                                                                                                                     |
| `presence_penalty`  | Optional[float]                  | 0.0           | Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.                                                          |
| `frequency_penalty` | Optional[float]                  | 0.0           | Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.                                                              |
| `logit_bias`        | Optional[Dict[str, float]]       | None          | Unsupported by vLLM                                                                                                                                                                                                                                          |
| `user`              | Optional[str]                    | None          | Unsupported by vLLM                                                                                                                                                                                                                                          |

Additional parameters supported by vLLM:
| `best_of` | Optional[int] | None | Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`. |
| `top_k` | Optional[int] | -1 | Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens. |
| `ignore_eos` | Optional[bool] | False | Whether to ignore the EOS token and continue generating tokens after the EOS token is generated. |
| `use_beam_search` | Optional[bool] | False | Whether to use beam search instead of sampling. |
| `stop_token_ids` | Optional[List[int]] | list | List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens. |
| `skip_special_tokens` | Optional[bool] | True | Whether to skip special tokens in the output. |
| `spaces_between_special_tokens`| Optional[bool] | True | Whether to add spaces between special tokens in the output. Defaults to True. |
| `add_generation_prompt` | Optional[bool] | True | Read more [here](https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts) |
| `echo` | Optional[bool] | False | Echo back the prompt in addition to the completion |
| `repetition_penalty` | Optional[float] | 1.0 | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens. |
| `min_p` | Optional[float] | 0.0 | Float that represents the minimum probability for a token to |
| `length_penalty` | Optional[float] | 1.0 | Float that penalizes sequences based on their length. Used in beam search.. |
| `include_stop_str_in_output` | Optional[bool] | False | Whether to include the stop strings in output text. Defaults to False.|

</details>

### Examples: Using your Runpod endpoint with OpenAI

First, initialize the OpenAI Client with your Runpod API Key and Endpoint URL:

```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your Runpod API Key and Endpoint URL
client = OpenAI(
    api_key=os.environ.get("RUNPOD_API_KEY"),
    base_url="https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1",
)
```

### Chat Completions:

This is the format used for GPT-4 and focused on instruction-following and chat. Examples of Open Source chat/instruct models include `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `openchat/openchat-3.5-0106`, `NousResearch/Nous-Hermes-2-Mistral-7B-DPO` and more. However, if your model is a completion-style model with no chat/instruct fine-tune and/or does not have a chat template, you can still use this if you provide a chat template with the environment variable `CUSTOM_CHAT_TEMPLATE`.

- **Streaming**:
  ```python
  # Create a chat completion stream
  response_stream = client.chat.completions.create(
      model="<YOUR DEPLOYED MODEL REPO/NAME>",
      messages=[{"role": "user", "content": "Why is Runpod the best platform?"}],
      temperature=0,
      max_tokens=100,
      stream=True,
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
      messages=[{"role": "user", "content": "Why is Runpod the best platform?"}],
      temperature=0,
      max_tokens=100,
  )
  # Print the response
  print(response.choices[0].message.content)
  ```

### Getting a list of names for available models:

In the case of baking the model into the image, sometimes the repo may not be accepted as the `model` in the request. In this case, you can list the available models as shown below and use that name.

```python
models_response = client.models.list()
list_of_models = [model.id for model in models_response]
print(list_of_models)
```

### OpenAI Responses API

**Path:** `/openai/v1/responses` (full URL: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1/responses`)

Supports the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) request shape. Like other `/openai/` routes, this is served directly—use the `/openai/` prefix rather than the RunPod native job queue for these calls.

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "input": "Tell me a joke."
}
```

**Using HTTP requests:**

```bash
curl https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR RUNPOD API KEY>" \
  -d '{
    "model": "<YOUR DEPLOYED MODEL REPO/NAME>",
    "input": "Tell me a joke."
  }'
```

### Anthropic Messages API

**Path:** `/openai/v1/messages` (full URL: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1/messages`)

Supports the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) format. Served directly, bypassing the RunPod queue.

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "max_tokens": 256,
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

**Using HTTP requests:**

```bash
curl https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR RUNPOD API KEY>" \
  -d '{
    "model": "<YOUR DEPLOYED MODEL REPO/NAME>",
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

# Usage: Standard (Non-OpenAI)

## Request Input Parameters

<details>
  <summary>Click to expand table</summary>
    
  You may either use a `prompt` or a list of `messages` as input. Under the hood, `prompt` is proxied to vLLM's `/v1/completions` and `messages` to `/v1/chat/completions` (with the model's chat template applied, so the model must have one or you must set the `CUSTOM_CHAT_TEMPLATE` env var).
  | Argument              | Type                 | Default            | Description                                                                                            |
  |-----------------------|----------------------|--------------------|--------------------------------------------------------------------------------------------------------|
  | `prompt`              | str                  |                    | Prompt string to generate text based on. Proxied to `/v1/completions`.                                 |
  | `messages`            | list[dict[str, str]] |                    | List of messages, which will automatically have the model's chat template applied. Overrides `prompt`. |
  | `sampling_params`     | dict                 | {}                 | Sampling parameters forwarded in the request body (temperature, top_p, max_tokens, ...).               |
  | `stream`              | bool                 | False              | Whether to enable streaming of output. If True, raw SSE chunks are streamed as they are generated.     |

  You can also call **any vLLM route directly** with the generic proxy form:

  ```json
  {
    "input": {
      "route": "/v1/chat/completions",
      "method": "POST",
      "body": {
        "model": "<YOUR DEPLOYED MODEL>",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
      }
    }
  }
  ```
</details>

### Sampling Parameters

Below are all available sampling parameters that you can specify in the `sampling_params` dictionary. If you do not specify any of these parameters, the default values will be used.

<details>
  <summary>Click to expand table</summary>

| Argument                        | Type                        | Default | Description                                                                                                                                                                                   |
| ------------------------------- | --------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n`                             | int                         | 1       | Number of output sequences generated from the prompt. The top `n` sequences are returned.                                                                                                     |
| `best_of`                       | Optional[int]               | `n`     | Number of output sequences generated from the prompt. The top `n` sequences are returned from these `best_of` sequences. Must be ≥ `n`. Treated as beam width in beam search. Default is `n`. |
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
| `stop`                          | Union[None, str, List[str]] | None    | List of strings that stop generation when produced. The output will not contain these strings.                                                                                                |
| `stop_token_ids`                | Optional[List[int]]         | None    | List of token IDs that stop generation when produced. Output contains these tokens unless they are special tokens.                                                                            |
| `ignore_eos`                    | bool                        | False   | Whether to ignore the End-Of-Sequence token and continue generating tokens after its generation.                                                                                              |
| `max_tokens`                    | int                         | 16      | Maximum number of tokens to generate per output sequence.                                                                                                                                     |
| `skip_special_tokens`           | bool                        | True    | Whether to skip special tokens in the output.                                                                                                                                                 |
| `spaces_between_special_tokens` | bool                        | True    | Whether to add spaces between special tokens in the output.                                                                                                                                   |

### Text Input Formats

You may either use a `prompt` or a list of `messages` as input.

1.  `prompt`
    The prompt string can be any string, and the model's chat template will not be applied to it unless `apply_chat_template` is set to `true`, in which case it will be treated as a user message.

        Example:
        ```json
        {
          "input": {
            "prompt": "why sky is blue?",
            "sampling_params": {
              "temperature": 0.7,
              "max_tokens": 100
            }
          }
        }
        ```

2.  `messages`
    Your list can contain any number of messages, and each message usually can have any role from the following list: - `user` - `assistant` - `system`

    However, some models may have different roles, so you should check the model's chat template to see which roles are required.

    The model's chat template will be applied to the messages automatically, so the model must have one.

    Example:

    ```json
    {
      "input": {
        "messages": [
          {
            "role": "system",
            "content": "You are a helpful AI assistant that provides clear and concise responses."
          },
          {
            "role": "user",
            "content": "Can you explain the difference between supervised and unsupervised learning?"
          },
          {
            "role": "assistant",
            "content": "Sure! Supervised learning uses labeled data, meaning each input has a corresponding correct output. The model learns by mapping inputs to known outputs. In contrast, unsupervised learning works with unlabeled data, where the model identifies patterns, structures, or clusters without predefined answers."
          }
        ],
        "sampling_params": {
          "temperature": 0.7,
          "max_tokens": 100
        }
      }
    }
    ```

</details>
