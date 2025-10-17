<div align="center">

# OpenAI-Compatible vLLM Serverless Endpoint Worker

Deploy OpenAI-Compatible Blazing-Fast LLM Endpoints powered by the [vLLM](https://github.com/vllm-project/vllm) Inference Engine on RunPod Serverless with just a few clicks.

</div>

## Table of Contents

- [Setting up the Serverless Worker](#setting-up-the-serverless-worker)
  - [Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]](#option-1-deploy-any-model-using-pre-built-docker-image-recommended)
    - [Configuration](#configuration)
  - [Option 2: Build Docker Image with Model Inside](#option-2-build-docker-image-with-model-inside)
    - [Prerequisites](#prerequisites)
    - [Arguments](#arguments)
    - [Example: Building an image with OpenChat-3.5](#example-building-an-image-with-openchat-35)
      - [(Optional) Including Huggingface Token](#optional-including-huggingface-token)
  - [Compatible Model Architectures](#compatible-model-architectures)
- [Usage: OpenAI Compatibility](#usage-openai-compatibility)
  - [Modifying your OpenAI Codebase to use your deployed vLLM Worker](#modifying-your-openai-codebase-to-use-your-deployed-vllm-worker)
  - [OpenAI Request Input Parameters](#openai-request-input-parameters)
  - [Chat Completions [RECOMMENDED]](#chat-completions-recommended)
  - [Examples: Using your RunPod endpoint with OpenAI](#examples-using-your-runpod-endpoint-with-openai)
    - [Chat Completions](#chat-completions)
    - [Getting a list of names for available models](#getting-a-list-of-names-for-available-models)
- [Usage: Standard (Non-OpenAI)](#usage-standard-non-openai)
  - [Request Input Parameters](#request-input-parameters)
  - [Sampling Parameters](#sampling-parameters)
    - [Text Input Formats](#text-input-formats)

# Setting up the Serverless Worker

## Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]

**🚀 Deploy Guide**: Follow our [step-by-step deployment guide](https://docs.runpod.io/serverless/vllm/get-started) to deploy using the RunPod Console.

**📦 Docker Image**: `runpod/worker-v1-vllm:<version>`

- **Available Versions**: See [GitHub Releases](https://github.com/runpod-workers/worker-vllm/releases)
- **CUDA Compatibility**: Requires CUDA >= 12.1

### Configuration

Configure worker-vllm using environment variables:

| Environment Variable                | Description                                       | Default             | Options                                                            |
| ----------------------------------- | ------------------------------------------------- | ------------------- | ------------------------------------------------------------------ |
| `MODEL_NAME`                        | Path of the model weights                         | "facebook/opt-125m" | Local folder or Hugging Face repo ID                               |
| `HF_TOKEN`                          | HuggingFace access token for gated/private models |                     | Your HuggingFace access token                                      |
| `MAX_MODEL_LEN`                     | Model's maximum context length                    |                     | Integer (e.g., 4096)                                               |
| `QUANTIZATION`                      | Quantization method                               |                     | "awq", "gptq", "squeezellm", "bitsandbytes"                        |
| `TENSOR_PARALLEL_SIZE`              | Number of GPUs                                    | 1                   | Integer                                                            |
| `GPU_MEMORY_UTILIZATION`            | Fraction of GPU memory to use                     | 0.95                | Float between 0.0 and 1.0                                          |
| `MAX_NUM_SEQS`                      | Maximum number of sequences per iteration         | 256                 | Integer                                                            |
| `CUSTOM_CHAT_TEMPLATE`              | Custom chat template override                     |                     | Jinja2 template string                                             |
| `ENABLE_AUTO_TOOL_CHOICE`           | Enable automatic tool selection                   | false               | boolean (true or false)                                            |
| `TOOL_CALL_PARSER`                  | Parser for tool calls                             |                     | "mistral", "hermes", "llama3_json", "granite", "deepseek_v3", etc. |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | Override served model name in API                 |                     | String                                                             |
| `MAX_CONCURRENCY`                   | Maximum concurrent requests                       | 30                  | Integer                                                            |

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
  - `BASE_PATH`: Storage directory where huggingface cache and model will be located. (default: `/runpod-volume`, which will utilize network storage if you attach it or create a local directory within the image if you don't. If your intention is to bake the model into the image, you should set this to something like `/models` to make sure there are no issues if you were to accidentally attach network storage.)
  - `QUANTIZATION`
  - `WORKER_CUDA_VERSION`: `12.1.0` (`12.1.0` is recommended for optimal performance).
  - `TOKENIZER_NAME`: Tokenizer repository if you would like to use a different tokenizer than the one that comes with the model. (default: `None`, which uses the model's tokenizer)
  - `TOKENIZER_REVISION`: Tokenizer revision to load (default: `main`).

For the remaining settings, you may apply them as environment variables when running the container. Supported environment variables are listed in the [Environment Variables](#environment-variables) section.

### Example: Building an image with OpenChat-3.5

```bash
docker build -t username/image:tag --build-arg MODEL_NAME="openchat/openchat_3.5" --build-arg BASE_PATH="/models" .
```

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

# Compatible Model Architectures

You can deploy **any model on Hugging Face** that is supported by vLLM. For the complete and up-to-date list of supported model architectures, see the [vLLM Supported Models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models).

# Usage: OpenAI Compatibility

The vLLM Worker is fully compatible with OpenAI's API, and you can use it with any OpenAI Codebase by changing only 3 lines in total. The supported routes are <ins>Chat Completions</ins> and <ins>Models</ins> - with both streaming and non-streaming.

## Modifying your OpenAI Codebase to use your deployed vLLM Worker

**Python** (similar to Node.js, etc.):

1. When initializing the OpenAI Client in your code, change the `api_key` to your RunPod API Key and the `base_url` to your RunPod Serverless Endpoint URL in the following format: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1`, filling in your deployed endpoint ID. For example, if your Endpoint ID is `abc1234`, the URL would be `https://api.runpod.ai/v2/abc1234/openai/v1`.

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
       messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
       temperature=0,
       max_tokens=100,
   )
   ```
   - After:
   ```python
   response = client.chat.completions.create(
       model="<YOUR DEPLOYED MODEL REPO/NAME>",
       messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
       temperature=0,
       max_tokens=100,
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

### Chat Completions [RECOMMENDED]

<details>
  <summary>Supported Chat Completions Inputs and Descriptions</summary>

| Parameter           | Type                             | Default Value | Description                                                                                                                                                                                                                                                  |
| ------------------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `messages`          | Union[str, List[Dict[str, str]]] |               | List of messages, where each message is a dictionary with a `role` and `content`. The model's chat template will be applied to the messages automatically, so the model must have one or it should be specified as `CUSTOM_CHAT_TEMPLATE` env var.           |
| `model`             | str                              |               | The model repo that you've deployed on your RunPod Serverless Endpoint. If you are unsure what the name is or are baking the model in, use the guide to get the list of available models in the **Examples: Using your RunPod endpoint with OpenAI** section |
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

### Examples: Using your RunPod endpoint with OpenAI

First, initialize the OpenAI Client with your RunPod API Key and Endpoint URL:

```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your RunPod API Key and Endpoint URL
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
      messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
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
      messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
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

# Usage: Standard (Non-OpenAI)

## Request Input Parameters

<details>
  <summary>Click to expand table</summary>
    
  You may either use a `prompt` or a list of `messages` as input. If you use `messages`, the model's chat template will be applied to the messages automatically, so the model must have one. If you use `prompt`, you may optionally apply the model's chat template to the prompt by setting `apply_chat_template` to `true`.
  | Argument              | Type                 | Default            | Description                                                                                            |
  |-----------------------|----------------------|--------------------|--------------------------------------------------------------------------------------------------------|
  | `prompt`              | str                  |                    | Prompt string to generate text based on.                                                               |
  | `messages`            | list[dict[str, str]] |                    | List of messages, which will automatically have the model's chat template applied. Overrides `prompt`. |
  | `apply_chat_template` | bool                 | False              | Whether to apply the model's chat template to the `prompt`.                                            |
  | `sampling_params`     | dict                 | {}                 | Sampling parameters to control the generation, like temperature, top_p, etc. You can find all available parameters in the `Sampling Parameters` section below. |
  | `stream`              | bool                 | False              | Whether to enable streaming of output. If True, responses are streamed as they are generated.          |
  | `max_batch_size`          | int                  | env var `DEFAULT_BATCH_SIZE` | The maximum number of tokens to stream every HTTP POST call.                                                   |
  | `min_batch_size`          | int                  | env var `DEFAULT_MIN_BATCH_SIZE` | The minimum number of tokens to stream every HTTP POST call.                                           |
  | `batch_size_growth_factor` | int                  | env var `DEFAULT_BATCH_SIZE_GROWTH_FACTOR` | The growth factor by which `min_batch_size` will be multiplied for each call until `max_batch_size` is reached.           |
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
