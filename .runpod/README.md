![vLLM worker banner](https://image.runpod.ai/preview/vllm/vllm-banner.png)

Run LLMs using [vLLM](https://docs.vllm.ai) with an OpenAI-compatible API

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-vllm)](https://www.runpod.io/console/hub/runpod-workers/worker-vllm)

Current vLLM version: [0.20.2](https://github.com/vllm-project/vllm/releases/tag/v0.20.2)

---

## Endpoint Configuration

All behaviour is controlled through environment variables:

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
| `REASONING_PARSER`                  | Parser for reasoning-capable models               |                     | "deepseek_r1", "qwen3", "granite", "hunyuan_a13b"                  |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | Override served model name in API                 |                     | String                                                             |
| `MAX_CONCURRENCY`                   | Maximum concurrent requests                       | 300                 | Integer                                                            |
| `ENFORCE_EAGER`                     |  If True, we will disable CUDA graph and always execute the model in eager mode. If False, we will use CUDA graph and eager execution in hybrid for maximal performance and flexibility. | true | boolean (true or false) | 

**Pass any vLLM engine arg** not listed above by setting an env var with the **UPPERCASED** field name (e.g. `MAX_MODEL_LEN=4096`, `ENABLE_CHUNKED_PREFILL=true`). The worker auto-discovers all `AsyncEngineArgs` fields from env. See the [vLLM engine args docs](https://docs.vllm.ai/en/latest/configuration/engine_args) for all available options.

**Configuration file:** You can also supply a `config.yaml` instead of (or alongside) env vars. Mount it at `/vllm_config.yaml` in the container, or set `VLLM_CONFIG_FILE` to a custom path. Use the same key names as `vllm serve` — hyphens and underscores both work:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
max-model-len: 8192
gpu-memory-utilization: 0.90
quantization: awq
```

Environment variables always override config file values.

For complete configuration options, see the [full configuration documentation](https://github.com/runpod-workers/worker-vllm/blob/main/docs/configuration.md).

### Specify Transformers Version
To change the version of the [Transformers library](https://github.com/huggingface/transformers) use the `TRANSFORMERS_VERSION` environment variable to specify the version you want to use. Note this might break the handler, so use for development purposes. 

## API Usage

This worker supports two API formats: **RunPod native** and **OpenAI-compatible**.

### RunPod Native API

For testing directly in the RunPod UI, use these examples in your endpoint's request tab.

#### Chat Completions

```json
{
  "input": {
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "sampling_params": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }
}
```

#### Chat Completions (Streaming)

```json
{
  "input": {
    "messages": [
      { "role": "user", "content": "Write a short story about a robot." }
    ],
    "sampling_params": {
      "max_tokens": 500,
      "temperature": 0.8
    },
    "stream": true
  }
}
```

#### Text Generation

For direct text generation without chat format:

```json
{
  "input": {
    "prompt": "The capital of France is",
    "sampling_params": {
      "max_tokens": 64,
      "temperature": 0.0
    }
  }
}
```

#### List Models

```json
{
  "input": {
    "openai_route": "/v1/models"
  }
}
```

---

### OpenAI-Compatible API

For external clients and SDKs, use the `/openai/v1` path prefix with your RunPod API key.

#### Chat Completions

**Path:** `/openai/v1/chat/completions`

```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" }
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

#### Chat Completions (Streaming)

```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    { "role": "user", "content": "Write a short story about a robot." }
  ],
  "max_tokens": 500,
  "temperature": 0.8,
  "stream": true
}
```

#### Text Completions

**Path:** `/openai/v1/completions`

```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "prompt": "The capital of France is",
  "max_tokens": 100,
  "temperature": 0.7
}
```

#### List Models

**Path:** `/openai/v1/models`

```json
{}
```

#### OpenAI Responses API

**Path:** `/openai/v1/responses`

Supports the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) format. Note: this route bypasses the RunPod queue and is served directly — use `/openai/` prefixed paths rather than the RunPod job queue for these endpoints.

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "input": "Tell me a joke."
}
```

#### Anthropic Messages API

**Path:** `/openai/v1/messages`

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

#### Response Format

Both APIs return the same response format:

```json
{
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Paris." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10 }
}
```

---

## Usage

Below are minimal `python` snippets so you can copy-paste to get started quickly.

> Replace `<ENDPOINT_ID>` with your endpoint ID and `<API_KEY>` with a [RunPod API key](https://docs.runpod.io/get-started/api-keys).

### OpenAI compatible API

Minimal Python example using the official `openai` SDK:

```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your Runpod API Key and Endpoint URL
client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url=f"https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)
```

`Chat Completions (Non-Streaming)`

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
    temperature=0,
    max_tokens=100,
)
print(f"Response: {response.choices[0].message.content}")
```

`Chat Completions (Streaming)`

```python
response_stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
    temperature=0,
    max_tokens=100,
    stream=True
)
for response in response_stream:
    print(response.choices[0].delta.content or "", end="", flush=True)
```

### RunPod Native API

```python
import requests

response = requests.post(
    "https://api.runpod.ai/v2/<ENDPOINT_ID>/run",
    headers={"Authorization": "Bearer <API_KEY>"},
    json={
        "input": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms"}
            ],
            "sampling_params": {
                "temperature": 0.7,
                "max_tokens": 150
            }
        }
    }
)

result = response.json()
print(result["output"])
```

## Compatibility

For supported models, see the [vLLM supported models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html).

Anything not recognized by worker-vllm is forwarded to vLLM's engine, so advanced options in the vLLM docs (guided generation, LoRA, speculative decoding, etc.) also work.

## Documentation

- **[🚀 Deployment Guide](https://docs.runpod.io/serverless/vllm/get-started)** - Step-by-step setup
- **[📖 Configuration Reference](https://github.com/runpod-workers/worker-vllm/blob/main/docs/configuration.md)** - All environment variables
- **[🏗️ Advanced Deployment](https://github.com/runpod-workers/worker-vllm/blob/main/docs/deployment.md)** - Custom builds and strategies
- **[🔧 Development Guide](https://github.com/runpod-workers/worker-vllm/blob/main/docs/conventions.md)** - Architecture and patterns
