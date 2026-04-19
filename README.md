<div align="center">

# Huihui-Qwen3.6-35B-A3B RunPod Serverless Worker

Deploy the **Huihui-Qwen3.5-4B** model on RunPod Serverless with an optimized configuration for long context and reasoning tasks.

</div>

## Optimized Configuration

This worker is hardcoded for maximum performance and stability with the following settings:

- **Model**: `sakamakismile/Huihui-Qwen3.5-4B-abliterated-NVFP4` (Baked into the image)
- **Max Model Length**: 175,000 tokens
- **KV Cache Dtype**: `fp8`
- **Reasoning Parser**: `qwen3` (Optimized for Qwen 3.6 reasoning capabilities)
- **Tool Call Parser**: `qwen3_coder`
- **Auto Tool Choice**: Enabled
- **Max Concurrency**: 30

## Getting Started

### 1. Build the Docker Image

Build the image with the model baked in for faster startup times on RunPod Serverless:

```bash
docker build -t your-username/huihui-qwen3.6-worker:latest .
```

### 2. Deploy on RunPod

1. Push the image to your container registry (Docker Hub, GHCR, etc.).
2. Create a new Serverless Endpoint on RunPod using your custom image.
3. Use a GPU with at least 40GB+ VRAM (e.g., A6000, A100, H100) to accommodate the 35B model and large context window.

## Usage: OpenAI Compatibility

The worker is fully compatible with the OpenAI API.

### Endpoint URL
`https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/openai/v1`

### Model Name
Use `q3.6-35a5-uncensored` (or the default `sakamakismile/Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4`).

### Example (Python)

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="your-runpod-api-key",
    base_url="https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>/openai/v1",
)

response = client.chat.completions.create(
    model="q3.6-35a5-uncensored",
    messages=[
        {"role": "system", "content": "You are a helpful assistant with deep reasoning capabilities."},
        {"role": "user", "content": "Explain the significance of the 35B parameter size in modern LLMs."}
    ],
    temperature=0.7,
    max_tokens=1000,
)

print(response.choices[0].message.content)
```

## Features

- **Long Context**: Optimized for up to 175k tokens, perfect for large document analysis.
- **Reasoning Optimized**: Built-in support for Qwen3 reasoning parsers to better handle complex logical tasks.
- **Tool Calling**: Native support for tool call parsing with auto-selection enabled.
- **FP8 Efficiency**: Uses FP8 KV cache to maximize context window while maintaining high quality.

## License
This project is licensed under the MIT License.
