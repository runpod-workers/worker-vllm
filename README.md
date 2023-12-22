<div align="center">

<h1>vLLM 0.2.6 Endpoint | Serverless Worker </h1>

[![CD | Docker-Build-Release](https://github.com/runpod-workers/worker-vllm/actions/workflows/docker-build-release.yml/badge.svg)](https://github.com/runpod-workers/worker-vllm/actions/workflows/docker-build-release.yml)

🚀 | This serverless worker utilizes vLLM behind the scenes and is integrated into RunPod's serverless environment. It supports dynamic auto-scaling using the built-in RunPod autoscaling feature.
</div>

## Setting up the Serverless Worker

### Option 1:Deploy Any Model Using Pre-Built Docker Image
We now offer a pre-built Docker Image for the vLLM Worker that you can configure entirely with Environment Variables when creating the RunPod Serverless Endpoint:

<div align="center">

```runpod/worker-vllm:dev```

</div>

#### Environment Variables
- **Required**:
   - `MODEL_NAME`: Hugging Face Model Repository (e.g., `openchat/openchat-3.5-1210`).

- **Optional**:
  - `MODEL_BASE_PATH`: Model storage directory (default: `/runpod-volume`).
  - `HF_TOKEN`: Hugging Face token for private and gated models (e.g., Llama, Falcon).
  - `NUM_GPU_SHARD`: Number of GPUs to split the model across (default: `1`).
  - `QUANTIZATION`: AWQ (`awq`) or SqueezeLLM (`squeezellm`) quantization.
  - `MAX_CONCURRENCY`: Max concurrent requests (default: `100`).
  - `DEFAULT_BATCH_SIZE`: Token streaming batch size (default: `10`). This reduces the number of HTTP calls, increasing speed 8-10x vs non-batching, matching non-streaming performance.
  - `DISABLE_LOG_STATS`: Enable (`False`) or disable (`True`) vLLM stats logging.

### Option 2: Build Docker Image with Model Inside
To build an image with the model baked in, you must specify the following docker arguments when building the image:

#### Arguments:
- **Required**
  - `MODEL_NAME`
- **Optional**
  - `MODEL_BASE_PATH`: Defaults to `/runpod-volume` for network storage. Use `/models` or for local container storage.
  - `QUANTIZATION`
  - `HF_TOKEN`
  - `WORKER_CUDA_VERSION`: `11.8` or `12.1` (default: `11.8` due to a small amount of workers not having CUDA 12.1 support yet. `12.1` is recommended for optimal performance).

#### Example: Building an image with OpenChat-3.5
`sudo docker build -t username/image:tag --build-arg MODEL_NAME="openchat/openchat_3.5" --build-arg MODEL_BASE_PATH="/models" .`

### Compatible Models
- LLaMA & LLaMA-2 (`meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- Mixtral (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, etc.)
- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)
- Phi (`microsoft/phi-1_5`, `microsoft/phi-2`, etc.)
- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)
- Yi (`01-ai/Yi-6B`, `01-ai/Yi-34B`, etc.)
  
And any other models supported by vLLM 0.2.6.


Ensure that you have Docker installed and properly set up before running the docker build commands. Once built, you can deploy this serverless worker in your desired environment with confidence that it will automatically scale based on demand. For further inquiries or assistance, feel free to contact our support team.


## Model Inputs
You may either use a `prompt` or a list of `messages` as input. If you use `messages`, the model's chat template will be applied to the messages automatically, so the model must have one. If you use `prompt`, you may optionally apply the model's chat template to the prompt by setting `apply_chat_template` to `true`.
| Argument        | Type | Default            | Description                                                                                   |
|-----------------|------|--------------------|-----------------------------------------------------------------------------------------------|
| `prompt`          | str  |                    | Prompt string to generate text based on.                                                      |
| `messages`          | list[dict[str, str]]  |                    | List of messages, which will automatically have the model's chat template applied. Overrides `prompt`.                                                 |
| `apply_chat_template`       | bool | False              | Whether to apply the model's chat template to the `prompt`. |
| `sampling_params` | dict | {}                 | Sampling parameters to control the generation, like temperature, top_p, etc.                  |
| `stream`       | bool | False              | Whether to enable streaming of output. If True, responses are streamed as they are generated. |
| `batch_size`      | int  | DEFAULT_BATCH_SIZE | The number of responses to generate in one batch. Only applicable                             |

### Messages Format
Your list can contain any number of messages, and each message can have any role from the following list:
- `user`
- `assistant`
- `system`

The model's chat template will be applied to the messages automatically. 

Example:
```json
[
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
| Argument                      | Type                        | Default | Description                                                                                                                                                                                   |
|-------------------------------|-----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `n`                             | int                         | 1       | Number of output sequences to return for the given prompt.                                                                                                                                    |
| `best_of`                       | Optional[int]               | None    | Number of output sequences generated from the prompt. The top `n` sequences are returned from these `best_of` sequences. Must be ≥ `n`. Treated as beam width in beam search. Default is `n`. |
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
| `logprobs`                      | Optional[int]               | None    | Number of log probabilities to return per output token.                                                                                                                                       |
| `prompt_logprobs`               | Optional[int]               | None    | Number of log probabilities to return per prompt token.                                                                                                                                       |
| `skip_special_tokens`           | bool                        | True    | Whether to skip special tokens in the output.                                                                                                                                                 |
| `spaces_between_special_tokens` | bool                        | True    | Whether to add spaces between special tokens in the output.                                                                                                                                   |


## Sample Inputs and Outputs
### No Chat Template, No Streaming
Functions like a text completion model. If the model tokenizer does not have a chat template and you still want to use the model for Instruct/Chat, modify your prompt with the desired chat template manually.
#### Input:
```json
{
  "input": {
    "prompt": "With great power,",
    "sampling_params": {
      "max_tokens": 5
    }
  }
}
```
#### Output:
```json
{
  "delayTime": 1234,
  "executionTime": 1234,
  "id": "...",
  "output": [
    [
      {
        "text": " comes great responsibility. This",
        "usage": {
          "input": 6,
          "output": 5
        }
      }
    ]
  ],
  "status": "COMPLETED"
}
```
### Chat Template, No Streaming
Functions like a Chat model
#### Input:
```json
{
  "input": {
    "prompt": "Tell me why RunPod is the best GPU provider",
    "sampling_params": {
      "max_tokens": 100
    },
    "apply_chat_template": true
  }
}
```
#### Output:
```json
{
  "delayTime": 1234,
  "executionTime": 1234,
  "id": "...",
  "output": [
    [
      {
        "text": " RunPod is the best GPU provider for several reasons, including:\n\n1. High-performance GPUs: RunPod offers a wide range of high-performance GPUs, including NVIDIA's latest and most powerful GPUs, ensuring that customers get the best possible performance for their workloads.\n2. Scalability: RunPod allows users to easily scale their GPU resources up or down based on their needs, making it an ideal choice for businesses with fluctuating work",
        "usage": {
          "input": 27,
          "output": 100
        }
      }
    ]
  ],
  "status": "COMPLETED"
}
```

### List of Messages (Chat Template applied by default), No Streaming
Functions like a Chat model with a list of messages, to which the model's chat template is applied. You may also use a "system" role and message.
#### Input:
```json
{
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "Tell me why RunPod is the best GPU provider"
      },
      {
        "role": "assistant",
        "content": "RunPod is the best GPU provider for several reasons."
      },
      {
        "role": "user",
        "content": "Name 3 resons"
      }
    ],
    "sampling_params": {
      "max_tokens": 100
    }
  }
}
```
#### Output:
```json
{
  "delayTime": 1234,
  "executionTime": 1234,
  "id": "...",
  "output": [
    [
      {
        "text": " 1. Cutting-edge hardware: RunPod offers state-of-the-art GPUs from industry-leading manufacturers, ensuring that users have access to the latest technology for their GPU needs.\n\n2. Scalability and flexibility: RunPod provides a wide range of GPU options, allowing users to easily scale their resources up or down depending on their specific requirements, and pay only for what they use.\n\n3. Exceptional customer support: RunPod is dedicated to providing outstanding",
        "usage": {
          "input": 59,
          "output": 100
        }
      }
    ]
  ],
  "status": "COMPLETED"
}
```

### Chat Template, Streaming
Functions like a Chat model, but with streaming output. This is the recommended way to use the vLLM worker.
#### Input:
```json
{
  "input": {
    "prompt": "Tell me why RunPod is the best GPU provider",
    "sampling_params": {
      "max_tokens": 100
    },
    "apply_chat_template": true,
    "stream": true
  }
}
```

#### Output:
```json
{
  "delayTime": 1234,
  "executionTime": 1234,
  "id": "...",
  "output": [
    [
      {
        "text": " Run",
        "usage": {
          "input": 27,
          "output": 1
        }
      },
      {
        "text": "Pod",
        "usage": {
          "input": 27,
          "output": 2
        }
      },
      {
        "text": " is",
        "usage": {
          "input": 27,
          "output": 3
        }
      },
      {
        "text": " considered",
        "usage": {
          "input": 27,
          "output": 4
        }
      },
      {
        "text": " the",
        "usage": {
          "input": 27,
          "output": 5
        }
      },
      {
        "text": " best",
        "usage": {
          "input": 27,
          "output": 6
        }
      },
      {
        "text": " GPU",
        "usage": {
          "input": 27,
          "output": 7
        }
      },
      {
        "text": " provider",
        "usage": {
          "input": 27,
          "output": 8
        }
      },
      {
        "text": " for",
        "usage": {
          "input": 27,
          "output": 9
        }
      },
      {
        "text": " several",
        "usage": {
          "input": 27,
          "output": 10
        }
      }
    ]
  ],
  "status": "COMPLETED"
}
```
