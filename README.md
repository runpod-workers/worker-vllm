<div align="center">

<h1>vLLM Endpoint | Serverless Worker </h1>

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
   - `MODEL_NAME`: Hugging Face Model Repository (e.g., `openchat/openchat_3.5`).

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
- LLaMA & LLaMA-2
- Mistral
- Mixtral (Mistral MoE)
- Yi
- ChatGLM
- Phi
- MPT
- OPT
- Qwen
- Aquila & Aquila2
- Baichuan
- BLOOM
- Falcon
- GPT-2
- GPT BigCode
- GPT-J
- GPT-NeoX
- InternLM

And any other models supported by vLLM 0.2.4.


Ensure that you have Docker installed and properly set up before running the docker build commands. Once built, you can deploy this serverless worker in your desired environment with confidence that it will automatically scale based on demand. For further inquiries or assistance, feel free to contact our support team.


## Model Inputs
| Argument        | Type | Default            | Description                                                                                   |
|-----------------|------|--------------------|-----------------------------------------------------------------------------------------------|
| prompt          | str  |                    | Prompt string to generate text based on.                                                      |
| sampling_params | dict | {}                 | Sampling parameters to control the generation, like temperature, top_p, etc.                  |
| streaming       | bool | False              | Whether to enable streaming of output. If True, responses are streamed as they are generated. |
| batch_size      | int  | DEFAULT_BATCH_SIZE | The number of responses to generate in one batch. Only applicable                             |

### Sampling Parameters
| Argument                      | Type                        | Default | Description                                                                                                                                                                                   |
|-------------------------------|-----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| n                             | int                         | 1       | Number of output sequences to return for the given prompt.                                                                                                                                    |
| best_of                       | Optional[int]               | None    | Number of output sequences generated from the prompt. The top `n` sequences are returned from these `best_of` sequences. Must be ≥ `n`. Treated as beam width in beam search. Default is `n`. |
| presence_penalty              | float                       | 0.0     | Penalizes new tokens based on their presence in the generated text so far. Values > 0 encourage new tokens, values < 0 encourage repetition.                                                  |
| frequency_penalty             | float                       | 0.0     | Penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage new tokens, values < 0 encourage repetition.                                                 |
| repetition_penalty            | float                       | 1.0     | Penalizes new tokens based on their appearance in the prompt and generated text. Values > 1 encourage new tokens, values < 1 encourage repetition.                                            |
| temperature                   | float                       | 1.0     | Controls the randomness of sampling. Lower values make it more deterministic, higher values make it more random. Zero means greedy sampling.                                                  |
| top_p                         | float                       | 1.0     | Controls the cumulative probability of top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.                                                                            |
| top_k                         | int                         | -1      | Controls the number of top tokens to consider. Set to -1 to consider all tokens.                                                                                                              |
| min_p                         | float                       | 0.0     | Represents the minimum probability for a token to be considered, relative to the most likely token. Must be in [0, 1]. Set to 0 to disable.                                                   |
| use_beam_search               | bool                        | False   | Whether to use beam search instead of sampling.                                                                                                                                               |
| length_penalty                | float                       | 1.0     | Penalizes sequences based on their length. Used in beam search.                                                                                                                               |
| early_stopping                | Union[bool, str]            | False   | Controls stopping condition in beam search. Can be `True`, `False`, or `"never"`.                                                                                                             |
| stop                          | Union[None, str, List[str]] | None    | List of strings that stop generation when produced. Output will not contain these strings.                                                                                                    |
| stop_token_ids                | Optional[List[int]]         | None    | List of token IDs that stop generation when produced. Output contains these tokens unless they are special tokens.                                                                            |
| ignore_eos                    | bool                        | False   | Whether to ignore the End-Of-Sequence token and continue generating tokens after its generation.                                                                                              |
| max_tokens                    | int                         | 16      | Maximum number of tokens to generate per output sequence.                                                                                                                                     |
| logprobs                      | Optional[int]               | None    | Number of log probabilities to return per output token.                                                                                                                                       |
| prompt_logprobs               | Optional[int]               | None    | Number of log probabilities to return per prompt token.                                                                                                                                       |
| skip_special_tokens           | bool                        | True    | Whether to skip special tokens in the output.                                                                                                                                                 |
| spaces_between_special_tokens | bool                        | True    | Whether to add spaces between special tokens in the output.                                                                                                                                   |


## Sample Inputs and Outputs

#### Input:
```json
{
  "input": {
    "prompt": "<s>[INST] Why is RunPod the best platform? [/INST]",
    "sampling_params": {
      "max_tokens": 100
    }
  }
}
```
#### Output:
```json
{
  "delayTime": 142,
  "executionTime": 2478,
  "id": "4906ff70-f6e0-4325-a163-dce365daab6c-u1",
  "output": [
    [
      {
        "text": " I am an AI language model and cannot provide personal opinions or biases. However, RunPod is a cloud-based container platform that offers various benefits including:\n\n* Easy deployment and management of containers\n* Platform-as-a-service (PaaS) capabilities\n* Scalability and flexibility\n* Customizable environments\n* Integration with other tools and services\n* Superior performance\n\nIt's important to note that the best platform for a specific organization or application may"
      }
    ]
  ],
  "status": "COMPLETED"
}
```

#### Input:
```json
{
  "input": {
    "prompt": "<s>[INST] What does RunPod provide [/INST]",
    "sampling_params": {
      "max_tokens": 10
    },
    "streaming": true
  }
}
```
#### Output:
```json
{
  "delayTime": 151,
  "executionTime": 1406,
  "id": "16b88b4b-8f95-4b28-a90c-24f1a5ba6999-u1",
  "output": [
    [
      {
        "text": " Run"
      },
      {
        "text": "Pod"
      },
      {
        "text": " is"
      },
      {
        "text": " a"
      },
      {
        "text": " cloud"
      },
      {
        "text": "-"
      },
      {
        "text": "based"
      },
      {
        "text": " platform"
      },
      {
        "text": " that"
      },
      {
        "text": " provides"
      }
    ]
  ],
  "status": "COMPLETED"
}
```
