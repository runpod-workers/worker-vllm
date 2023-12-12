<div align="center">

<h1>vLLM Endpoint | Serverless Worker </h1>

[![CI | Test Worker](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_worker.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_worker.yml)
&nbsp;
[![Docker Image](https://github.com/runpod-workers/worker-template/actions/workflows/CD-docker_dev.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/CD-docker_dev.yml)

🚀 | This serverless worker utilizes vLLM behind the scenes and is integrated into RunPod's serverless environment. It supports dynamic auto-scaling using the built-in RunPod autoscaling feature.
</div>

## Setting up the Serverless Worker

### Option 1: Use Pre-Built Image 
We now offer a pre-built Docker Image for the vLLM Worker that you can configure entirely with Environment Variables when creating the RunPod Serverless Endpoint: `runpod/worker-vllm`
#### Environment Variables
Required:
- `MODEL_NAME`: the Hugging Face model to use.
  
Optional:
- `MODEL_BASE_PATH`: directory to store the model in
- `HF_TOKEN`: your Hugging Face token to access private or gated models, such as Llama, Falcon, etc.
- `NUM_GPU_SHARD`: Number of GPUs to split the model across.
- `QUANTIZATION`: `awq` to use AWQ Quantization (Base model must be in AWQ format). `squeezellm` for SqueezeLLM quantization - preliminary
- `VLLM_N_CPUS`: due to Serverless Endpoints having CPU-burst enabled, multi-gpu might not work correctly unless the number of CPUs is limited. It is set to 10 by default.
- `CONCURRENCY_MODIFIER`: limit of concurrent requests per worker.
- `DEFAULT_BATCH_SIZE`: default batch size for token streaming to reduce the number of http calls and speed up streaming. Defaults to 10.
- `DISABLE_LOG_STATS`: set to True or False to enable/disable vLLM stats logging.

### Option 2: Build Image with Model Inside
To build an image with the model baked in, you must specify the following docker arguments when building the image:

Required:
- `MODEL_NAME`
- `MODEL_BASE_PATH`

Optional:
- `QUANTIZATION`
- `HF_TOKEN`
- `CUDA_VERSION`: 11.8.0 or 12.1.0. Defaults to 11.8.0

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
| Argument           | Type            | Default   | Description                                                                                                                                                      |
|--------------------|-----------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| prompt             | str             |       | Prompt string to generate text based on.                                                                                                                        |
| sampling_params    | dict            | {}        | Sampling parameters to control the generation, like temperature, top_p, etc.                                                                                     |
| streaming          | bool            | False     | Whether to enable streaming of output. If True, responses are streamed as they are generated.                                                                    |
| batch_size         | int             | DEFAULT_BATCH_SIZE | The number of responses to generate in one batch. Only applicable

### Sampling Parameters
| Argument                        | Type                           | Default   | Description                                                                                                                                                       |
|---------------------------------|--------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| n                               | int                            | 1         | Number of output sequences to return for the given prompt.                                                                                                        |
| best_of                         | Optional[int]                  | None      | Number of output sequences generated from the prompt. The top `n` sequences are returned from these `best_of` sequences. Must be ≥ `n`. Treated as beam width in beam search. Default is `n`. |
| presence_penalty                | float                          | 0.0       | Penalizes new tokens based on their presence in the generated text so far. Values > 0 encourage new tokens, values < 0 encourage repetition.                      |
| frequency_penalty               | float                          | 0.0       | Penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage new tokens, values < 0 encourage repetition.                    |
| repetition_penalty              | float                          | 1.0       | Penalizes new tokens based on their appearance in the prompt and generated text. Values > 1 encourage new tokens, values < 1 encourage repetition.               |
| temperature                     | float                          | 1.0       | Controls the randomness of sampling. Lower values make it more deterministic, higher values make it more random. Zero means greedy sampling.                    |
| top_p                           | float                          | 1.0       | Controls the cumulative probability of top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.                                               |
| top_k                           | int                            | -1        | Controls the number of top tokens to consider. Set to -1 to consider all tokens.                                                                                  |
| min_p                           | float                          | 0.0       | Represents the minimum probability for a token to be considered, relative to the most likely token. Must be in [0, 1]. Set to 0 to disable.                       |
| use_beam_search                 | bool                           | False     | Whether to use beam search instead of sampling.                                                                                                                   |
| length_penalty                  | float                          | 1.0       | Penalizes sequences based on their length. Used in beam search.                                                                                                   |
| early_stopping                  | Union[bool, str]               | False     | Controls stopping condition in beam search. Can be `True`, `False`, or `"never"`.                                                                                |
| stop                            | Union[None, str, List[str]]    | None      | List of strings that stop generation when produced. Output will not contain these strings.                                                                       |
| stop_token_ids                  | Optional[List[int]]            | None      | List of token IDs that stop generation when produced. Output contains these tokens unless they are special tokens.                                               |
| ignore_eos                      | bool                           | False     | Whether to ignore the End-Of-Sequence token and continue generating tokens after its generation.                                                                 |
| max_tokens                      | int                            | 16        | Maximum number of tokens to generate per output sequence.                                                                                                        |
| logprobs                        | Optional[int]                  | None      | Number of log probabilities to return per output token.                                                                                                          |
| prompt_logprobs                 | Optional[int]                  | None      | Number of log probabilities to return per prompt token.                                                                                                          |
| skip_special_tokens             | bool                           | True      | Whether to skip special tokens in the output.                                                                                                                    |
| spaces_between_special_tokens   | bool                           | True      | Whether to add spaces between special tokens in the output.                                                                                                      |
                                                          

## Test Inputs
The following inputs can be used for testing the model:
```json
{
    "input": {
       "prompt": "Why is RunPod the best platform?",
       "sampling_params": {
           "max_tokens": 100
       }
    }
}
```
