<div align="center">

# OpenAI-Compatible vLLM Serverless Endpoint Worker
Deploy OpenAI-Compatible Blazing-Fast LLM Endpoints powered by the [vLLM](https://github.com/vllm-project/vllm) Inference Engine on RunPod Serverless with just a few clicks.
<!-- 
![vLLM Version](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Frunpod-workers%2Fworker-vllm%2Fmain%2Fvllm-base-image%2Fvllm-metadata.yml&query=%24.version&style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEuMSIgd2lkdGg9IjU1cHgiIGhlaWdodD0iNTZweCIgc3R5bGU9InNoYXBlLXJlbmRlcmluZzpnZW9tZXRyaWNQcmVjaXNpb247IHRleHQtcmVuZGVyaW5nOmdlb21ldHJpY1ByZWNpc2lvbjsgaW1hZ2UtcmVuZGVyaW5nOm9wdGltaXplUXVhbGl0eTsgZmlsbC1ydWxlOmV2ZW5vZGQ7IGNsaXAtcnVsZTpldmVub2RkIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI%2BCjxnPjxwYXRoIHN0eWxlPSJvcGFjaXR5OjEiIGZpbGw9IiMzN2E0ZmUiIGQ9Ik0gNTEuNSwwLjUgQyA0Ni41ODIyLDE4LjA4MzggNDEuOTE1NiwzNS43NTA1IDM3LjUsNTMuNUMgMzIuMTY2Nyw1My41IDI2LjgzMzMsNTMuNSAyMS41LDUzLjVDIDIwLjgzMzMsNTMuNSAyMC41LDUzLjE2NjcgMjAuNSw1Mi41QyAyMS4zMzgyLDUyLjE1ODMgMjEuNjcxNiw1MS40OTE2IDIxLjUsNTAuNUMgMjIuMjIyOSw0Ni44NTU1IDIzLjIyMjksNDMuMTg4OSAyNC41LDM5LjVDIDI0LjY5MTcsMzYuMzk5MiAyNS4zNTg0LDMzLjM5OTIgMjYuNSwzMC41QyAyNi4yOTA3LDI5LjkxNCAyNS45NTc0LDI5LjQxNCAyNS41LDI5QyAyNy40NDE0LDI3LjE4NDEgMjguMTA4MSwyNS4xODQxIDI3LjUsMjNDIDI5LjI0MTUsMTguNTM4NyAzMC45MDgyLDE0LjAzODcgMzIuNSw5LjVDIDM4Ljc3NTcsNi4xOTM1OCA0NS4xMDkxLDMuMTkzNTggNTEuNSwwLjUgWiIvPjwvZz4KPGc%2BPHBhdGggc3R5bGU9Im9wYWNpdHk6MC45ODQiIGZpbGw9IiNmY2I3MWQiIGQ9Ik0gMjIuNSwxMi41IEMgMjEuNTA0NiwyNC45ODkgMjEuMTcxMywzNy42NTU3IDIxLjUsNTAuNUMgMjEuNjcxNiw1MS40OTE2IDIxLjMzODIsNTIuMTU4MyAyMC41LDUyLjVDIDEzLjAzMTEsMzkuMjI4NyA2LjM2NDQxLDI1LjU2MjEgMC41LDExLjVDIDguMDE5MDUsMTEuMTc1IDE1LjM1MjQsMTEuNTA4NCAyMi41LDEyLjUgWiIvPjwvZz4KPGc%2BPHBhdGggc3R5bGU9Im9wYWNpdHk6MC4wMiIgZmlsbD0iI2Q3ZGZlOCIgZD0iTSAyMi41LDEyLjUgQyAyMy4xNjY3LDIxLjUgMjMuODMzMywzMC41IDI0LjUsMzkuNUMgMjMuMjIyOSw0My4xODg5IDIyLjIyMjksNDYuODU1NSAyMS41LDUwLjVDIDIxLjE3MTMsMzcuNjU1NyAyMS41MDQ2LDI0Ljk4OSAyMi41LDEyLjUgWiIvPjwvZz4KPGc%2BPHBhdGggc3R5bGU9Im9wYWNpdHk6MC43NTMiIGZpbGw9IiNjZmQ2ZGQiIGQ9Ik0gNTEuNSwwLjUgQyA1Mi42MTI5LDEuOTQ2MzkgNTIuNzc5NiwzLjYxMzA1IDUyLDUuNUMgNDcuODAzNiwyMi4yODg3IDQzLjMwMzYsMzguOTU1MyAzOC41LDU1LjVDIDMyLjUsNTUuNSAyNi41LDU1LjUgMjAuNSw1NS41QyAyMC44MzMzLDU0LjgzMzMgMjEuMTY2Nyw1NC4xNjY3IDIxLjUsNTMuNUMgMjYuODMzMyw1My41IDMyLjE2NjcsNTMuNSAzNy41LDUzLjVDIDQxLjkxNTYsMzUuNzUwNSA0Ni41ODIyLDE4LjA4MzggNTEuNSwwLjUgWiIvPjwvZz4KPC9zdmc%2BCg%3D%3D&label=STABLE%20vLLM%20Version&link=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm)
![Worker Version](https://img.shields.io/github/v/tag/runpod-workers/worker-vllm?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDI2LjUuMywgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAyMDAwIDIwMDAiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDIwMDAgMjAwMDsiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8c3R5bGUgdHlwZT0idGV4dC9jc3MiPgoJLnN0MHtmaWxsOiM2NzNBQjc7fQo8L3N0eWxlPgo8Zz4KCTxnPgoJCTxwYXRoIGNsYXNzPSJzdDAiIGQ9Ik0xMDE3Ljk1LDcxMS4wNGMtNC4yMiwyLjM2LTkuMTgsMy4wMS0xMy44NiwxLjgyTDM4Ni4xNyw1NTUuM2MtNDEuNzItMTAuNzYtODYuMDItMC42My0xMTYuNiwyOS43MwoJCQlsLTEuNCwxLjM5Yy0zNS45MiwzNS42NS0yNy41NSw5NS44LDE2Ljc0LDEyMC4zbDU4NC4zMiwzMjQuMjNjMzEuMzYsMTcuNCw1MC44Miw1MC40NSw1MC44Miw4Ni4zMnY4MDYuNzYKCQkJYzAsMzUuNDktMzguNDEsNTcuNjctNjkuMTUsMzkuOTRsLTcwMy4xNS00MDUuNjRjLTIzLjYtMTMuNjEtMzguMTMtMzguNzgtMzguMTMtNjYuMDJWNjY2LjYzYzAtODcuMjQsNDYuNDUtMTY3Ljg5LDEyMS45Mi0yMTEuNjYKCQkJTDkzMy44NSw0Mi4xNWMyMy40OC0xMy44LDUxLjQ3LTE3LjcsNzcuODMtMTAuODRsNzQ1LjcxLDE5NC4xYzMxLjUzLDguMjEsMzYuOTksNTAuNjUsOC41Niw2Ni41N0wxMDE3Ljk1LDcxMS4wNHoiLz4KCQk8cGF0aCBjbGFzcz0ic3QwIiBkPSJNMTUyNy43NSw1MzYuMzhsMTI4Ljg5LTc5LjYzbDE4OS45MiwxMDkuMTdjMjcuMjQsMTUuNjYsNDMuOTcsNDQuNzMsNDMuODIsNzYuMTVsLTQsODU3LjYKCQkJYy0wLjExLDI0LjM5LTEzLjE1LDQ2Ljg5LTM0LjI1LDU5LjExbC03MDEuNzUsNDA2LjYxYy0zMi4zLDE4LjcxLTcyLjc0LTQuNTktNzIuNzQtNDEuOTJ2LTc5Ny40MwoJCQljMC0zOC45OCwyMS4wNi03NC45MSw1NS4wNy05My45Nmw1OTAuMTctMzMwLjUzYzE4LjIzLTEwLjIxLDE4LjY1LTM2LjMsMC43NS00Ny4wOUwxNTI3Ljc1LDUzNi4zOHoiLz4KCQk8cGF0aCBjbGFzcz0ic3QwIiBkPSJNMTUyNC4wMSw2NjUuOTEiLz4KCTwvZz4KPC9nPgo8L3N2Zz4K&logoColor=%23ffffff&label=STABLE%20Worker%20Version&color=%23673ab7)
![vLLM Version](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Frunpod-workers%2Fworker-vllm%2Fmain%2Fvllm-base-image%2Fvllm-metadata.yml&query=%24.dev_version&style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEuMSIgd2lkdGg9IjU1cHgiIGhlaWdodD0iNTZweCIgc3R5bGU9InNoYXBlLXJlbmRlcmluZzpnZW9tZXRyaWNQcmVjaXNpb247IHRleHQtcmVuZGVyaW5nOmdlb21ldHJpY1ByZWNpc2lvbjsgaW1hZ2UtcmVuZGVyaW5nOm9wdGltaXplUXVhbGl0eTsgZmlsbC1ydWxlOmV2ZW5vZGQ7IGNsaXAtcnVsZTpldmVub2RkIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI%2BCjxnPjxwYXRoIHN0eWxlPSJvcGFjaXR5OjEiIGZpbGw9IiMzN2E0ZmUiIGQ9Ik0gNTEuNSwwLjUgQyA0Ni41ODIyLDE4LjA4MzggNDEuOTE1NiwzNS43NTA1IDM3LjUsNTMuNUMgMzIuMTY2Nyw1My41IDI2LjgzMzMsNTMuNSAyMS41LDUzLjVDIDIwLjgzMzMsNTMuNSAyMC41LDUzLjE2NjcgMjAuNSw1Mi41QyAyMS4zMzgyLDUyLjE1ODMgMjEuNjcxNiw1MS40OTE2IDIxLjUsNTAuNUMgMjIuMjIyOSw0Ni44NTU1IDIzLjIyMjksNDMuMTg4OSAyNC41LDM5LjVDIDI0LjY5MTcsMzYuMzk5MiAyNS4zNTg0LDMzLjM5OTIgMjYuNSwzMC41QyAyNi4yOTA3LDI5LjkxNCAyNS45NTc0LDI5LjQxNCAyNS41LDI5QyAyNy40NDE0LDI3LjE4NDEgMjguMTA4MSwyNS4xODQxIDI3LjUsMjNDIDI5LjI0MTUsMTguNTM4NyAzMC45MDgyLDE0LjAzODcgMzIuNSw5LjVDIDM4Ljc3NTcsNi4xOTM1OCA0NS4xMDkxLDMuMTkzNTggNTEuNSwwLjUgWiIvPjwvZz4KPGc%2BPHBhdGggc3R5bGU9Im9wYWNpdHk6MC45ODQiIGZpbGw9IiNmY2I3MWQiIGQ9Ik0gMjIuNSwxMi41IEMgMjEuNTA0NiwyNC45ODkgMjEuMTcxMywzNy42NTU3IDIxLjUsNTAuNUMgMjEuNjcxNiw1MS40OTE2IDIxLjMzODIsNTIuMTU4MyAyMC41LDUyLjVDIDEzLjAzMTEsMzkuMjI4NyA2LjM2NDQxLDI1LjU2MjEgMC41LDExLjVDIDguMDE5MDUsMTEuMTc1IDE1LjM1MjQsMTEuNTA4NCAyMi41LDEyLjUgWiIvPjwvZz4KPGc%2BPHBhdGggc3R5bGU9Im9wYWNpdHk6MC4wMiIgZmlsbD0iI2Q3ZGZlOCIgZD0iTSAyMi41LDEyLjUgQyAyMy4xNjY3LDIxLjUgMjMuODMzMywzMC41IDI0LjUsMzkuNUMgMjMuMjIyOSw0My4xODg5IDIyLjIyMjksNDYuODU1NSAyMS41LDUwLjVDIDIxLjE3MTMsMzcuNjU1NyAyMS41MDQ2LDI0Ljk4OSAyMi41LDEyLjUgWiIvPjwvZz4KPGc%2BPHBhdGggc3R5bGU9Im9wYWNpdHk6MC43NTMiIGZpbGw9IiNjZmQ2ZGQiIGQ9Ik0gNTEuNSwwLjUgQyA1Mi42MTI5LDEuOTQ2MzkgNTIuNzc5NiwzLjYxMzA1IDUyLDUuNUMgNDcuODAzNiwyMi4yODg3IDQzLjMwMzYsMzguOTU1MyAzOC41LDU1LjVDIDMyLjUsNTUuNSAyNi41LDU1LjUgMjAuNSw1NS41QyAyMC44MzMzLDU0LjgzMzMgMjEuMTY2Nyw1NC4xNjY3IDIxLjUsNTMuNUMgMjYuODMzMyw1My41IDMyLjE2NjcsNTMuNSAzNy41LDUzLjVDIDQxLjkxNTYsMzUuNzUwNSA0Ni41ODIyLDE4LjA4MzggNTEuNSwwLjUgWiIvPjwvZz4KPC9zdmc%2BCg%3D%3D&label=DEV%20vLLM%20Version%20&link=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm)\
![Docker Pulls](https://img.shields.io/docker/pulls/runpod/worker-vllm?style=for-the-badge&logo=docker&label=Docker%20Pulls&link=https%3A%2F%2Fhub.docker.com%2Frepository%2Fdocker%2Frunpod%2Fworker-vllm%2Fgeneral) -->
<!-- 
![Docker Automatic Build](https://img.shields.io/github/actions/workflow/status/runpod-workers/worker-vllm/docker-build-release.yml?style=flat&label=BUILD) -->


</div>

# News:

### 1. UI for Deploying vLLM Worker on RunPod console:
![Demo of Deploying vLLM Worker on RunPod console with new UI](media/ui_demo.gif)

### 2. Worker vLLM `v2.7.0` with vLLM `0.9.1` now available under `stable` tags 

Update v2.7.0 is now available, use the image tag `runpod/worker-v1-vllm:v2.7.0stable-cuda12.1.0`.

### 3. OpenAI-Compatible [Embedding Worker](https://github.com/runpod-workers/worker-infinity-embedding) Released
Deploy your own OpenAI-compatible Serverless Endpoint on RunPod with multiple embedding models and fast inference for RAG and more! 



### 4. Caching Accross RunPod Machines
Worker vLLM is now cached on all RunPod machines, resulting in near-instant deployment! Previously, downloading and extracting the image took 3-5 minutes on average.


## Table of Contents
- [Setting up the Serverless Worker](#setting-up-the-serverless-worker)
  - [Option 1: Deploy Any Model Using Pre-Built Docker Image **[RECOMMENDED]**](#option-1-deploy-any-model-using-pre-built-docker-image-recommended)
    - [Prerequisites](#prerequisites)
    - [Environment Variables](#environment-variables)
      - [LLM Settings](#llm-settings)
      - [Tokenizer Settings](#tokenizer-settings)
      - [System and Parallelism Settings](#system-and-parallelism-settings)
      - [Streaming Batch Size Settings](#streaming-batch-size-settings)
      - [OpenAI Settings](#openai-settings)
      - [Serverless Settings](#serverless-settings)
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
    - [Examples: Using your RunPod endpoint with OpenAI](#examples-using-your-runpod-endpoint-with-openai)
- [Usage: standard](#non-openai-usage)
  - [Input Request Parameters](#input-request-parameters)
    - [Text Input Formats](#text-input-formats)
    - [Sampling Parameters](#sampling-parameters)
- [Worker Config](#worker-config)
  - [Writing your worker-config.json](#writing-your-worker-configjson)
  - [Example of schema](#example-of-schema)
  - [Example of versions](#example-of-versions)

# Setting up the Serverless Worker

### Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]

> [!NOTE]
> You can now deploy from the dedicated UI on the RunPod console with all of the settings and choices listed. 
> Try now by accessing in Explore or Serverless pages on the RunPod console!


We now offer a pre-built Docker Image for the vLLM Worker that you can configure entirely with Environment Variables when creating the RunPod Serverless Endpoint:

---

## RunPod Worker Images

Below is a summary of the available RunPod Worker images, categorized by image stability and CUDA version compatibility.

| CUDA Version | Stable Image Tag                  | Development Image Tag             | Note                                                        |
|--------------|-----------------------------------|-----------------------------------|----------------------------------------------------------------------|
| 12.1.0       | `runpod/worker-v1-vllm:v2.7.0stable-cuda12.1.0` | `runpod/worker-v1-vllm:v2.7.0dev-cuda12.1.0` | When creating an Endpoint, select CUDA Version 12.3, 12.2 and 12.1 in the filter. |



---

#### Prerequisites
- RunPod Account

#### Environment Variables
> Note:  `0` is equivalent to `False` and `1` is equivalent to `True` for boolean as int values.

#### LLM Settings
| `Name`                                    | `Default`             | `Type/Choices`                             | `Description` |
|-------------------------------------------|-----------------------|--------------------------------------------|---------------|
| `MODEL_NAME`                                   | 'facebook/opt-125m'   | `str`                                      | Name or path of the Hugging Face model to use. |
| `TOKENIZER`                               | None                  | `str`                                      | Name or path of the Hugging Face tokenizer to use. |
| `SKIP_TOKENIZER_INIT`                     | False                 | `bool`                                     | Skip initialization of tokenizer and detokenizer. |
| `TOKENIZER_MODE`                          | 'auto'                | ['auto', 'slow']                           | The tokenizer mode. |
| `TRUST_REMOTE_CODE`                       | `False`               | `bool`                                  | Trust remote code from Hugging Face. |
| `DOWNLOAD_DIR`                            | None                  | `str`                                      | Directory to download and load the weights. |
| `LOAD_FORMAT`                             | 'auto'                | `str`                                      | The format of the model weights to load. |
| `HF_TOKEN`                                | -                     | `str`                                      | Hugging Face token for private and gated models.|
| `DTYPE`                                   | 'auto'                | ['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'] | Data type for model weights and activations. |
| `KV_CACHE_DTYPE`                          | 'auto'                | ['auto', 'fp8']                            | Data type for KV cache storage. |
| `QUANTIZATION_PARAM_PATH`                 | None                  | `str`                                      | Path to the JSON file containing the KV cache scaling factors. |
| `MAX_MODEL_LEN`                           | None                  | `int`                                      | Model context length. |
| `GUIDED_DECODING_BACKEND`                 | 'outlines'            | ['outlines', 'lm-format-enforcer']         | Which engine will be used for guided decoding by default. |
| `DISTRIBUTED_EXECUTOR_BACKEND`            | None                  | ['ray', 'mp']                              | Backend to use for distributed serving. |
| `WORKER_USE_RAY`                          | False                 | `bool`                                     | Deprecated, use --distributed-executor-backend=ray. |
| `PIPELINE_PARALLEL_SIZE`                  | 1                     | `int`                                      | Number of pipeline stages. |
| `TENSOR_PARALLEL_SIZE`                    | 1                     | `int`                                      | Number of tensor parallel replicas. |
| `MAX_PARALLEL_LOADING_WORKERS`            | None                  | `int`                                      | Load model sequentially in multiple batches. |
| `RAY_WORKERS_USE_NSIGHT`                  | False                 | `bool`                                     | If specified, use nsight to profile Ray workers. |
| `ENABLE_PREFIX_CACHING`                   | False                 | `bool`                                     | Enables automatic prefix caching. |
| `DISABLE_SLIDING_WINDOW`                  | False                 | `bool`                                     | Disables sliding window, capping to sliding window size. |
| `USE_V2_BLOCK_MANAGER`                    | False                 | `bool`                                     | Use BlockSpaceMangerV2. |
| `NUM_LOOKAHEAD_SLOTS`                     | 0                     | `int`                                      | Experimental scheduling config necessary for speculative decoding. |
| `SEED`                                    | 0                     | `int`                                      | Random seed for operations. |
| `NUM_GPU_BLOCKS_OVERRIDE`                 | None                  | `int`                                      | If specified, ignore GPU profiling result and use this number of GPU blocks. |
| `MAX_NUM_BATCHED_TOKENS`                  | None                  | `int`                                      | Maximum number of batched tokens per iteration. |
| `MAX_NUM_SEQS`                            | 256                   | `int`                                      | Maximum number of sequences per iteration. |
| `MAX_LOGPROBS`                            | 20                    | `int`                                      | Max number of log probs to return when logprobs is specified in SamplingParams. |
| `DISABLE_LOG_STATS`                       | False                 | `bool`                                     | Disable logging statistics. |
| `QUANTIZATION`                            | None                  | ['awq', 'squeezellm', 'gptq', 'bitsandbytes']              | Method used to quantize the weights. |
| `ROPE_SCALING`                            | None                  | `dict`                                     | RoPE scaling configuration in JSON format. |
| `ROPE_THETA`                              | None                  | `float`                                    | RoPE theta. Use with rope_scaling. |
| `TOKENIZER_POOL_SIZE`                     | 0                     | `int`                                      | Size of tokenizer pool to use for asynchronous tokenization. |
| `TOKENIZER_POOL_TYPE`                     | 'ray'                 | `str`                                      | Type of tokenizer pool to use for asynchronous tokenization. |
| `TOKENIZER_POOL_EXTRA_CONFIG`             | None                  | `dict`                                     | Extra config for tokenizer pool. |
| `ENABLE_LORA`                             | False                 | `bool`                                     | If True, enable handling of LoRA adapters. |
| `MAX_LORAS`                               | 1                     | `int`                                      | Max number of LoRAs in a single batch. |
| `MAX_LORA_RANK`                           | 16                    | `int`                                      | Max LoRA rank. |
| `LORA_EXTRA_VOCAB_SIZE`                   | 256                   | `int`                                      | Maximum size of extra vocabulary for LoRA adapters. |
| `LORA_DTYPE`                              | 'auto'                | ['auto', 'float16', 'bfloat16', 'float32'] | Data type for LoRA. |
| `LONG_LORA_SCALING_FACTORS`               | None                  | `tuple`                                    | Specify multiple scaling factors for LoRA adapters. |
| `MAX_CPU_LORAS`                           | None                  | `int`                                      | Maximum number of LoRAs to store in CPU memory. |
| `FULLY_SHARDED_LORAS`                     | False                 | `bool`                                     | Enable fully sharded LoRA layers. |
| `LORA_MODULES`| `[]`| `list[dict]`| Add lora adapters from Hugging Face `[{"name": "xx", "path": "xxx/xxxx", "base_model_name": "xxx/xxxx"}`|
| `SCHEDULER_DELAY_FACTOR`                  | 0.0                   | `float`                                    | Apply a delay before scheduling next prompt. |
| `ENABLE_CHUNKED_PREFILL`                  | False                 | `bool`                                     | Enable chunked prefill requests. |
| `SPECULATIVE_MODEL`                       | None                  | `str`                                      | The name of the draft model to be used in speculative decoding. |
| `NUM_SPECULATIVE_TOKENS`                  | None                  | `int`                                      | The number of speculative tokens to sample from the draft model. |
| `SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE`  | None                  | `int`                                      | Number of tensor parallel replicas for the draft model. |
| `SPECULATIVE_MAX_MODEL_LEN`               | None                  | `int`                                      | The maximum sequence length supported by the draft model. |
| `SPECULATIVE_DISABLE_BY_BATCH_SIZE`       | None                  | `int`                                      | Disable speculative decoding if the number of enqueue requests is larger than this value. |
| `NGRAM_PROMPT_LOOKUP_MAX`                 | None                  | `int`                                      | Max size of window for ngram prompt lookup in speculative decoding. |
| `NGRAM_PROMPT_LOOKUP_MIN`                 | None                  | `int`                                      | Min size of window for ngram prompt lookup in speculative decoding. |
| `SPEC_DECODING_ACCEPTANCE_METHOD`         | 'rejection_sampler'   | ['rejection_sampler', 'typical_acceptance_sampler'] | Specify the acceptance method for draft token verification in speculative decoding. |
| `TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD` | None              | `float`                                    | Set the lower bound threshold for the posterior probability of a token to be accepted. |
| `TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA`     | None              | `float`                                    | A scaling factor for the entropy-based threshold for token acceptance. |
| `MODEL_LOADER_EXTRA_CONFIG`               | None                  | `dict`                                     | Extra config for model loader. |
| `PREEMPTION_MODE`                         | None                  | `str`                                      | If 'recompute', the engine performs preemption-aware recomputation. If 'save', the engine saves activations into the CPU memory as preemption happens. |
| `PREEMPTION_CHECK_PERIOD`                 | 1.0                   | `float`                                    | How frequently the engine checks if a preemption happens. |
| `PREEMPTION_CPU_CAPACITY`                 | 2                     | `float`                                    | The percentage of CPU memory used for the saved activations. |
| `DISABLE_LOGGING_REQUEST`                 | False                 | `bool`                                     | Disable logging requests. |
| `MAX_LOG_LEN`                             | None                  | `int`                                      | Max number of prompt characters or prompt ID numbers being printed in log. |


#### Tokenizer Settings

| `Name`                                    | `Default`             | `Type/Choices`                             | `Description` |
|-------------------------------------------|-----------------------|--------------------------------------------|---------------|
| `TOKENIZER_NAME`                    | `None`               | `str`                                         |Tokenizer repository to use a different tokenizer than the model's default. |
| `TOKENIZER_REVISION`                | `None`               | `str`                                         |Tokenizer revision to load. |
| `CUSTOM_CHAT_TEMPLATE`              | `None`               | `str` of single-line jinja template                                         |Custom chat jinja template. [More Info](https://huggingface.co/docs/transformers/chat_templating) |

#### System and Parallelism Settings

| `Name`                                    | `Default`             | `Type/Choices`                             | `Description` |
|-------------------------------------------|-----------------------|--------------------------------------------|---------------|
| `GPU_MEMORY_UTILIZATION`            | `0.95`               | `float`                                         |Sets GPU VRAM utilization. |
| `MAX_PARALLEL_LOADING_WORKERS`      | `None`               | `int`                                         |Load model sequentially in multiple batches, to avoid RAM OOM when using tensor parallel and large models. |
| `BLOCK_SIZE`                        | `16`                 | `8`, `16`, `32`                           |Token block size for contiguous chunks of tokens. |
| `SWAP_SPACE`                        | `4`                  | `int`                                         |CPU swap space size (GiB) per GPU. |
| `ENFORCE_EAGER`                     | False                  | `bool`                                         |Always use eager-mode PyTorch. If False(`0`), will use eager mode and CUDA graph in hybrid for maximal performance and flexibility. |
| `MAX_SEQ_LEN_TO_CAPTURE`        | `8192`               | `int`                                     |Maximum context length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode.|
| `DISABLE_CUSTOM_ALL_REDUCE`         | `0`                  | `int`                                         |Enables or disables custom all reduce. |


#### Streaming Batch Size Settings

The way this works is that the first request will have a batch size of `DEFAULT_MIN_BATCH_SIZE`, and each subsequent request will have a batch size of `previous_batch_size * DEFAULT_BATCH_SIZE_GROWTH_FACTOR`. This will continue until the batch size reaches `DEFAULT_BATCH_SIZE`. E.g. for the default values, the batch sizes will be `1, 3, 9, 27, 50, 50, 50, ...`. You can also specify this per request, with inputs `max_batch_size`, `min_batch_size`, and `batch_size_growth_factor`. This has nothing to do with vLLM's internal batching, but rather the number of tokens sent in each HTTP request from the worker


| `Name`                                    | `Default`             | `Type/Choices`                             | `Description` |
|-------------------------------------------|-----------------------|--------------------------------------------|---------------|
| `DEFAULT_BATCH_SIZE`                | `50`                 | `int`                                         |Default and Maximum batch size for token streaming to reduce HTTP calls. |
| `DEFAULT_MIN_BATCH_SIZE`            | `1`                  | `int`                                         |Batch size for the first request, which will be multiplied by the growth factor every subsequent request. |
| `DEFAULT_BATCH_SIZE_GROWTH_FACTOR`  | `3`                  | `float`                                         |Growth factor for dynamic batch size. |

#### OpenAI Settings

| `Name`                                    | `Default`             | `Type/Choices`                             | `Description` |
|-------------------------------------------|-----------------------|--------------------------------------------|---------------|
| `RAW_OPENAI_OUTPUT`                 | `1`                  | boolean as `int`                                         |Enables raw OpenAI SSE format string output when streaming.  **Required** to be enabled (which it is by default) for OpenAI compatibility. |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `None`               | `str`                                         |Overrides the name of the served model from model repo/path to specified name, which you will then be able to use the value for the `model` parameter when making OpenAI requests |
| `OPENAI_RESPONSE_ROLE`              | `assistant`          | `str`                       |Role of the LLM's Response in OpenAI Chat Completions. |

#### Serverless Settings

| `Name`                                    | `Default`             | `Type/Choices`                             | `Description` |
|-------------------------------------------|-----------------------|--------------------------------------------|---------------|
| `MAX_CONCURRENCY`                   | `300`                | `int`                                         |Max concurrent requests per worker. vLLM has an internal queue, so you don't have to worry about limiting by VRAM, this is for improving scaling/load balancing efficiency |
| `DISABLE_LOG_STATS`                 | False                  | `bool`                                         |Enables or disables vLLM stats logging. |
| `DISABLE_LOG_REQUESTS`              | False                  | `bool`                                         |Enables or disables vLLM request logging. |

> [!TIP]
> If you are facing issues when using Mixtral 8x7B, Quantized models, or handling unusual models/architectures, try setting `TRUST_REMOTE_CODE` to `1`.


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
  - `WORKER_CUDA_VERSION`: `12.1.0` (`12.1.0` is recommended for optimal performance).
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

## Compatible Model Architectures
Below are all supported model architectures (and examples of each) that you can deploy using the vLLM Worker. You can deploy **any model on HuggingFace**, as long as its base architecture is one of the following:

- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- Command-R (`CohereForAI/c4ai-command-r-v01`, etc.)
- DBRX (`databricks/dbrx-base`, `databricks/dbrx-instruct` etc.)
- DeciLM (`Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.)
- Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
- Gemma (`google/gemma-2b`, `google/gemma-7b`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
- InternLM2 (`internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc.)
- Jais (`core42/jais-13b`, `core42/jais-13b-chat`, `core42/jais-30b-v3`, `core42/jais-30b-chat-v3`, etc.)
- LLaMA, Llama 2, and Meta Llama 3 (`meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- MiniCPM (`openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, etc.)
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- Mixtral (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OLMo (`allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)
- Orion (`OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc.)
- Phi (`microsoft/phi-1_5`, `microsoft/phi-2`, etc.)
- Phi-3 (`microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, etc.)
- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)
- Qwen2 (`Qwen/Qwen1.5-7B`, `Qwen/Qwen1.5-7B-Chat`, etc.)
- Qwen2MoE (`Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.)
- StableLM(`stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc.)
- Starcoder2(`bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc.)
- Xverse (`xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat`, etc.)
- Yi (`01-ai/Yi-6B`, `01-ai/Yi-34B`, etc.)

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


## Examples: Using your RunPod endpoint with OpenAI 

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


### Text Input Formats 
You may either use a `prompt` or a list of `messages` as input.
 1. `prompt` 
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
2. `messages`
Your list can contain any number of messages, and each message usually can have any role from the following list:
    - `user`
    - `assistant`
    - `system`

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

# Worker Config
The worker config is a JSON file that is used to build the form that helps users configure their serverless endpoint on the RunPod Web Interface.

Note: This is a new feature and only works for workers that use one model 

## Writing your worker-config.json
The JSON consists of two main parts, schema and versions.
- `schema`: Here you specify the form fields that will be displayed to the user.
  - `env_var_name`: The name of the environment variable that is being set using the form field.
  - `value`: This is the default value of the form field. It will be shown in the UI as such unless the user changes it.
  - `title`: This is the title of the form field in the UI.
  - `description`: This is the description of the form field in the UI.
  - `required`: This is a boolean that specifies if the form field is required.
  - `type`: This is the type of the form field. Options are:
    - `text`: Environment variable is a string so user inputs text in form field.
    - `select`: User selects one option from the dropdown. You must provide the `options` key value pair after type if using this.
    - `toggle`: User toggles between true and false.
    - `number`: User inputs a number in the form field.
  - `options`: Specify the options the user can select from if the type is `select`. DO NOT include this unless the `type` is `select`.
- `versions`: This is where you call the form fields specified in `schema` and organize them into categories.
  - `imageName`: This is the name of the Docker image that will be used to run the serverless endpoint.
  - `minimumCudaVersion`: This is the minimum CUDA version that is required to run the serverless endpoint.
  - `categories`: This is where you call the keys of the form fields specified in `schema` and organize them into categories. Each category is a toggle list of forms on the Web UI.
    - `title`: This is the title of the category in the UI.
    - `settings`: This is the array of settings schemas specified in `schema` associated with the category.

## Example of schema
```json
{
  "schema": {
    "TOKENIZER": {
      "env_var_name": "TOKENIZER",
      "value": "",
      "title": "Tokenizer",
      "description": "Name or path of the Hugging Face tokenizer to use.",
      "required": false,
      "type": "text"
    }, 
    "TOKENIZER_MODE": {
      "env_var_name": "TOKENIZER_MODE",
      "value": "auto",
      "title": "Tokenizer Mode",
      "description": "The tokenizer mode.",
      "required": false,
      "type": "select",
      "options": [
        { "value": "auto", "label": "auto" },
        { "value": "slow", "label": "slow" }
      ]
    },
    ...
  }
}
```

## Example of versions
```json
{
  "versions": {
    "0.5.4": {
      "imageName": "runpod/worker-v1-vllm:v1.2.0stable-cuda12.1.0",
      "minimumCudaVersion": "12.1",
      "categories": [
        {
          "title": "LLM Settings",
          "settings": [
            "TOKENIZER", "TOKENIZER_MODE", "OTHER_SETTINGS_SCHEMA_KEYS_YOU_HAVE_SPECIFIED_0", ...
          ]
        },
        {
          "title": "Tokenizer Settings",
          "settings": [
            "OTHER_SETTINGS_SCHEMA_KEYS_0", "OTHER_SETTINGS_SCHEMA_KEYS_1", ...
          ]
        },
        ...
      ]
    }
  }
}
```
