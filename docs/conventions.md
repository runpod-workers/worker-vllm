# Worker vLLM - Development Conventions & Architecture Guide

## Project Overview

**worker-vllm** is a RunPod serverless worker that provides OpenAI-compatible endpoints for Large Language Model (LLM) inference, powered by the vLLM engine. It enables blazing-fast LLM deployment on RunPod's serverless infrastructure with minimal configuration.

### Core Purpose

- **Primary Function**: Deploy any Hugging Face LLM as an OpenAI-compatible API endpoint
- **Platform**: RunPod Serverless infrastructure
- **Engine**: vLLM (high-performance LLM inference engine)
- **Compatibility**: Drop-in replacement for OpenAI API (Chat Completions, Models)

## High-Level Architecture

### 1. **Entry Point & Request Flow**

```
RunPod Request → handler.py → JobInput → Engine Selection → vLLM Generation → Streaming Response
```

**Key Components:**

- `src/handler.py`: Main entry point using RunPod serverless framework
- `src/utils.py`: Request parsing and utility classes (`JobInput`, `BatchSize`)
- Two engine modes: OpenAI-compatible vs. standard vLLM

### 2. **Engine Architecture**

#### Core Classes:

- **`vLLMEngine`**: Base engine handling vLLM initialization and generation
- **`OpenAIvLLMEngine`**: Wrapper providing OpenAI API compatibility
- **Engine Selection**: Automatic routing based on `job_input.openai_route`

#### Key Design Patterns:

- **Dual API Support**: Same codebase serves both OpenAI-compatible and native vLLM APIs
- **Streaming by Default**: Token-level streaming with configurable batching
- **Dynamic Batching**: Adaptive batch sizes that grow from min → max for efficiency

### 3. **Configuration System**

#### Environment-Based Configuration:

- **Single Source of Truth**: All configuration via environment variables
- **Hierarchical Loading**: `DEFAULT_ARGS` → `os.environ` → `local_model_args.json` (for baked models)
- **vLLM Argument Mapping**: Automatic translation of env vars to vLLM `AsyncEngineArgs`

#### Key Configuration Files:

- `src/engine_args.py`: Centralized configuration management
- `src/constants.py`: Default values for core settings
- `worker-config.json`: UI form generation for RunPod console

## Core Development Concepts

### 1. **Deployment Models**

#### Option 1: Pre-built Images (Recommended)

- **Image**: `runpod/worker-v1-vllm:<version>` (see [GitHub Releases](https://github.com/runpod-workers/worker-vllm/releases))
- **Configuration**: Entirely via environment variables
- **Model Loading**: Downloads model at runtime from Hugging Face
- **Use Case**: Quick deployment, model experimentation

#### Option 2: Baked Model Images

- **Build Process**: Model downloaded during Docker build
- **Storage**: Model embedded in container image
- **Configuration**: Stored in `/local_model_args.json`
- **Use Case**: Production deployments, faster cold starts

### 2. **Request Processing Patterns**

#### Input Handling:

```python
class JobInput:
    - llm_input: str | List[Dict] (prompt or messages)
    - sampling_params: SamplingParams (generation settings)
    - stream: bool (streaming vs batch response)
    - openai_route: bool (API compatibility mode)
    - batch_size configs: Dynamic batching parameters
```

#### Response Streaming:

- **Batched Streaming**: Tokens grouped into configurable batch sizes
- **Dynamic Growth**: `min_batch_size * growth_factor^n` up to `max_batch_size`
- **Usage Tracking**: Input/output token counting for billing

### 3. **Model & Tokenizer Management**

#### Tokenizer Handling:

- **Wrapper Pattern**: `TokenizerWrapper` for consistent chat template application
- **Special Cases**: Mistral models use vLLM's native tokenizer
- **Chat Templates**: Automatic application for message-based inputs

#### Model Loading:

- **Multi-GPU Support**: Automatic tensor parallelism detection
- **Quantization**: Support for AWQ, GPTQ, BitsAndBytes
- **Caching**: Hugging Face cache management

## Development Patterns & Best Practices

### 1. **Code Organization**

#### File Structure:

```
src/
├── handler.py          # RunPod entry point
├── engine.py          # Core vLLM engines
├── engine_args.py     # Configuration management
├── utils.py           # Request parsing & utilities
├── tokenizer.py       # Tokenizer wrapper
├── constants.py       # Default constants
└── download_model.py  # Model downloading logic
```

#### Separation of Concerns:

- **Engine Logic**: Isolated in `engine.py` classes
- **Configuration**: Centralized in `engine_args.py`
- **Request Handling**: Abstracted via `JobInput` class
- **Platform Integration**: Contained in `handler.py`

### 2. **Error Handling & Logging**

#### Logging Strategy:

- **Structured Logging**: Consistent format across components
- **Performance Tracking**: Timer decorators for critical operations
- **Error Context**: Detailed error messages with configuration context

#### Error Responses:

- **OpenAI Compatibility**: Standard OpenAI error format
- **Graceful Degradation**: Fallback behaviors for edge cases

### 3. **Environment Variable Conventions**

#### Naming Patterns:

- **vLLM Settings**: Match vLLM parameter names (uppercase)
- **RunPod Settings**: `MAX_CONCURRENCY`, `DEFAULT_BATCH_SIZE`
- **OpenAI Settings**: `OPENAI_` prefix for compatibility settings
- **Feature Flags**: `ENABLE_*`, `DISABLE_*` pattern

#### Type Conventions:

- **Booleans**: String 'true'/'false' or int 0/1
- **Lists**: Comma-separated strings
- **Objects**: JSON strings for complex configurations

### 4. **Docker & Deployment**

#### Multi-Stage Builds:

- **Base**: CUDA runtime environment
- **Dependencies**: Python packages and vLLM
- **Model Download**: Optional model baking stage
- **Runtime**: Final application layer

#### Build Arguments:

- **MODEL_NAME**: Primary model identifier
- **BASE_PATH**: Storage location strategy
- **QUANTIZATION**: Optimization settings
- **WORKER_CUDA_VERSION**: CUDA compatibility

#### CI/CD Strategy:

- **Development Builds**: All non-main branches → `runpod/worker-v1-vllm:dev-<branch-name>`
- **Release Builds**: Git tags (numeric) only → `runpod/worker-v1-vllm:<version>`
- **Dependency Updates**: Automated runpod package version monitoring

#### Docker Bake Configuration:

- **File**: `docker-bake.hcl` (flexible variable-based configuration)
- **Variables**: `DOCKERHUB_REPO`, `DOCKERHUB_IMG`, `RELEASE_VERSION`, `HUGGINGFACE_ACCESS_TOKEN`
- **Platform**: `linux/amd64` (GPU-optimized)

## Release & Versioning Strategy

### 1. **Version Tagging**

- **Development**: `dev-<branch-name>` (e.g., `dev-feature-new-api`)
- **Specific Versions**: `2.7.0`, `2.8.0` (semantic versioning without "v" prefix)
- **Version Discovery**: Check [GitHub Releases](https://github.com/runpod-workers/worker-vllm/releases) for available versions

### 2. **Release Workflow**

1. **Feature Development**: Work on feature branches → triggers dev builds
2. **Main Branch Staging**: Merge features to main → stable codebase (no builds)
3. **Version Release**: Create git tag from main branch (e.g., `2.8.0`) → triggers versioned release + GitHub release
4. **Docker Hub**: Versioned image pushed with tag

### 3. **Branch Strategy**

- **Feature Branches**: `feature/*`, `fix/*`, `feat/*` etc. → Dev builds
- **Main Branch**: Stable codebase ready for release (no automatic builds)
- **Git Tags**: Must be created from main branch for formal version releases

### 4. **Deployment Recommendations**

- **Production**: Use specific version tags (e.g., `2.7.0`) for stability
- **Development**: Use `dev-<branch>` for testing specific features
- **Version Selection**: Check [GitHub Releases](https://github.com/runpod-workers/worker-vllm/releases) for available versions
- **Release Process**: Always tag from main branch: `git checkout main && git tag 2.8.0 && git push origin 2.8.0`

## Performance & Scaling Considerations

### 1. **Memory Management**

- **GPU Utilization**: Default 95% GPU memory utilization
- **KV Cache**: Configurable cache types (auto, fp8)
- **Swap Space**: CPU offloading for large contexts

### 2. **Concurrency Patterns**

- **Max Concurrency**: 300 concurrent requests by default
- **vLLM Queuing**: Internal request batching and scheduling
- **RunPod Integration**: Concurrency modifier for auto-scaling

### 3. **Optimization Features**

- **Prefix Caching**: Automatic caching of common prefixes
- **Speculative Decoding**: Draft model acceleration
- **Chunked Prefill**: Memory-efficient long context handling

## Testing & Development

### 1. **Local Development**

- **Environment**: Virtual environment with GPU support
- **Configuration**: `.env` files for local testing
- **Model Testing**: Small models for development (facebook/opt-125m)

### 2. **Docker Development**

- **Build Strategy**: `docker-bake.hcl` for consistent builds
- **Testing Images**: Separate dev/stable image tags
- **Layer Caching**: Optimized for rapid iteration

### 3. **Configuration Validation**

- **Argument Matching**: Automatic validation against vLLM parameters
- **Environment Validation**: Type checking and default value handling
- **Runtime Validation**: Model compatibility checks

## API Conventions

### 1. **OpenAI Compatibility**

- **Endpoint Mapping**: `/openai/v1/chat/completions`, `/openai/v1/models`
- **Request Format**: Exact OpenAI request/response schemas
- **Authentication**: RunPod API key in Authorization header
- **Model Names**: Hugging Face repo names or custom overrides

### 2. **Native vLLM API**

- **Input Format**: `prompt` or `messages` with `sampling_params`
- **Streaming**: Token-level streaming with configurable batching
- **Extensibility**: Support for vLLM-specific features

## Common Patterns & Utilities

### 1. **Configuration Loading**

```python
# Standard pattern for new configuration options
def get_engine_args():
    args = DEFAULT_ARGS
    args.update(os.environ)  # Environment override
    args.update(get_local_args())  # Baked model override
    return match_vllm_args(args)  # Validate against vLLM
```

### 2. **Error Handling**

```python
# Standard error response pattern
def create_error_response(message: str, err_type: str = "BadRequestError"):
    return ErrorResponse(message=message, type=err_type)
```

### 3. **Async Generation**

```python
# Standard streaming pattern
async def generate(self, job_input: JobInput):
    async for batch in self._generate_vllm(...):
        yield batch  # Batch-level yielding for efficiency
```

## Extension Points

### 1. **New Model Architectures**

- **Engine Args**: Add new parameters in `engine_args.py`
- **Compatibility**: Update vLLM argument mapping
- **Validation**: Add architecture-specific validation

### 2. **New API Features**

- **Engine Extension**: Extend `vLLMEngine` or `OpenAIvLLMEngine`
- **Input Parsing**: Extend `JobInput` class
- **Response Format**: Add new response generators

### 3. **Performance Optimizations**

- **Batching Strategy**: Modify `BatchSize` class
- **Memory Management**: Add new caching strategies
- **Hardware Optimization**: GPU-specific optimizations

## Security & Best Practices

### 1. **Secret Management**

- **Build Secrets**: Docker secrets for HF tokens
- **Runtime Secrets**: Environment variable injection
- **Token Handling**: Secure authentication patterns

### 2. **Resource Limits**

- **Memory Bounds**: Configurable GPU memory limits
- **Request Limits**: Concurrency and timeout controls
- **Model Safety**: Trust remote code flags

### 3. **Logging Security**

- **Sanitization**: No secrets in logs
- **Request Logging**: Configurable request/response logging
- **Performance Monitoring**: Safe metrics collection

---

This guide should be consulted whenever working on the worker-vllm codebase to ensure consistency with established patterns and architectural decisions.
