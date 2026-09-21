# Worker image = official vLLM OpenAI server image + RunPod serverless wrapper.
# vLLM upgrades are now a single build ARG:
#   docker buildx build --build-arg VLLM_VERSION=v0.23.0 ...
ARG VLLM_VERSION=v0.28.0
FROM vllm/vllm-openai:${VLLM_VERSION}
# Re-declare so the stage can reference it in RUN steps below.
ARG VLLM_VERSION

# RunPod serverless SDK + HTTP proxy deps (vLLM itself comes from the base image).
COPY builder/requirements.txt /requirements.txt
RUN python3 -m ensurepip --upgrade 2>/dev/null || true \
    && python3 -m pip install --no-cache-dir -r /requirements.txt

# bitsandbytes moved out of vLLM core into the out-of-tree vllm-bnb-plugin in
# v0.28.0 (vllm-project/vllm#43529). The --no-deps is load-bearing: the plugin
# declares an UNPINNED `vllm` dependency, and letting pip resolve it replaces
# the base image's vLLM — it once backtracked to vllm 0.19.1, producing an
# image tagged v0.28.0 that actually contained 0.19.1 (broken deep_ep/NCCL
# mix). Its only real runtime need is the `bitsandbytes` package, pinned in
# builder/requirements.txt.
RUN python3 -m pip install --no-cache-dir --no-deps "vllm-bnb-plugin>=0.0.3"

# Guard rail: the base image's vLLM must survive our pip installs — fail the
# build loudly if the installed version no longer matches the requested tag.
RUN test "$(python3 -c 'import vllm; print(vllm.__version__)')" = "${VLLM_VERSION#v}"

# Setup for Option 2: building the image with the model baked in.
ARG MODEL_NAME=""
ARG MODEL_REVISION=""
ARG TOKENIZER_NAME=""
ARG TOKENIZER_REVISION=""
ARG QUANTIZATION=""
ARG BASE_PATH="/runpod-volume"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    QUANTIZATION=$QUANTIZATION \
    BASE_PATH=$BASE_PATH \
    # The RunPod network volume mounts at $BASE_PATH; keep the HF cache there so
    # model downloads persist across worker cold starts.
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false

COPY src /src

# Optionally bake the model into the image at build time. Pass the
# HF_TOKEN build secret if the repo is gated:
#   docker buildx build --secret id=HF_TOKEN ... --build-arg MODEL_NAME=...
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -n "$MODEL_NAME" ]; then \
        if [ -f /run/secrets/HF_TOKEN ]; then \
            export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
        fi && \
        python3 /src/download_model.py; \
    fi

# main.py spawns `vllm serve` with args built from the environment, waits for
# /health, then starts the RunPod serverless loop. Explicit ENTRYPOINT so the
# base image's own entrypoint can never swallow our command.
ENTRYPOINT ["python3", "/src/main.py"]
