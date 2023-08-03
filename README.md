<div align="center">

<h1>vLLM Endpoint | Serverless Worker </h1>

[![CI | Test Worker](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_worker.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/CI-test_worker.yml)
&nbsp;
[![Docker Image](https://github.com/runpod-workers/worker-template/actions/workflows/CD-docker_dev.yml/badge.svg)](https://github.com/runpod-workers/worker-template/actions/workflows/CD-docker_dev.yml)

🚀 | This serverless worker utilizes vLLM (very Large Language Model) behind the scenes and is integrated into RunPod's serverless environment. It supports dynamic auto-scaling using the built-in RunPod autoscaling feature.
</div>

#### Docker Arguments:
1. `HUGGING_FACE_HUB_TOKEN`: Your private Hugging Face token. This token is required for downloading models that necessitate agreement to an End User License Agreement (EULA), such as the llama2 family of models.
2. `MODEL_NAME`: The Hugging Face model to use. Please ensure that the chosen model is supported by vLLM. Refer to the list of supported models for compatibility.
3. `TOKENIZER`: (Optional) The specified tokenizer to use. If you want to use the default tokenizer for the model, do not provide this docker argument at all.
4. `STREAMING`: Whether to use HTTP Streaming or not. Specify True if you want to enable HTTP Streaming; otherwise, omit this argument.

#### llama2 7B Chat:
`docker build . --platform linux/amd64 --build-arg HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here --build-arg MODEL_NAME=meta-llama/Llama-2-7b-chat-hf --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=True`

#### llama2 13B Chat:
`docker build . --platform linux/amd64 --build-arg HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here --build-arg MODEL_NAME=meta-llama/Llama-2-13b-chat-hf --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer --build-arg STREAMING=True`

Please make sure to replace your_hugging_face_token_here with your actual Hugging Face token to enable model downloads that require it.

Ensure that you have Docker installed and properly set up before running the docker build commands. Once built, you can deploy this serverless worker in your desired environment with confidence that it will automatically scale based on demand. For further inquiries or assistance, feel free to contact our support team.
