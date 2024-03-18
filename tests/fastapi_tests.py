"""
Instructions:
1. Run middleware.py
2. change path to HF cache
3. run pytest tests/fastapi_tests.py -rsx -vvv --disable-warningspytest tests/fastapi_tests.py -rsx -vvv --disable-warnings 
4. Change wait time in line 54 if needed, it should be a bit above handler initialization
"""

import os
import subprocess
import time

import sys
import pytest
import requests
import ray  # using Ray for overall ease of process management, parallel requests, and debugging.
import openai  # use the official client for correctness check
from huggingface_hub import snapshot_download  # downloading lora to test lora requests

MAX_SERVER_START_WAIT_S = 600  # wait for server to start for 60 seconds
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
PATH_TO_HF_CACHE= "/devdisk/.cache/huggingface/hub/"
LORA_NAME = "typeof/zephyr-7b-beta-lora"  # technically this needs Mistral-7B-v0.1 as base, but we're not testing generation quality here
pytestmark = pytest.mark.asyncio

MODELS = [
    #"mistralai/Mistral-7B-Instruct-v0.1",
    "NousResearch/Llama-2-7b-hf",
    #"facebook/opt-125m",
]

PRECISION = [
    "half",
    #"float16",
    #"bfloat16"
]

QUANTIZATION = [
    False,
    #"awq",
    #"gptq",
    #"squeezellm"
]

TP = [
    False,
    #True
]

MAX_TOKENS = [
    100,
    1024,
    2048
]

@ray.remote(num_gpus=1)
class ServerRunner:
    def __init__(self, args):
        env = os.environ.copy()
        env["HF_HUB_CACHE"] = "1"
        env["MODEL_NAME"]=args["model_name"]
        env["BASE_PATH"]=PATH_TO_HF_CACHE
        env["DTYPE"]="bfloat16"
        env["MAX_MODEL_LENGTH"]="2048"
        env["ENFORCE_EAGER"]="1"
        env["CUSTOM_CHAT_TEMPLATE"]=CHAT_TEMPLATE
    
        self.proc = subprocess.Popen(
            ["python3.10", "src/handler.py", "--rp_serve_api"],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server()

    def ready(self):
        return True

    def _wait_for_server(self):
        # run health check
        time.sleep(15)
        # while True:
        #     try:
        #         if requests.get(
        #                 "http://0.0.0.0:8888/health").status_code == 200:
        #             break
        #     except Exception as err:
        #         if self.proc.poll() is not None:
        #             raise RuntimeError("Server exited unexpectedly.") from err

        #         time.sleep(5)
        #         # if time.time() - start > MAX_SERVER_START_WAIT_S:
        #         #     raise RuntimeError(
        #         #         "Server failed to start in time.") from err

    def __del__(self):
        if hasattr(self, "proc"):
            self.proc.terminate()

@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("precision", PRECISION)
@pytest.mark.parametrize("quantization", QUANTIZATION)
@pytest.mark.parametrize("use_tp", TP)
@pytest.mark.parametrize("max_tokens")
async def test_single_completion(model_name, precision, quantization, use_tp, max_tokens):
    env = os.environ.copy()

    ray.shutdown()
    ray.init()

    params = [
        "--model",
        model_name,
        "--dtype",
        precision,
        "--enforce-eager", 
    ]

    if use_tp:
        params += ["--tensor-parallel-size", 1]
    
    if quantization:
        params += ["--quantization", quantization]

    server = ServerRunner.remote(
        {'model_name': model_name}
    )

    ray.get(server.ready.remote())
    client = openai.AsyncOpenAI(
        base_url="http://0.0.0.0:8888/v1",
        api_key="token-abc123",
    )
    completion = await client.completions.create(model=model_name,
                                                 prompt="Hello, my name is",
                                                 max_tokens=5,
                                                 temperature=0.0)
    print(completion)
    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1
    assert completion.choices[0].text is not None and len(
        completion.choices[0].text) >= 5
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11)

    # test using token IDs
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    assert completion.choices[0].text is not None and len(
        completion.choices[0].text) >= 5

    ray.shutdown()

@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("precision", PRECISION)
@pytest.mark.parametrize("quantization", QUANTIZATION)
@pytest.mark.parametrize("use_tp", TP)
async def test_single_chat_session(model_name, precision, quantization, use_tp):
    env = os.environ.copy()

    ray.shutdown()
    ray.init()

    params = [
        "--model",
        model_name,
        "--dtype",
        precision,
        "--enforce-eager", 
    ]

    if use_tp:
        params += ["--tensor-parallel-size", 1]
    
    if quantization:
        params += ["--quantization", quantization]

    server = ServerRunner.remote(
        {'model_name': model_name}
    )

    ray.get(server.ready.remote())
    client = openai.AsyncOpenAI(
        base_url="http://0.0.0.0:8888/v1",
        api_key="token-abc123",
    )
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
    )
    assert chat_completion.id is not None
    assert chat_completion.choices is not None and len(
        chat_completion.choices) == 1
    assert chat_completion.choices[0].message is not None
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0

@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("precision", PRECISION)
@pytest.mark.parametrize("quantization", QUANTIZATION)
@pytest.mark.parametrize("use_tp", TP)
async def test_completion_streaming(model_name, precision, quantization, use_tp):
    env = os.environ.copy()

    ray.shutdown()
    ray.init()

    params = [
        "--model",
        model_name,
        "--dtype",
        precision,
        "--enforce-eager", 
    ]

    if use_tp:
        params += ["--tensor-parallel-size", 1]
    
    if quantization:
        params += ["--quantization", quantization]

    server = ServerRunner.remote(
        {'model_name': model_name}
    )

    ray.get(server.ready.remote())
    client = openai.AsyncOpenAI(
        base_url="http://0.0.0.0:8888/v1",
        api_key="token-abc123",
    )
    prompt = "What is an LLM?"

    single_completion = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
    )
    single_output = single_completion.choices[0].text
    single_usage = single_completion.usage

    stream = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.usage == single_usage
    assert "".join(chunks) == single_output

@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("precision", PRECISION)
@pytest.mark.parametrize("quantization", QUANTIZATION)
@pytest.mark.parametrize("use_tp", TP)
async def test_chat_streaming(model_name, precision, quantization, use_tp):
    env = os.environ.copy()

    ray.shutdown()
    ray.init()

    params = [
        "--model",
        model_name,
        "--dtype",
        precision,
        "--enforce-eager", 
    ]

    if use_tp:
        params += ["--tensor-parallel-size", 1]
    
    if quantization:
        params += ["--quantization", quantization]

    server = ServerRunner.remote(
        {'model_name': model_name}
    )

    ray.get(server.ready.remote())
    client = openai.AsyncOpenAI(
        base_url="http://0.0.0.0:8888/v1",
        api_key="token-abc123",
    )
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
    )
    print(chat_completion)
    output = chat_completion.choices[0].message.content
    stop_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        stream=True,
    )
    chunks = []
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            chunks.append(delta.content)
    assert chunk.choices[0].finish_reason == stop_reason
    assert "".join(chunks) == output

@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("precision", PRECISION)
@pytest.mark.parametrize("quantization", QUANTIZATION)
@pytest.mark.parametrize("use_tp", TP)
async def test_batch_completions(model_name, precision, quantization, use_tp):
    # test simple list

    env = os.environ.copy()

    ray.shutdown()
    ray.init()

    params = [
        "--model",
        model_name,
        "--dtype",
        precision,
        "--enforce-eager", 
    ]

    if use_tp:
        params += ["--tensor-parallel-size", 1]
    
    if quantization:
        params += ["--quantization", quantization]

    server = ServerRunner.remote(
        {'model_name': model_name}
    )

    ray.get(server.ready.remote())
    client = openai.AsyncOpenAI(
        base_url="http://0.0.0.0:8888/v1",
        api_key="token-abc123",
    )

    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(batch.choices) == 2
    assert batch.choices[0].text == batch.choices[1].text

    # test n = 2
    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        n=2,
        max_tokens=5,
        temperature=0.0,
        extra_body=dict(
            # NOTE: this has to be true for n > 1 in vLLM, but not necessary for official client.
            use_beam_search=True),
    )
    assert len(batch.choices) == 4
    assert batch.choices[0].text != batch.choices[
        1].text, "beam search should be different"
    assert batch.choices[0].text == batch.choices[
        2].text, "two copies of the same prompt should be the same"
    assert batch.choices[1].text == batch.choices[
        3].text, "two copies of the same prompt should be the same"

    # test streaming
    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        max_tokens=5,
        temperature=0.0,
        stream=True,
    )
    texts = [""] * 2
    async for chunk in batch:
        assert len(chunk.choices) == 1
        choice = chunk.choices[0]
        texts[choice.index] += choice.text
    assert texts[0] == texts[1]

if __name__ == "__main__":
    pytest.main([__file__])
