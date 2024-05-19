# Instructions for using Multi-Modal LLMs in vLLM on RunPod Serverless

## Setup

### 1. Change the container image
Change container image from `runpod/worker-vllm:stable-...` to `alpayariyakrunpod/worker-vllm:multimodal` in the endpoint template.

### Set Environment Variables
Set the following environment variables in the endpoint template:
- `IMAGE_FEATURE_SIZE`
- `IMAGE_INPUT_SHAPE`
- `IMAGE_TOKEN_ID`
- `ENABLE_MULTIMODAL`=`1`


## EXAMPLE: llava-1.5
### Environment Variables:
    ```
    IMAGE_TOKEN_ID=32000
    IMAGE_INPUT_SHAPE="1,3,336,336"
    IMAGE_FEATURE_SIZE=576
    ENABLE_MULTIMODAL=1```

### Usage
1. Import libraries, set your RunPod API key, and Endpoint ID. 
```python
import os 
import time
import requests

ENDPOINT_ID = "<YOUR ENDPOINT ID HERE>"
RUNPOD_API_KEY = "<YOUR RUNPOD API KEY HERE>"

header = {
    "Authorization": "Bearer " + RUNPOD_API_KEY,
}
runpod_endpoint = "https://api.runpod.ai/v2/" + ENDPOINT_ID 
```

2. Set your feature size(model-specific), message, image URL(or base64 encoding of it, will show how to use that later), and optionally sampling parameters.
```python
feature_size = 576
message = "The image is from RunPod's blogpost, what is it about?"
image_url = "https://blog.runpod.io/content/images/size/w2000/2024/05/IMG_4619-1.png"

# Example sampling parameters
max_tokens = 4096
temperature = 0.7


test_input = {
    "input": {
        "prompt": "<image>" * feature_size + "\nUSER: " + message + "\nASSISTANT:",
        "image_url":  image_url,
        "sampling_params": {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    }
}
```

3. Send a request to the endpoint.
- NON-STREAMING:
    ```python
    response = requests.post(runpod_endpoint + "/runsync", json=test_input, headers=header) # /runsync is synchronous, you can do async non-streaming with /run
    print("\nNON-STREAMING:\n", response.json()["output"][0]["choices"][0]["tokens"][0])
    ```
    Output:
    
    >The image is an advertisement from RunPod, announcing a 20 million seed round. This indicates that RunPod has recently secured funding to grow their business, potentially related to running podcasts or providing digital solutions for the running community. The announcement likely aims to attract more investors, partners, or customers to join their cause and contribute to the development of their company.

- STREAMING:
    ```python
    test_input["input"]["stream"] = True
    job_id = requests.post(runpod_endpoint + "/run", json=test_input, headers=headers).json()["id"]
    print("\nSTREAMING:")
    while True:
        response = requests.get(runpod_endpoint + "/status/" + job_id, headers=headers)
        stream = response.json().get("output")
        if stream:
            for batch in stream:
                for token in batch["choices"][0]["tokens"]:
                    print(token, end="", flush=True)
        
        if response.json()["status"] == "COMPLETED":
            break
        time.sleep(0.1)```

