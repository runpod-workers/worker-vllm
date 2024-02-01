import requests
import json
import time
import os

ENDPOINT_ID = "pst10x7hvwoz2k"
RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')

test_payload = {
        "messages":  [
            {"role": "user", "content": "Write me a 3000 word long and detailed essay about how the french revolution impacted the rest of europe over the 18th century."},
        ],
        "batch_size": 2, # How many tokens to yield per batch
        "apply_chat_template": True,
        "sampling_params": {
            "max_tokens": 4,
            "temperature": 0,
            "ignore_eos": True,
            "n":1
        },
        "stream": True,
        "use_openai_format": True
}

base_url = f'https://api.runpod.ai/v2/{ENDPOINT_ID}'
run_url = base_url + "/run"
headers = {
    'Authorization': f'Bearer {RUNPOD_API_KEY}',
    'Content-Type': 'application/json',
}

job_id = requests.post(run_url, headers=headers, data=json.dumps({"input":test_payload})).json()["id"]

stream_url = base_url + f'/stream/{job_id}'
response = {}

while response.get('status') != 'COMPLETED':
    response = requests.get(stream_url, headers=headers).json()
    stream = response.get('stream', [])
    if stream:
        batch_data = ""
        for chunk in stream:
            batch_data += chunk["output"]
        print(batch_data, end="")
    time.sleep(0.1)