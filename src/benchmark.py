import concurrent.futures
import requests
import time
import os

RUNPOD_ENDPOINT = os.environ('RUNPOD_ENDPOINT')
RUNPOD_API_KEY = os.environ('RUNPOD_API_KEY')

url = "https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/run"

headers = {
    "Authorization": RUNPOD_API_KEY,
    "Content-Type": "application/json"
}

prompt = """
Write me an essay about how the french revolution impacted the rest of europe over the 18th century. 
"""

payload = {
    "input": {
        "stream": True,
        "prompt": prompt,
        "sampling_params": {
            "max_tokens": 1000,
            "n": "1",
            "presence_penalty": "0.2",
            "frequency_penalty": "0.7",
            "temperature": "0.3",
        }
    }
}


def make_request(url, headers, payload):
    response = requests.post(url, headers=headers, json=payload)
    return response

while True:
    # Number of concurrent requests to make per second.
    num_requests = 100

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_request, url, headers, payload)
                   for _ in range(num_requests)]

    # Wait for all requests to complete
    for future in concurrent.futures.as_completed(futures):
        response = future.result()
        # Handle response as needed
        print(response.status_code)

    # Sleep for 1 second before starting the next iteration
    time.sleep(1)
