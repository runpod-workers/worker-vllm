import requests
import json
import time
import os

url = "https://api.runpod.ai/v2/4hlrhh430u5tz7/runsync"

headers = {
    "Authorization":"UGXLPIBFYJXUCKOBNR8CCJ3KP8GWJNI97F91QT6Y",
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

import concurrent.futures
import time

def make_request(url, headers, payload):
    response = requests.post(url, headers=headers, json=payload)
    return response

url = "https://api.runpod.ai/v2/4hlrhh430u5tz7/status/1a73bdc5-aede-4c0a-8fa2-2ac2fb3010dc"

while True:
    # Number of concurrent requests to make
    num_requests = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_request, url, headers, payload) for _ in range(num_requests)]

    # Wait for all requests to complete
    for future in concurrent.futures.as_completed(futures):
        response = future.result()
        # Handle response as needed
        print(response.json())
        print(response.status_code)

    # Sleep for 1 second before starting the next iteration
    time.sleep(1)


# response_json = json.loads(response.text)

# get_status = requests.get(status_url, headers=headers)
# print(get_status.text)