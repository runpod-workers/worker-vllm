#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''
import runpod
import requests
import json
from typing import Dict

def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    # Prepare the request
    llm_endpoint_data: Dict = event['llm_endpoint']
    llm_body_data: Dict = event['llm_body']

    llm_endpoint_url: str = llm_endpoint_data['url']
    llm_endpoint_request_type: str = llm_endpoint_data['request_type']

    url = "http://127.0.0.1:443/" + llm_endpoint_url
    headers = {
        "Content-Type": "application/json"
    }

    # Process the request
    if llm_endpoint_request_type.lower() == 'post':
        response = requests.post(url, headers=headers,
                                 data=json.dumps(llm_body_data))

    elif llm_endpoint_request_type.lower() == 'get':
        response = requests.get(url, headers=headers)

    response_data = response.json()

    return response_data


runpod.serverless.start({"handler": handler}, serverless_llm=True)
