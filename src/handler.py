#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''
from typing import Dict

import json
import requests
from urllib.parse import urljoin

import runpod
from vllm.entrypoints.runpod.api_server import start_vllm_runpod

# Start the VLLM serving layer on our RunPod worker.
vllm = start_vllm_runpod(
    served_model='facebook/opt-125m', port=443, host='127.0.0.1')


def prepare_request(event: Dict) -> Dict:
    """
        # Pre-processing Steps for the Prompt
        # Include any necessary code here to pre-process the prompt.

        # Example:
        # Step 1: Clean the prompt
        # cleaned_prompt = clean_text(prompt)

        # Step 2: Apply specific transformations
        # transformed_prompt = apply_transformations(tokenized_prompt)

        # Step 3: Format the prompt for the model
        # formatted_prompt = format_for_model(transformed_prompt)
    """

    request_data = {
        'url': urljoin("http://127.0.0.1:443/", event['llm_endpoint']['url']),
        'headers': {
            "Content-Type": "application/json"
        },
        'request_type': event['llm_endpoint']['request_type'],
        'body_data': event['llm_body']
    }

    return request_data


def make_vllm_request(request_data: Dict) -> requests.Response:
    url = request_data['url']
    headers = request_data['headers']
    request_type = request_data['request_type']

    if request_type.lower() == 'post':
        body_data = request_data['body_data']
        response = requests.post(url, headers=headers,
                                 data=json.dumps(body_data))
    elif request_type.lower() == 'get':
        response = requests.get(url, headers=headers)
    else:
        raise ValueError(f"Invalid request type: {request_type}")

    return response


def process_response(response: requests.Response) -> Dict:
    # Process the json response.
    response_data = response.json()

    """
        # Additional Post-processing Steps
        # You can include any necessary code here to process the LLM's generated output.

        # Example:
        # Step 1: Extract relevant information
        # result = llm_output['data']['result']
        # relevant_info = result['info']

        # Step 2: Clean the data
        # cleaned_data = preprocess(relevant_info)

        # Step 3: Apply transformations or filters
        # transformed_data = apply_transformations(cleaned_data)

        # Step 4: Finalize the output
        # final_output = format_output(transformed_data)

        # Return the final processed output
        return final_output
    """

    return response_data


def handler(event):
    '''
    This is the handler function that will be called by the serverless worker.
    '''
    # Prepare the request for vllm.
    request_data = prepare_request(event)

    # Make the request.
    response = make_vllm_request(request_data)

    # Process the response from vllm.
    response_data = process_response(response)

    return response_data


"""
Provide access to our custom handler, which allows us to incorporate pre-processing and 
post-processing steps into the prompt. This custom handler enhances the functionality 
of our program by allowing us to perform additional tasks before and after the prompt execution.

Furthermore, we pass the 'runpod vllm' instance to ensure efficient auto-scaling based on 
the usage of the vllm (very large language model). This inclusion enables the program to 
dynamically adjust its resource allocation to accommodate the demands of the vllm, 
optimizing its performance and scalability.
"""
runpod.serverless.start(
    {"handler": handler, "vllm": vllm}, serverless_llm=True)
