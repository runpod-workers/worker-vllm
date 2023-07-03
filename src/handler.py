#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''

import runpod

# Load models into VRAM here so they can be warm between requests


def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    print(event)

    # do the things

    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return "Hello World"


runpod.serverless.start({"handler": handler})
