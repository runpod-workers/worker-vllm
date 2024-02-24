#!/bin/bash

git clone https://github.com/runpod/vllm-fork-for-sls-worker.git

cp -r vllm-fork-for-sls-worker vllm-12.1.0
cp -r vllm-fork-for-sls-worker vllm-11.8.0
rm -rf vllm-fork-for-sls-worker

cd vllm-11.8.0
git checkout cuda-11.8

echo "vLLM Base Image Builder Setup Complete."