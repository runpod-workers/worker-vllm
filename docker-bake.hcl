variable "PUSH" {
  default = "false"
}

variable "REPOSITORY" {
  default = "runpod"
}

variable "BASE_IMAGE_VERSION" {
  default = "1.0.0"
}

group "all" {
  targets = ["base", "main"]
}

group "base" {
  targets = ["base-11.8.0", "base-12.1.0"]
}

group "main" {
  targets = ["worker-11.8.0", "worker-12.1.0"]
}

target "base-11.8.0" {
  tags = ["${REPOSITORY}/worker-vllm:base-${BASE_IMAGE_VERSION}-cuda11.8.0"]
  context = "vllm-base-image"
  dockerfile = "Dockerfile"
  args = {
    WORKER_CUDA_VERSION = "11.8.0"
  }
  output = ["type=docker,push=${PUSH}"]
}

target "base-12.1.0" {
  tags = ["${REPOSITORY}/worker-vllm:base-${BASE_IMAGE_VERSION}-cuda12.1.0"]
  context = "vllm-base-image"
  dockerfile = "Dockerfile"
  args = {
    WORKER_CUDA_VERSION = "12.1.0"
  }
  output = ["type=docker,push=${PUSH}"]
}

target "worker-11.8.0" {
  tags = ["${REPOSITORY}/worker-vllm:worker-${BASE_IMAGE_VERSION}-cuda11.8.0"]
  context = "."
  dockerfile = "Dockerfile"
  args = {
    BASE_IMAGE_VERSION = "${BASE_IMAGE_VERSION}"
    WORKER_CUDA_VERSION = "11.8.0"
  }
  output = ["type=docker,push=${PUSH}"]
}

target "worker-12.1.0" {
  tags = ["${REPOSITORY}/worker-vllm:worker-${BASE_IMAGE_VERSION}-cuda12.1.0"]
  context = "."
  dockerfile = "Dockerfile"
  args = {
    BASE_IMAGE_VERSION = "${BASE_IMAGE_VERSION}"
    WORKER_CUDA_VERSION = "12.1.0"
  }
  output = ["type=docker,push=${PUSH}"]
}
