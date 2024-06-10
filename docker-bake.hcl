variable "PUSH" {
  default = "true"
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
  targets = ["base-1180", "base-1210"]
}

group "main" {
  targets = ["worker-1180", "worker-1210"]
}

target "base-1180" {
  tags = ["${REPOSITORY}/worker-vllm:base-${BASE_IMAGE_VERSION}-cuda11.8.0"]
  context = "vllm-base-image"
  dockerfile = "Dockerfile"
  args = {
    WORKER_CUDA_VERSION = "11.8.0"
  }
  output = ["type=docker,push=${PUSH}"]
}

target "base-1210" {
  tags = ["${REPOSITORY}/worker-vllm:base-${BASE_IMAGE_VERSION}-cuda12.1.0"]
  context = "vllm-base-image"
  dockerfile = "Dockerfile"
  args = {
    WORKER_CUDA_VERSION = "12.1.0"
  }
  output = ["type=docker,push=${PUSH}"]
}

target "worker-1180" {
  tags = ["${REPOSITORY}/worker-vllm:${BASE_IMAGE_VERSION}-cuda11.8.0"]
  context = "."
  dockerfile = "Dockerfile"
  args = {
    BASE_IMAGE_VERSION = "${BASE_IMAGE_VERSION}"
    WORKER_CUDA_VERSION = "11.8.0"
  }
  output = ["type=docker,push=${PUSH}"]
}

target "worker-1210" {
  tags = ["${REPOSITORY}/worker-vllm:${BASE_IMAGE_VERSION}-cuda12.1.0"]
  context = "."
  dockerfile = "Dockerfile"
  args = {
    BASE_IMAGE_VERSION = "${BASE_IMAGE_VERSION}"
    WORKER_CUDA_VERSION = "12.1.0"
  }
  output = ["type=docker,push=${PUSH}"]
}
