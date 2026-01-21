variable "DOCKERHUB_REPO" {
  default = "madiatorlabs"
}

variable "DOCKERHUB_IMG" {
  default = "worker-v1-vllm"
}

variable "RELEASE_VERSION" {
  default = "dev-checks4"
}

variable "HUGGINGFACE_ACCESS_TOKEN" {
  default = ""
}

group "default" {
  targets = ["worker-vllm"]
}

target "worker-vllm" {
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
}