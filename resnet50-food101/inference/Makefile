# Makefile for running the Food101 inference backend with Docker

# Variables
DOCKER_IMAGE=food101-inference:latest
SCRIPT=/app/resnet50_inference_torchrun.py
APP_DIR=/mnt/weka/tmp/resnet50demo
SHM_SIZE=2g
NETWORK_MODE=host
DOCKER_SOCKET=/var/run/docker.sock
DOCKERFILE_INF=$(APP_DIR)/Dockerfile.inf
DOCKERFILE=$(APP_DIR)/Dockerfile

# Targets
.PHONY: run clean build-food101 build-inference-app

# Help
help:
	@echo "Available tasks:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | grep -vE '^#' | awk -F':' '{print "  " $1}'

# Run the Docker container
run:
	@echo "Running the Docker container..."
	@sudo docker run --gpus all \
		--net $(NETWORK_MODE) \
		--shm-size=$(SHM_SIZE) \
		-v $(DOCKER_SOCKET):$(DOCKER_SOCKET) \
		-v $(APP_DIR):/app \
		$(DOCKER_IMAGE) python $(SCRIPT)

# Clean up Docker containers and images (optional)
clean:
	@echo "Stopping and removing all containers..."
	@sudo docker stop $$(sudo docker ps -q) || true
	@sudo docker rm $$(sudo docker ps -a -q) || true
	@echo "Removing all dangling images..."
	@sudo docker rmi $$(sudo docker images -f "dangling=true" -q) || true

# Build food101-inference image
build-inference-docker:
	@echo "Building the food101-inference Docker image..."
	@sudo docker build -t food101-inference:latest -f $(DOCKERFILE_INF) $(APP_DIR)

# Build inference-app image
build-training-docker:
	@echo "Building the inference-app Docker image..."
	@sudo docker build -t inference-app:latest -f $(DOCKERFILE) $(APP_DIR)
