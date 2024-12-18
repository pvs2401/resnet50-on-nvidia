# Base image: NVIDIA's PyTorch container
FROM nvcr.io/nvidia/pytorch:23.09-py3

# Set the working directory inside the container
WORKDIR /app

# Copy requirements (if any additional libraries are specified)
#COPY requirements.txt ./

# Update pip and install required Python libraries
RUN python -m pip install --upgrade pip && \
    pip install datasets && \
    pip install --upgrade Pillow torchvision torchmetrics && \
    pip install --force-reinstall opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64 && \
    pip install albumentations && \
    pip install deepspeed && \
    apt-get update && \
    apt-get install -y \
    vim \
    net-tools \
    iproute2 \
    procps \
    lsof \
    iputils-ping \
    traceroute \
    dnsutils \
    net-tools \
    && apt-get clean

# Mount a volume for external data sharing
VOLUME /mnt/weka/tmp/vshenoy:/app

# Expose the necessary port
EXPOSE 9999

# Set the default working directory inside the container
WORKDIR /app

# Default command when the container starts
CMD ["bash"]

