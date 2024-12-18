# ResNet50-Food101 on NVIDIA H100/A100

This repository provides scripts and resources for training the HuggingFace Food101 dataset on the Microsoft ResNet50 model.

## Overview

- **Training and Inference Scripts**:  
  The PyTorch/TorchRun scripts located in the `training` and `inference` folders have been tested on a multi-node NVIDIA H100/A100 setup using the Slurm job manager for training. Inference was tested on a single GPU (H100/A100).  

- **Step-by-Step Validation**:  
  A Jupyter notebook is included to validate the step-by-step training of this model and dataset. The notebook has been tested on Google Colab with a Tesla T4 GPU.

- **Well-Documented Code**:  
  The training script contains detailed comments explaining the flow and rationale behind various code blocks.

## Docker Integration

- **Prebuilt Containers**:  
  Both the training and inference scripts are packaged as Docker containers (see the `Dockerfile`), which include all the required dependencies.

- **Docker commands used to build the training and inference containers**
    - *Training docker image* : cd training; docker build -t inference-app:latest .

    - *Inference docker image* : cd inference; docker build -t food101-inference:latest -f Dockerfile.inf  .

- **Running Docker**:  
  Check the `docker run` commands in the Slurm scripts to ensure that the necessary host directories are available on GPU worker nodes. Alternatively, modify the paths to match your directory structure.  
  Example:  
  Host directory `/mnt/weka/tmp/resnet50demo` is mounted inside the container at `/app`, where the Python scripts are read.

## Slurm Setup

- **Temporary Folders**:  
  The Slurm scripts require additional folders for temporary files:  
    - Line 18: `logs` directory  
    - Line 35: environment variables  

  Ensure these folders are created and accessible on all worker GPU nodes.

## Things that can be looked at

- **Migrating to Kubernetes or OpenShift**:  
  Migrating this setup to Kubernetes or OpenShift using the machine learning training frameworks provided by these container orchestration platforms could be an exciting next step!
