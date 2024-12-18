# resnet50-food101-on-nvidia H100/A100

* Scripts for training huggingface food101 dataset on MS resnet50 model.

* The PyTorch/TorchRun scripts in the training and inference folders have been tested on a multi-node H100/A100 setup using Slurm job manager for training. The inference was tested on a single GPU ( H100/A100 ).

* The Jupyter notebook can be used to validate the step-step training of this model/dataset on google colab Tesla T4 GPU.

* The training script has enough comments that describes the flow and what-why of various code blocks.

* Both the training and the inference scripts are packaged as docker containers ( check Dockerfile ) which contains the required packages built into the containers. Check the docker run command in the slurm scripts and make sure that the host directories are present GPU worker nodes or uses the path based on users directory structure ( For ex. /mnt/weka/tmp/resnet50demo from where docker reads the python scripts and mounts inside the container @ /app)

* There are other folders used by slurm as placeholders for placing temporary files ( Line 18 : logs, Line 35: env variables etc.). These needs to be present on the worker GPU nodes as well.

* It will be interesting to migrate this to Kubernets or Openshift using the ML training frameworks proides by those CAAS platforms. May be the next step !

