#!/bin/bash -v
#SBATCH --output=slurm-%j.out
#SBATCH --job-name=multi_node_training       # Job name
#SBATCH --nodes=4                            # Number of nodes
#SBATCH --ntasks-per-node=1                  # Tasks (containers) per node
#SBATCH --gres=gpu:8                         # Number of GPUs per node
#SBATCH --cpus-per-task=4                    # CPUs per task
#SBATCH --time=04:00:00                      # Max runtime
#SBATCH --partition=H100-RAILS-ALL           # GPU partition
export MASTER_ADDR=$(ifconfig gpu0_eth | awk '/inet / {gsub("inet", "", $2); print $2}')  # IP address of the GPU0 interface on the 1st worker node.
export MASTER_PORT=29500  # Set master port
export BATCH_SIZE=16
export EPOCHS=20

# Enable NCCL logs for GPU communication
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_FILE="/mnt/weka/tmp/resnet50demo/results/plots/nccl_logs_${SLURM_NODEID}.txt"
export NCCL_TOPO_DUMP_FILE="nccl_topology.xml"


# Load Docker module
module load docker

# First command: Run `printenv` on each node
srun --ntasks=4 --ntasks-per-node=1 bash -c 'printenv > /home/vshenoy/slurm_env_files/${SLURM_JOBID}_${SLURM_NODEID}.env'

srun --ntasks=4 --ntasks-per-node=1 bash -c '
    sudo docker run \
        --rm \
        --gpus all \
        --net host \
        --shm-size=2g \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /mnt/weka/tmp/resnet50demo:/app \
        --env-file /home/vshenoy/slurm_env_files/${SLURM_JOBID}_${SLURM_NODEID}.env \
        inference-app:latest \
        torchrun \
            --nnodes=$SLURM_NNODES \
            --node_rank=$SLURM_NODEID \
            --nproc_per_node=$SLURM_GPUS_ON_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            /app/resnet50_training_torchrun.py \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS'
