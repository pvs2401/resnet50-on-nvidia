import argparse
import os
import socket
import time
import json
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as models
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from tqdm.auto import tqdm
from helper_functions import Plotter

import warnings
warnings.simplefilter("ignore")

# Constants
MODEL_NAME = "resnet50"
NUM_CLASSES = 101
IMAGE_SIZE = 224
DATASET_NAME = "food101"
MODEL_SAVE_DIR = f"./results/models"
RESULTS_FILE = "./results/plots/results.json"
PLOT_FILE = "./results/plots/results.png"
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
JOB_ID = int(os.environ["SLURM_JOBID"])
BACKEND = "nccl"
HOST_NAME = socket.gethostname()
EARLY_STOPPING_PATIENCE = 5
START_TIME = float('-inf')
DATASET_LABEL_MAPPING_FILE = "./results/food101_label_mapping.json"

#Final training results
metrics_data = {
    "epochs": [],
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "epoch_completion_time": [],
    "early_stopping_counter": 0,
    "best_epoch": None,
    "best_accuracy": float('-inf'),
    "system_model": torch.cuda.get_device_name(),
    "gpus_per_node": torch.cuda.device_count(),
    "node_count": int(WORLD_SIZE / torch.cuda.device_count()),
    "job_completion_time": float('-inf'),
    "job_id": JOB_ID,
    "batch_size": None,
    "trainable_params": None
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size per GPU")
    parser.add_argument("--epochs", type = int, default = 4, help = "Number of epochs to train")
    parser.add_argument("--num_workers", type = int, default = 2, help = "Number of workers for DataLoader")
    parser.add_argument("--dist_url", default = "tcp://127.0.0.1:29500", help = "URL for initializing the process group")
    return parser.parse_args()

# Image Transforms
class ImageTransforms:
    def __init__(self):
        # Initialize the transformation pipeline using Compose to chain multiple transformations
        self.transform = transforms.Compose([
            # Resize the image to 224x224.
            transforms.Resize((224, 224)),
            # Convert the image to a PIL Image.
            transforms.ToImage(),
            # Convert the image data type to float32 and normalize the pixel values to the range [0, 1]
            transforms.ToDtype(torch.float32, scale=True),
            # Normalize the image using the ImageNet mean and standard deviation values
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, examples):
        examples["image"] = [self.transform(image) for image in examples["image"]]
        return examples

# Main Model Class
class ResNetModelClass(nn.Module):
    def __init__(self, num_classes = NUM_CLASSES):
        super(ResNetModelClass, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        self.resnet50.fc = self.fc

    def forward(self, x):
        return self.resnet50(x)

# Main Trainer class
class Trainer:
    def __init__(self, model, device, loss_fn, optimizer,rank):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            metrics_data["trainable_params"] = trainable_params
            print(f"[{device} on {HOST_NAME}] Total parameters: {total_params} Trainable parameters: {trainable_params}")


# Training Step
    def train_step(self, dataloader, rank:int, local_rank: int):
        # Set the model to training mode (activates dropout, batchnorm, etc.)
        device = torch.device(f"cuda:{local_rank}")
        self.model.train()

        # Initialize variables to accumulate the training loss and accuracy
        train_loss, train_acc = 0, 0

        # On GPU0 Log the total number of batches processed per GPU
        if rank == 0:
            print(f"[{device} on {HOST_NAME}] Total Training Batches To Be Processed: {len(dataloader)}")

        for batch_idx, batch in enumerate(tqdm(dataloader,
                                                dynamic_ncols = True,
                                                unit = 'batch',
                                                total = len(dataloader),
                                                desc = f"[[{device} on {HOST_NAME}]] Training Progress")):

            # Move images and labels to the appropriate device (GPU/CPU)        
            X, y = batch["image"].to(self.device), batch["label"].to(self.device)

            if batch_idx == 0 and rank == 0:
                print(f"[{device} on {HOST_NAME}] Batch {batch_idx}: Image Data Shape: {X.shape}, Labels: {y[:5]}, Batch Size: {len(X)}")

            # Zero the gradients of the optimizer before performing backpropagation
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(X)

            # Loss computation
            loss = self.loss_fn(y_pred, y)

            # Backpropagate the loss and update model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            self.optimizer.step()

            # Accumulate the loss and accuracy for this batch
            train_loss += loss.item()
            train_acc += (y_pred.argmax(dim = 1) == y).sum().item() / len(y)

        return train_loss / len(dataloader), train_acc / len(dataloader)

# Validation Step
    def val_step(self, dataloader, rank:int, local_rank:int):

        device = torch.device(f"cuda:{local_rank}")
        # Set the model to evaluation mode (disables dropout, batchnorm, etc.)
        self.model.eval()

        # Initialize variables to accumulate the validation loss and accuracy
        val_loss, val_acc = 0, 0

        if rank == 0:
            print(f"[{device} on {HOST_NAME}] Total Validation Batches To Be Processed: {len(dataloader)}")

        for batch_idx, batch in enumerate(tqdm(dataloader,
                                                unit = 'batch',
                                                total = len(dataloader),
                                                desc = f"[{device} on {HOST_NAME}] Validation Progress")):

            X, y = batch["image"].to(self.device), batch["label"].to(self.device)

            if batch_idx == 0 and rank == 0:
                print(f"[{device} on {HOST_NAME}] Batch {batch_idx}: Image Data Shape: {X.shape},"
                f"Labels: {y[:5]},"
                f"Batch Size: {len(X)}")              
            
            # Forward pass: make predictions for the current batch
            val_pred_logits = self.model(X)

            # Compute the validation loss between predictions and true labels
            loss = self.loss_fn(val_pred_logits, y)

            # Accumulate the validation loss and accuracy for this batch
            val_loss += loss.item()
            val_acc += (val_pred_logits.argmax(dim = 1) == y).sum().item() / len(y)

        return val_loss / len(dataloader), val_acc / len(dataloader)

# Distributed Training Manager
class DistributedTrainer:
    def __init__(self, args, model_class, dataset_name, world_size, backend):
        self.args = args
        self.model_class = model_class
        self.dataset_name = dataset_name
        self.world_size = world_size
        self.backend = backend

    def setup_process_group(self):
        dist.init_process_group(
            backend = "nccl", 
            init_method = "env://",
            rank = RANK, 
            world_size = WORLD_SIZE
        )
        torch.cuda.set_device(LOCAL_RANK)  # Assign GPU to this process
        device = torch.device(f"cuda:{LOCAL_RANK}")
        print(f"Spawning distributed Torch on | HOST: {HOST_NAME} | GPU: {device} | LOCAL_RANK: {LOCAL_RANK} | RANK: {RANK:2d} | ")

    def load_data(self, rank):
        # Load the dataset (train and validation) using Hugging Face's load_dataset method
        dataset = load_dataset(self.dataset_name, cache_dir = "./data")

        transform = ImageTransforms()

        # Apply image transformation class to preprocess images (e.g., resizing, normalization)
        dataset["train"].set_transform(transform)
        dataset["validation"].set_transform(transform)

        ### Save the label mapping to a JSON file for use during inference
        if rank == 0:
            print(f"[{HOST_NAME}] Dataset {DATASET_NAME}: Saving classname-to-label mapping to {DATASET_LABEL_MAPPING_FILE}.")
            label_mapping = dataset["train"].features["label"].names
            with open(DATASET_LABEL_MAPPING_FILE, "w") as f:
                json.dump(label_mapping, f)

        # Initialize a DistributedSampler for training & validation data to ensure data is 
        # split across all distributed workers (GPUs)
        train_sampler = DistributedSampler(dataset["train"], 
                                            num_replicas = self.world_size, 
                                            rank = rank)

        val_sampler = DistributedSampler(dataset["validation"], 
                                            num_replicas = self.world_size, 
                                            rank = rank, 
                                            shuffle = False)

        # Create DataLoader for the training & validation set
        train_loader = DataLoader(dataset["train"], 
                                    batch_size = self.args.batch_size, 
                                    sampler = train_sampler,
                                    num_workers = self.args.num_workers, 
                                    pin_memory = True)

        val_loader = DataLoader(dataset["validation"], 
                                    batch_size = self.args.batch_size, 
                                    sampler = val_sampler,
                                    num_workers = self.args.num_workers, 
                                    pin_memory = True)

        return train_loader, val_loader

    def setup_model(self, rank, device):
        # Initialize the model by creating an instance of the model class and 
        # moving it to the specified device (GPU/CPU)
        model = self.model_class().to(device)

        # Wrap the model with DistributedDataParallel (DDP) for distributed training
        model = DDP(model, 
                    device_ids = [rank], 
                    find_unused_parameters = True)
        return model

def early_stopping(model, epoch, val_acc):
    # Applies early stopping logic and updates metrics data.
    # Saves the best model when a new best accuracy is achieved.
    device = torch.device(f"cuda:{LOCAL_RANK}")
    if metrics_data.get("val_accuracy") and len(metrics_data["val_accuracy"]) > 0 and val_acc > metrics_data["best_accuracy"]:
        # Reset early stopping counter and update best metrics
        metrics_data["early_stopping_counter"] = 0
        metrics_data["best_epoch"] = epoch + 1
        metrics_data["best_accuracy"] = val_acc

        # Save the best model state_dict
        MODEL_STATE = f"{MODEL_SAVE_DIR}/{MODEL_NAME}_best_model_state.pth"
        torch.save(model.module.state_dict(), MODEL_STATE)
        print(f"[{device} on {HOST_NAME}] Best model at EPOCH {epoch+1} saved at {MODEL_STATE}.")
    else:
        metrics_data["early_stopping_counter"] += 1
        print(f"[{device} on {HOST_NAME}] Increment early_stopping_counter to {metrics_data['early_stopping_counter']}" 
              f"due to degrading value accuracy.")
        if metrics_data["early_stopping_counter"] == EARLY_STOPPING_PATIENCE:
            print(f"[{device} on {HOST_NAME}] Early stopping the Model at EPOCH {epoch+1}.")
            return True  # Early stopping triggered

    return False  # Continue training

# Main worker function
def main_worker(args):
    torch.manual_seed(41)
    START_TIME = time.time()
    dist_trainer = DistributedTrainer(args, ResNetModelClass, DATASET_NAME, WORLD_SIZE, BACKEND)
    # Setup distributed process group on GPUs across nodes
    dist_trainer.setup_process_group()
    device = torch.device(f"cuda:{LOCAL_RANK}")
    # Load data and model
    train_loader, val_loader = dist_trainer.load_data(RANK)
    model = dist_trainer.setup_model(LOCAL_RANK, device)
    best_accuracy = 0
    model_freeze = True
    # Optimizer, loss, and scheduler setup

    # Define the optimizer. Using Stochastic Gradient Descent (SGD) with the following parameters:
    # - 'model.parameters()' passes the parameters of the model to the optimizer.
    # - 'lr=0.01' sets the learning rate to 0.01.
    # - 'momentum=0.9' helps accelerate SGD in the relevant direction and dampens oscillations.
    # - 'weight_decay=1e-4' applies L2 regularization to prevent overfitting by penalizing large weights.    
    optimizer = torch.optim.SGD(model.parameters(), 
                                    lr = 0.01, 
                                    momentum = 0.9, 
                                    weight_decay = 1e-4)

    # Define the loss function using Cross-Entropy Loss for classification tasks.
    loss_fn = nn.CrossEntropyLoss()

    # Define the learning rate scheduler. Using ReduceLROnPlateau to adjust the learning rate during training.
    # This scheduler reduces the learning rate when a plateau in the validation loss is detected.
    # The parameters are:
    # - 'optimizer' to be used with the scheduler.
    # - 'mode='min'' indicates the scheduler will reduce the learning rate when the validation loss stops decreasing.
    # - 'patience=2' means the learning rate will only be reduced after 2 epochs without improvement in the validation loss.
    # - 'factor=0.5' reduces the learning rate by half when the plateau is reached.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', 
                                                            patience = 2, 
                                                            factor = 0.5, 
                                                            verbose = True)
    
    trainer = Trainer(model, device, loss_fn, optimizer,RANK)
    # Freeze layers except final for the first 50% of epochs
    print(f"[{device} on {HOST_NAME}] Freezing all the layers except the final fc at epoch 1.")
    for param in model.module.resnet50.parameters():
        param.requires_grad = False
    for param in model.module.resnet50.fc.parameters():
        param.requires_grad = True

    # Main Training loop
    for epoch in range(args.epochs):
        if RANK == 0:
            print(f"[{device} on {HOST_NAME}] Starting epoch {epoch+1}/{args.epochs}...")
        # Create CUDA events for this epoch
        start_event = torch.cuda.Event(enable_timing = True)
        end_event = torch.cuda.Event(enable_timing = True)

        # Unfreeze all layers after 50% of epochs
        if epoch == args.epochs // 2:
            print(f"[{device} on {HOST_NAME}] Unfreezing all layers at epoch {epoch + 1}.")
            model_freeze = False
            for param in model.module.resnet50.parameters():
                param.requires_grad = True

        train_loader.sampler.set_epoch(epoch)

        start_event.record()  # Start timing

        train_loss, train_acc = trainer.train_step(train_loader, RANK, LOCAL_RANK)
        val_loss, val_acc = trainer.val_step(val_loader, RANK, LOCAL_RANK)

        end_event.record()  # End timing
        torch.cuda.synchronize()  # Synchronize to ensure all CUDA operations are complete
        epoch_duration = start_event.elapsed_time(end_event) / 1000.0

        if RANK == 0:
            print(f"[{device} on {HOST_NAME}] Epoch {epoch+1:2d}/{args.epochs} | Train Loss: {train_loss:.4f} |"
                  f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | JCT: {epoch_duration:.4f} |")

            metrics_data["epochs"].append(epoch + 1)
            metrics_data["train_loss"].append(train_loss)
            metrics_data["val_loss"].append(val_loss)
            metrics_data["train_accuracy"].append(train_acc)
            metrics_data["val_accuracy"].append(val_acc)
            metrics_data["epoch_completion_time"].append(epoch_duration)
            metrics_data["batch_size"] = args.batch_size

            #Check for early stopping
            if ( not model_freeze ) and early_stopping(model, epoch, val_acc):
                break
        scheduler.step(val_loss)

    if RANK == 0:

        # Save complete final model
        final_model = {"model": model,
                    "criterion": loss_fn,
                    "epochs": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    "model_state": model.state_dict(),
                    "val_loss_min": val_loss}

        MODEL_FULL = f"{MODEL_SAVE_DIR}/{MODEL_NAME}_final_model.pth"

        torch.save(final_model, MODEL_FULL)
        print(f"[{device} on {HOST_NAME}] Full model saved at {MODEL_FULL}.")

        metrics_data["job_completion_time"] = time.time() - START_TIME
        # Save metrics to JSON (model is not included in the metrics_data)
        print(f"[{device} on {HOST_NAME}] Saving metrics ...")        
        with open(RESULTS_FILE, "w") as f:
            json.dump(metrics_data, f, indent = 4)
        print(f"[{device} on {HOST_NAME}] Metrics saved to {RESULTS_FILE}.")

        print(f"[{device} on {HOST_NAME}] Saving the metric plots to {PLOT_FILE}")
        # Initialize the plotter with the metrics file
        try:
            plotter = Plotter(RESULTS_FILE)
            plotter.plot_metrics(PLOT_FILE)
        except:
            print(f"[{device} on {HOST_NAME}] Plotting of metrics to {PLOT_FILE} failed !")

    print(f"[{device} on {HOST_NAME}] Destroying process group.")
    
    # Destroy process group
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    main_worker(args)
