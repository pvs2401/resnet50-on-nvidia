import subprocess
import torch
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

def get_device():
    match True:
        case _ if torch.cuda.is_available():
            return torch.device("cuda")
        case _ if torch.backends.mps.is_available():
            return torch.device("mps")
        case _:
            return torch.device("cpu")          

class LogFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
    def open(self, mode='w'):
        if self.file is None:
            self.file = open(self.file_path, mode)
        else:
            raise RuntimeError("Log file is already open.")
    def write(self, message):
        if self.file is not None:
            self.file.write(message + '\n')
            self.file.flush()  # Ensure immediate writing to disk
        else:
            raise RuntimeError("Log file is not open. Please call `open` first.")
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
        else:
            raise RuntimeError("Log file is not open.")
    def __enter__(self):
        """
        Context manager entry method.
        """
        self.open()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit method. Ensures the file is closed.
        """
        self.close()

# Plotter

class Plotter:
    def __init__(self, metrics_file):
        self.metrics_file = metrics_file

    def _load_metrics(self):
        """
        Load metrics from the JSON file.
        """
        if not os.path.exists(self.metrics_file):
            raise FileNotFoundError(f"Metrics file '{self.metrics_file}' not found.")
        
        with open(self.metrics_file, "r") as f:
            metrics = json.load(f)
        return metrics

    def plot_metrics(self, save_path="metrics_plot.png"):
        """
        Plot training and validation loss, accuracy, and epoch completion time from metrics.
        Save the plot to a specified file.
        
        Args:
        - save_path (str): The file path to save the plot as a .png file.
        """
        metrics = self._load_metrics()

        # Extract data directly from the JSON structure
        epochs = metrics["epochs"]
        train_losses = metrics["train_loss"]
        val_losses = metrics["val_loss"]
        train_accuracies = [acc * 100 for acc in metrics["train_accuracy"]]  # Convert to percentages
        val_accuracies = [acc * 100 for acc in metrics["val_accuracy"]]      # Convert to percentages
        epoch_completion_time = metrics["epoch_completion_time"]
        best_epoch = metrics["best_epoch"]
        best_accuracy = metrics["best_accuracy"] * 100  # Convert to percentage
        # Get system info for the title
        node_model = metrics["system_model"]
        gpus_per_node = metrics["gpus_per_node"]
        node_count = metrics["node_count"]
        batch_size = metrics["batch_size"]
        job_completion_time = metrics["job_completion_time"]
        job_id = metrics["job_id"]
        trainable_params = metrics["trainable_params"]
        # Create a figure with 2 rows and 2 columns, but leave the last subplot (2,2) empty
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # Adjusted for 2x2 grid layout

        # Add a main heading to the figure
        fig.suptitle(
            f"Model: {node_model} | GPUs per Node: {gpus_per_node} | JOB ID: {job_id} | Node Count: {node_count} | Batch Size: {batch_size} | Total params trained: {trainable_params}\n\nJCT: {job_completion_time:.2f}s",
            fontsize=18, fontweight='bold', y=1.03
        )

        # --- Graph 1: Training and Validation Loss ---
        axs[0, 0].plot(epochs, train_losses, label="Train Loss", color='tab:blue', linewidth=2)
        axs[0, 0].plot(epochs, val_losses, label="Validation Loss", color='tab:cyan', linewidth=2)
        axs[0, 0].fill_between(epochs, train_losses, color='tab:blue', alpha=0.1)
        axs[0, 0].fill_between(epochs, val_losses, color='tab:cyan', alpha=0.1)
        axs[0, 0].plot(epochs, train_losses, 'x', color='tab:blue', markersize=5)
        axs[0, 0].plot(epochs, val_losses, 'x', color='tab:cyan', markersize=5)
        axs[0, 0].set_xlabel("Epoch", fontsize=14)
        axs[0, 0].set_ylabel("Loss", fontsize=14)
        axs[0, 0].set_title("Training and Validation Loss", fontsize=16)
        axs[0, 0].legend(loc="upper right", fontsize=12)
        axs[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Graph 2: Training and Validation Accuracy ---
        axs[0, 1].plot(epochs, train_accuracies, label="Train Accuracy (%)", color='tab:green', linewidth=2)
        axs[0, 1].plot(epochs, val_accuracies, label="Validation Accuracy (%)", color='tab:cyan', linewidth=2)
        axs[0, 1].fill_between(epochs, train_accuracies, color='tab:green', alpha=0.1)
        axs[0, 1].fill_between(epochs, val_accuracies, color='tab:cyan', alpha=0.1)
        axs[0, 1].plot(epochs, train_accuracies, 'x', color='tab:green', markersize=5)
        axs[0, 1].plot(epochs, val_accuracies, 'x', color='tab:cyan', markersize=5)
        axs[0, 1].annotate(
            f"Best Epoch {best_epoch}\nVal Acc: {best_accuracy:.2f}%",
            xy=(best_epoch, best_accuracy),
            xytext=(best_epoch - 2, best_accuracy - 15),
            fontsize=12,
            color='darkred',
            ha='center',
            arrowprops=dict(facecolor='darkred', arrowstyle='->', lw=2, shrinkA=0, shrinkB=5)
        )
        axs[0, 1].set_xlabel("Epoch", fontsize=14)
        axs[0, 1].set_ylabel("Accuracy (%)", fontsize=14)
        axs[0, 1].set_title("Training and Validation Accuracy", fontsize=16)
        axs[0, 1].legend(loc="upper left", fontsize=12)
        axs[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Graph 3: Epoch Completion Time ---
        axs[1, 0].plot(epochs, epoch_completion_time, label="ECT", color='tab:red', linewidth=2, linestyle='--')
        axs[1, 0].fill_between(epochs, epoch_completion_time, color='tab:red', alpha=0.2)
        axs[1, 0].plot(epochs, epoch_completion_time, 'x', color='tab:red', markersize=5)
        axs[1, 0].set_xlabel("Epoch", fontsize=14)
        axs[1, 0].set_ylabel("Time (s)", fontsize=14)
        axs[1, 0].set_title("Epoch Completion Time", fontsize=16)
        axs[1, 0].legend(loc="lower left", fontsize=12)
        axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Remove the empty subplot in the 2nd row and 2nd column
        fig.delaxes(axs[1, 1])

        # Add the timestamp as a footer
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.5, 0.01, f"Generated on: {timestamp}", ha='center', fontsize=10, color='gray')

        # Layout adjustments to prevent clipping
        plt.tight_layout()

        # Save the plot as a .png file
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")

def TrackTime(start_time):
    """Function to calculate the time taken for an epoch."""
    end_time = time.time()
    epoch_duration = end_time - start_time
    return epoch_duration