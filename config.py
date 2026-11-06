
import os
import torch

# =====================================
# FLOOD DETECTION PROJECT CONFIGURATION
# =====================================

# Project information
PROJECT_NAME = "Attention-Guided Multimodal Transformer for Flood Detection"
RANDOM_SEED = 42

# Image settings
IMAGE_SIZE = 128
SAR_CHANNELS = 2
OPTICAL_CHANNELS = 13
NUM_CLASSES = 2

# Dataset settings
IGNORE_LABEL = -1

SAR_SUFFIX = "_S1Hand.tif"
OPTICAL_SUFFIX = "_S2Hand.tif"
LABEL_SUFFIX = "_LabelHand.tif"

# Training settings
BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

# Validation settings
VALIDATION_SPLIT = 0.2

# Device settings
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Project folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR, "dataset"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs"
)

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR, "checkpoints"
)

RESULTS_DIR = os.path.join(
    OUTPUT_DIR, "results"
)

# Create output folders automatically
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Display configuration
if __name__ == "__main__":
    print("=" * 45)
    print("FLOOD DETECTION CONFIGURATION")
    print("=" * 45)

    print("Project:", PROJECT_NAME)
    print("Image size:", IMAGE_SIZE)
    print("SAR channels:", SAR_CHANNELS)
    print("Optical channels:", OPTICAL_CHANNELS)
    print("Number of classes:", NUM_CLASSES)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Device:", DEVICE)
    print("Dataset folder:", DATASET_DIR)
    print("Output folder:", OUTPUT_DIR)
    print("Checkpoint folder:", CHECKPOINT_DIR)
    print("Results folder:", RESULTS_DIR)

    print("=" * 45)
    print("Configuration loaded successfully!")
