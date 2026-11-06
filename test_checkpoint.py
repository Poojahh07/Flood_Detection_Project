import torch
from config import DEVICE, CHECKPOINT_DIR
from model import FloodTransformer
import os

path = os.path.join(CHECKPOINT_DIR, "flood_transformer.pth")

model = FloodTransformer().to(DEVICE)
checkpoint = torch.load(path, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

print("Checkpoint loaded:", path)
print("Model device:", DEVICE)
print("Checkpoint keys:", len(checkpoint))
print("Model is in evaluation mode:", not model.training)
print("Checkpoint test successful!")
