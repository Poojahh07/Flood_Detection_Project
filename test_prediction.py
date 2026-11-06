import torch
from config import DEVICE, CHECKPOINT_DIR
from model import FloodTransformer
from data_utils import get_dataloader
import os

# Load trained model
model = FloodTransformer().to(DEVICE)

path = os.path.join(CHECKPOINT_DIR, "flood_transformer.pth")
model.load_state_dict(torch.load(path, map_location=DEVICE))
model.eval()

# Load the training scene
loader = get_dataloader(shuffle=False)

with torch.no_grad():
    for sar, optical, labels in loader:
        sar = sar.to(DEVICE)
        optical = optical.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(sar, optical)
        predictions = outputs.argmax(dim=1)

        valid = labels != 255

        print("Output shape:", tuple(outputs.shape))
        print("SAR shape:", tuple(sar.shape))
        print("Optical shape:", tuple(optical.shape))
        print("Valid target pixels:", valid.sum().item())

        if valid.any():
            print("Target Class 0:", ((labels == 0) & valid).sum().item())
            print("Target Class 1:", ((labels == 1) & valid).sum().item())
            print("Predicted Class 0:", ((predictions == 0) & valid).sum().item())
            print("Predicted Class 1:", ((predictions == 1) & valid).sum().item())

        break
