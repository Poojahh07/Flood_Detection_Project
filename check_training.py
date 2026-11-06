
import torch
import torch.nn as nn
from data_utils import get_dataloader
from model import FloodTransformer
from config import DEVICE

loader = get_dataloader(shuffle=False)
sar, optical, labels = next(iter(loader))

sar = sar.to(DEVICE)
optical = optical.to(DEVICE)
labels = labels.to(DEVICE)

model = FloodTransformer().to(DEVICE)

# Give the flood class more importance
weights = torch.tensor([1.0, 2.0], device=DEVICE)
criterion = nn.CrossEntropyLoss(
    weight=weights,
    ignore_index=255
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

print("Training diagnostic started...")

for epoch in range(1, 21):
    model.train()
    optimizer.zero_grad()

    output = model(sar, optical)
    loss = criterion(output, labels)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(sar, optical)
        prediction = output.argmax(dim=1)

        flood_pixels = (prediction == 1).sum().item()
        total_loss = loss.item()

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {total_loss:.4f} | "
        f"Predicted flood pixels: {flood_pixels}"
    )
    
    model.eval()

    with torch.no_grad():
        output = model(sar, optical)
        prediction = output.argmax(dim=1)

        valid = labels != 255
        correct = ((prediction == labels) & valid).sum().item()
        total = valid.sum().item()

        actual_flood = ((labels == 1) & valid).sum().item()
        predicted_flood = ((prediction == 1) & valid).sum().item()
        true_flood = ((prediction == 1) & (labels == 1)).sum().item()

        accuracy = correct / max(total, 1) * 100
        precision = true_flood / max(predicted_flood, 1) * 100
        recall = true_flood / max(actual_flood, 1) * 100

        print("\n===== FLOOD DETECTION RESULTS =====")
        print(f"Valid pixels: {total}")
        print(f"Actual flood pixels: {actual_flood}")
        print(f"Predicted flood pixels: {predicted_flood}")
        print(f"Correct flood predictions: {true_flood}")
        print(f"Pixel accuracy: {accuracy:.2f}%")
        print(f"Flood precision: {precision:.2f}%")
        print(f"Flood recall: {recall:.2f}%")


print("\nDiagnostic training completed.")
