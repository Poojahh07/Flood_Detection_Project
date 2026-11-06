
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    CHECKPOINT_DIR,
)
from data_utils import FloodDataset
from model import FloodTransformer


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    valid_pixels = 0
    correct_pixels = 0
    true_flood = 0
    predicted_flood = 0
    actual_flood = 0

    criterion = nn.CrossEntropyLoss(
        ignore_index=255,
        reduction="sum"
    )

    with torch.no_grad():
        for sar, optical, labels in loader:
            sar = sar.to(device)
            optical = optical.to(device)
            labels = labels.to(device)

            logits = model(sar, optical)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            valid = labels != 255

            total_loss += loss.item()
            valid_pixels += valid.sum().item()
            correct_pixels += (
                (predictions == labels) & valid
            ).sum().item()

            true_flood += (
                (predictions == 1) & (labels == 1) & valid
            ).sum().item()

            predicted_flood += (
                (predictions == 1) & valid
            ).sum().item()

            actual_flood += (
                (labels == 1) & valid
            ).sum().item()

    accuracy = 100 * correct_pixels / max(valid_pixels, 1)
    precision = 100 * true_flood / max(predicted_flood, 1)
    recall = 100 * true_flood / max(actual_flood, 1)

    return {
        "loss": total_loss / max(valid_pixels, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = FloodDataset()

    if len(dataset) < 2:
        print(
            "\nWARNING: Only one scene is available. "
            "Metrics are training-scene metrics, not "
            "independent validation results."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = FloodTransformer().to(DEVICE)

    class_weights = torch.tensor(
        [1.0, 2.0],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=255,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    best_loss = float("inf")
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "flood_transformer.pth",
    )

    epochs = max(1, EPOCHS)

    print("\nStarting flood detection training...")
    print("Device:", DEVICE)
    print("Scenes:", len(dataset))
    print("Epochs:", epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        valid_pixels = 0

        for sar, optical, labels in loader:
            sar = sar.to(DEVICE)
            optical = optical.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(sar, optical)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            valid_count = (labels != 255).sum().item()
            training_loss += loss.item() * valid_count
            valid_pixels += valid_count

        train_loss = training_loss / max(valid_pixels, 1)
        metrics = evaluate(model, loader, DEVICE)

        print(
            f"\nEpoch {epoch}/{epochs}"
            f"\nTrain loss: {train_loss:.4f}"
            f"\nEvaluation loss: {metrics['loss']:.4f}"
            f"\nPixel accuracy: {metrics['accuracy']:.2f}%"
            f"\nFlood precision: {metrics['precision']:.2f}%"
            f"\nFlood recall: {metrics['recall']:.2f}%"
        )

        # Save the model with the lowest evaluation loss.
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                },
                checkpoint_path,
            )

            print("Best checkpoint saved:", checkpoint_path)

    print("\nTraining completed!")
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
