
import torch
from torch.utils.data import DataLoader
from torch import nn

from config import DEVICE, CHECKPOINT_DIR
from data_utils import FloodDataset
from model import FloodTransformer


def main():
    checkpoint_path = (
        f"{CHECKPOINT_DIR}/flood_transformer.pth"
    )

    dataset = FloodDataset()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model = FloodTransformer().to(DEVICE)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    total_loss = 0.0
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for sar, optical, labels in loader:
            sar = sar.to(DEVICE)
            optical = optical.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(sar, optical)
            loss = criterion(logits, labels)

            valid = labels != 255
            if valid.sum().item() == 0:
                continue

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)
            correct += (
                (predictions == labels) & valid
            ).sum().item()
            total += valid.sum().item()

            true_positive += (
                (predictions == 1)
                & (labels == 1)
                & valid
            ).sum().item()

            false_positive += (
                (predictions == 1)
                & (labels == 0)
                & valid
            ).sum().item()

            false_negative += (
                (predictions == 0)
                & (labels == 1)
                & valid
            ).sum().item()

    if total == 0:
        print("No valid labelled pixels found.")
        return

    accuracy = 100.0 * correct / total

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        100.0 * true_positive / precision_denominator
        if precision_denominator else 0.0
    )

    recall = (
        100.0 * true_positive / recall_denominator
        if recall_denominator else 0.0
    )

    iou_denominator = (
        true_positive + false_positive + false_negative
    )

    iou = (
        100.0 * true_positive / iou_denominator
        if iou_denominator else 0.0
    )

    print("\n===== FLOOD DETECTION EVALUATION =====")
    print("Valid labelled pixels:", total)
    print("Evaluation loss:", round(total_loss / len(loader), 4))
    print("Pixel accuracy:", round(accuracy, 2), "%")
    print("Flood precision:", round(precision, 2), "%")
    print("Flood recall:", round(recall, 2), "%")
    print("Flood IoU:", round(iou, 2), "%")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
