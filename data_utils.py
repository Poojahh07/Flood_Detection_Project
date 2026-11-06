
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from config import DATASET_DIR, IMAGE_SIZE, BATCH_SIZE


def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    normalized = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[0]):
        band = image[channel]
        valid = np.isfinite(band) & (band != 0)

        if not np.any(valid):
            continue

        low, high = np.percentile(band[valid], [2, 98])

        if high > low:
            normalized[channel] = np.clip(
                (band - low) / (high - low),
                0.0,
                1.0
            )

        normalized[channel][~valid] = 0.0

    return normalized.astype(np.float32)


class FloodDataset(Dataset):
    def __init__(self):
        self.samples = []
        dataset_path = Path(DATASET_DIR)

        for sar_path in dataset_path.rglob("*_S1Hand.tif"):
            optical_path = sar_path.with_name(
                sar_path.name.replace(
                    "_S1Hand.tif",
                    "_S2Hand.tif"
                )
            )

            label_path = sar_path.with_name(
                sar_path.name.replace(
                    "_S1Hand.tif",
                    "_LabelHand.tif"
                )
            )

            if optical_path.exists() and label_path.exists():
                self.samples.append(
                    (str(sar_path), str(optical_path), str(label_path))
                )

        if not self.samples:
            raise FileNotFoundError(
                f"No matching SAR, optical and label files found in "
                f"{dataset_path}"
            )

        print("Matched dataset scenes:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sar_path, optical_path, label_path = self.samples[index]

        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)
            sar_valid = src.dataset_mask() > 0

        with rasterio.open(optical_path) as src:
            optical = src.read().astype(np.float32)
            optical_valid = src.dataset_mask() > 0

        with rasterio.open(label_path) as src:
            label = src.read(1)

        # Require finite SAR data and at least one nonzero band.
        sar_valid = (
            sar_valid
            & np.isfinite(sar).all(axis=0)
            & np.any(sar != 0, axis=0)
        )

        # Optical pixels with all bands zero are treated as invalid.
        optical_valid = (
            optical_valid
            & np.isfinite(optical).all(axis=0)
            & np.any(optical != 0, axis=0)
        )

        valid = sar_valid & optical_valid

        sar = normalize_image(sar)
        optical = normalize_image(optical)

        # Invalid labels become 255 (ignored by the loss).
        label = np.where(
            np.isin(label, [0, 1]),
            label,
            255
        ).astype(np.uint8)

        # Exclude pixels where either input image is invalid.
        label[~valid] = 255

        sar_tensor = torch.from_numpy(sar).unsqueeze(0)
        optical_tensor = torch.from_numpy(optical).unsqueeze(0)
        label_tensor = torch.from_numpy(
            label.astype(np.float32)
        )[None, None]

        size = (IMAGE_SIZE, IMAGE_SIZE)

        sar_tensor = F.interpolate(
            sar_tensor,
            size=size,
            mode="bilinear",
            align_corners=False
        )[0]

        optical_tensor = F.interpolate(
            optical_tensor,
            size=size,
            mode="bilinear",
            align_corners=False
        )[0]

        label_tensor = F.interpolate(
            label_tensor,
            size=size,
            mode="nearest"
        )[0, 0].to(torch.uint8)

        return sar_tensor, optical_tensor, label_tensor.long()


def get_dataloader(shuffle=True):
    dataset = FloodDataset()

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )
