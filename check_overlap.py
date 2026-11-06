
import numpy as np
import rasterio
from pathlib import Path

folder = Path("dataset")

sar_path = next(folder.rglob("*_S1Hand.tif"))
optical_path = next(folder.rglob("*_S2Hand.tif"))
label_path = next(folder.rglob("*_LabelHand.tif"))

with rasterio.open(sar_path) as src:
    sar = src.read().astype(np.float32)

with rasterio.open(optical_path) as src:
    optical = src.read().astype(np.float32)

with rasterio.open(label_path) as src:
    label = src.read(1)

# A pixel is considered valid only when all bands
# are finite and at least one band has nonzero data.
sar_valid = np.isfinite(sar).all(axis=0) & np.any(sar != 0, axis=0)
optical_valid = (
    np.isfinite(optical).all(axis=0)
    & np.any(optical != 0, axis=0)
)

both_valid = sar_valid & optical_valid

print("Total pixels:", label.size)
print("SAR valid pixels:", int(sar_valid.sum()))
print("Optical valid pixels:", int(optical_valid.sum()))
print("Both valid pixels:", int(both_valid.sum()))

print("\nLabel values and counts:")
values, counts = np.unique(label, return_counts=True)

for value, count in zip(values, counts):
    print(value, ":", int(count))

print("\nLabel counts in overlapping valid area:")
values, counts = np.unique(label[both_valid], return_counts=True)

for value, count in zip(values, counts):
    print(value, ":", int(count))
