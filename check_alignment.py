
import rasterio
from data_utils import FloodDataset

dataset = FloodDataset()
sar_path, optical_path, label_path = dataset.samples[0]

for name, path in [
    ("SAR", sar_path),
    ("OPTICAL", optical_path),
    ("LABEL", label_path),
]:
    with rasterio.open(path) as src:
        print(f"\n===== {name} =====")
        print("CRS:", src.crs)
        print("Size:", src.width, "x", src.height)
        print("Bands:", src.count)
        print("Bounds:", src.bounds)

print("\nAlignment information collected.")
