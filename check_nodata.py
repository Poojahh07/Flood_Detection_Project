
import numpy as np
import rasterio
from pathlib import Path

folder = Path("dataset")

for path in folder.rglob("*.tif"):
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
        print(f"\nFile: {path.name}")
        print("Shape:", image.shape)
        print("NoData metadata:", src.nodata)
        print("NaN values:", int(np.isnan(image).sum()))
        print("Infinite values:", int(np.isinf(image).sum()))
        print("Zero values:", int((image == 0).sum()))
        print("Total values:", image.size)

        if "S2Hand" in path.name:
            zero_per_pixel = np.all(image == 0, axis=0)
            print(
                "Pixels with all 13 bands zero:",
                int(zero_per_pixel.sum())
            )
