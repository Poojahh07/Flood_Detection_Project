
from pathlib import Path

folder = Path("dataset")

extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

images = [
    file for file in folder.rglob("*")
    if file.is_file() and file.suffix.lower() in extensions
]

print("Total image files:", len(images))

for image in images:
    print(image.name)
