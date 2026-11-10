
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

classes = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted Surface",
    "RS": "Rolled-in Scale",
    "Sc": "Scratches",
}

images = sorted(DATA_DIR.rglob("*.bmp"))

fig, axes = plt.subplots(
    len(classes), 3, figsize=(9, 12)
)

for row, (code, name) in enumerate(classes.items()):
    samples = [
        path for path in images
        if path.stem.split("_")[0] == code
        or path.parent.name == code
    ][:3]

    if len(samples) < 3:
        raise ValueError(f"Not enough images for {code}")

    for col, path in enumerate(samples):
        with Image.open(path) as image:
            axes[row, col].imshow(image, cmap="gray")

        axes[row, col].set_title(f"{name} ({code})")
        axes[row, col].axis("off")

fig.tight_layout()
plt.show()
