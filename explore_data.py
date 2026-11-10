
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

image_paths = sorted(DATA_DIR.rglob("*.bmp"))

if not image_paths:
    raise FileNotFoundError(f"No BMP images found in {DATA_DIR}")

class_names = {"Cr", "In", "Pa", "PS", "RS", "Sc"}
class_counts = Counter()
image_sizes = Counter()
image_modes = Counter()
invalid_images = []

for path in image_paths:
    label = path.stem.split("_")[0]

    if label not in class_names:
        label = path.parent.name

    try:
        with Image.open(path) as image:
            image.load()
            image_sizes[image.size] += 1
            image_modes[image.mode] += 1
    except (OSError, ValueError):
        invalid_images.append(path)
        continue

    class_counts[label] += 1

print(f"Total files: {len(image_paths)}")
print("\nClass distribution:")

for label, count in sorted(class_counts.items()):
    print(f"{label}: {count}")

print("\nImage sizes:", dict(image_sizes))
print("Image modes:", dict(image_modes))
print("Invalid images:", len(invalid_images))

fig, ax = plt.subplots(figsize=(8, 4))
labels = sorted(class_counts)

ax.bar(labels, [class_counts[label] for label in labels])
ax.set_title("NEU-CLS Class Distribution")
ax.set_xlabel("Defect class")
ax.set_ylabel("Image count")

fig.tight_layout()
plt.show()
