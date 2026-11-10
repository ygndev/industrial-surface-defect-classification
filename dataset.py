
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent

CLASSES = ("Cr", "In", "Pa", "PS", "RS", "Sc")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


class SurfaceDefectDataset(Dataset):
    def __init__(self, split, transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        with (ROOT / "data" / "splits.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))

        self.samples = [
            (ROOT / row["path"], CLASS_TO_IDX[row["label"]])
            for row in rows
            if row["split"] == split
        ]

        if not self.samples:
            raise ValueError(f"No samples found for split: {split}")

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("L")

            if self.transform is not None:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

        return image, label


def get_transforms(train=False):
    operations = []

    if train:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    return transforms.Compose(operations)
