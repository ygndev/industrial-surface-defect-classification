
import torch
from torch.utils.data import DataLoader

from dataset import SurfaceDefectDataset, get_transforms


def main():
    for split in ("train", "val", "test"):
        dataset = SurfaceDefectDataset(
            split=split,
            transform=get_transforms(train=(split == "train")),
        )

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=(split == "train"),
            num_workers=0,
        )

        images, labels = next(iter(loader))

        print(f"\n{split.upper()}")
        print(f"Dataset size: {len(dataset)}")
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Pixel range: {images.min():.3f} to {images.max():.3f}")


if __name__ == "__main__":
    main()
