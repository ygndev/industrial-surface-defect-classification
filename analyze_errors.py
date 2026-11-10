
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import CLASSES, SurfaceDefectDataset, get_transforms
from model import DefectCNN


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"
RESULTS_DIR = ROOT / "results"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = SurfaceDefectDataset(
        "test",
        transform=get_transforms(train=False),
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = DefectCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    errors = []

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)

            confidence, predictions = probabilities.max(dim=1)

            predictions = predictions.cpu().tolist()
            confidence = confidence.cpu().tolist()
            labels = labels.tolist()

            for index, (actual, predicted, score) in enumerate(
                zip(labels, predictions, confidence)
            ):
                if actual == predicted:
                    continue

                sample_index = batch_index * loader.batch_size + index
                path, _ = dataset.samples[sample_index]

                errors.append({
                    "path": path,
                    "actual": CLASSES[actual],
                    "predicted": CLASSES[predicted],
                    "confidence": score,
                })

    errors.sort(
        key=lambda error: error["confidence"],
        reverse=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "misclassified.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "path", "actual", "predicted", "confidence"
        ])

        for error in errors:
            writer.writerow([
                error["path"].relative_to(ROOT).as_posix(),
                error["actual"],
                error["predicted"],
                f"{error['confidence']:.6f}",
            ])

    print(f"Misclassified images: {len(errors)}")

    for error in errors:
        print(
            f"{error['path'].name}: "
            f"{error['actual']} -> {error['predicted']} "
            f"({error['confidence']:.1%})"
        )

    if not errors:
        return

    columns = 3
    rows = math.ceil(len(errors) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, rows * 3.8),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, error in zip(axes.flat, errors):
        with Image.open(error["path"]) as image:
            ax.imshow(image, cmap="gray")

        ax.set_title(
            f"True: {error['actual']} | "
            f"Pred: {error['predicted']}\n"
            f"Confidence: {error['confidence']:.1%}\n"
            f"{error['path'].name}",
            fontsize=10,
            pad=12,
        )

    fig.tight_layout(h_pad=3.0, w_pad=2.0)

    output = RESULTS_DIR / "misclassified.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")

    print(f"\nSaved: {output}")
    plt.show()


if __name__ == "__main__":
    main()

