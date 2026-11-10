
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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
        "test", transform=get_transforms(train=False)
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

    print(f"Device: {device}")
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASSES))),
        average="macro",
        zero_division=0,
    )

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASSES))),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *CLASSES])

        for name, row in zip(CLASSES, matrix):
            writer.writerow([name, *row])

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASSES,
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("NEU-CLS Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
