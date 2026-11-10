
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import CLASSES, get_transforms
from model import DefectCNN


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"
RESULTS_DIR = ROOT / "results"

SAMPLES = ("PS_69.bmp", "PS_217.bmp", "Pa_255.bmp")


def compute_gradcam(model, image, class_index):
    activations = None
    gradients = None

    # Capture the last convolutional feature maps.
    target_layer = model.features[8]

    def save_activations(module, inputs, output):
        nonlocal activations
        activations = output

    def save_gradients(gradient):
        nonlocal gradients
        gradients = gradient

    handle = target_layer.register_forward_hook(save_activations)

    try:
        model.zero_grad(set_to_none=True)
        scores = model(image)

        if activations is None:
            raise RuntimeError("Feature maps were not captured")

        activations.register_hook(save_gradients)
        scores[0, class_index].backward()

        if gradients is None:
            raise RuntimeError("Gradients were not captured")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * activations).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        heatmap = F.interpolate(
            heatmap,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        heatmap = heatmap[0, 0].detach().cpu().numpy()
        maximum = heatmap.max()

        if maximum > 0:
            heatmap = heatmap / maximum

        return heatmap
    finally:
        handle.remove()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = DefectCNN().to(device)
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = get_transforms(train=False)

    fig, axes = plt.subplots(
        len(SAMPLES), 3, figsize=(12, 10),
        constrained_layout=True,
    )

    for row, filename in enumerate(SAMPLES):
        matches = list(DATA_DIR.rglob(filename))

        if len(matches) != 1:
            raise ValueError(
                f"Expected one match for {filename}, found {len(matches)}"
            )

        path = matches[0]

        with Image.open(path) as image:
            gray = np.asarray(image.convert("L")).copy()
            tensor = transform(image.convert("L")).unsqueeze(0)
            tensor = tensor.to(device)

        with torch.no_grad():
            scores = model(tensor)
            predicted = scores.argmax(dim=1).item()
            confidence = torch.softmax(scores, dim=1)[0, predicted].item()

        true_label = path.stem.split("_")[0]
        true_index = CLASSES.index(true_label)

        predicted_cam = compute_gradcam(model, tensor, predicted)
        true_cam = compute_gradcam(model, tensor, true_index)

        panels = axes[row]

        panels[0].imshow(gray, cmap="gray")
        panels[0].set_title(f"{filename} | True: {true_label}")

        panels[1].imshow(gray, cmap="gray")
        panels[1].imshow(
            predicted_cam, cmap="jet", alpha=0.45,
            vmin=0, vmax=1,
        )
        panels[1].set_title(
            f"Predicted: {CLASSES[predicted]} ({confidence:.1%})"
        )

        panels[2].imshow(gray, cmap="gray")
        panels[2].imshow(
            true_cam, cmap="jet", alpha=0.45,
            vmin=0, vmax=1,
        )
        panels[2].set_title(f"True-class CAM: {true_label}")

        for ax in panels:
            ax.axis("off")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "gradcam.png"

    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.show()


if __name__ == "__main__":
    main()
