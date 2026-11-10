
import torch
from torch import nn


class DefectCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


if __name__ == "__main__":
    model = DefectCNN()

    example = torch.randn(16, 1, 200, 200)
    output = model(example)

    parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("Output shape:", output.shape)
    print("Total parameters:", parameters)
