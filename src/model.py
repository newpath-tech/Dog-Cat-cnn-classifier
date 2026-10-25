import torch
import torch.nn as nn


class ThreeLayerCNN(nn.Module):
    """
    Convolutional Neural Network for Dog vs Cat Binary Classification
    using exactly three convolutional blocks followed by a fully connected classifier.
    
    Architecture:
      - Block 1: Conv2d(3 -> 32, kernel=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2x2)
      - Block 2: Conv2d(32 -> 64, kernel=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2x2)
      - Block 3: Conv2d(64 -> 128, kernel=3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2x2)
      - Pooling: AdaptiveAvgPool2d((4, 4))
      - Classifier: Flatten -> Linear(2048 -> 256) -> ReLU -> Dropout(0.5) -> Linear(256 -> num_classes)
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(ThreeLayerCNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive average pooling produces a fixed 4x4 spatial grid
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully Connected Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        logits = self.fc(x)
        return logits


def get_model_summary(model: nn.Module, input_size=(1, 3, 128, 128)):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_input = torch.randn(*input_size)
    output = model(dummy_input)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "output_shape": list(output.shape)
    }


if __name__ == "__main__":
    net = ThreeLayerCNN(num_classes=2)
    summary = get_model_summary(net)
    print("Model initialized successfully.")
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Output Logits Shape: {summary['output_shape']}")
