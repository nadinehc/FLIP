import torch.nn as nn
import torch

class TriggerDecoder(nn.Module):
    def __init__(self, image_channels=3, num_classes=10):
        super(TriggerDecoder, self).__init__()
        # This is essentially a standard image classifier
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, poisoned_image):
        logits = self.net(poisoned_image)
        return logits