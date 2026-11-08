import torch
import torch.nn as nn
import torch.nn.functional as F

class TriggerEncoder(nn.Module):
    def __init__(self, image_channels=3, num_classes=10):
        super(TriggerEncoder, self).__init__()
        self.num_classes = num_classes
        
        # The input will be the image concatenated with the spatially expanded label
        in_channels = image_channels + num_classes
        
        # Simple CNN to generate the residual noise
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output channels must match the original image channels
            nn.Conv2d(32, image_channels, kernel_size=3, padding=1),
            # Tanh keeps the residual noise bounded (e.g., between -0.1 and 0.1)
            nn.Tanh() 
        )
        
        # A scaling factor to ensure the noise isn't too large initially
        self.noise_scale = 0.5

    def forward(self, image, target_labels):
        batch_size, _, H, W = image.size()
        
        # 1. Convert label to one-hot: [B] -> [B, num_classes]
        one_hot = F.one_hot(target_labels, num_classes=self.num_classes).float()
        
        # 2. Expand spatially: [B, num_classes] -> [B, num_classes, H, W]
        label_map = one_hot.view(batch_size, self.num_classes, 1, 1).expand(-1, -1, H, W)
        
        # 3. Concatenate image and label map along the channel dimension
        x = torch.cat([image, label_map], dim=1)
        
        # 4. Generate the residual noise
        noise = self.net(x) * self.noise_scale
        
        # 5. Add noise to original image and clamp to valid pixel range [0, 1]
        poisoned_image = torch.clamp(image + noise, -1.0, 1.0)
        
        return poisoned_image