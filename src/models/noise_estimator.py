import torch
from torch import nn
from torch.nn import functional as F

class NoiseEstimator(nn.Module):
    """Independent feature stem; no running statistics or stochastic layers."""
    def __init__(self, feature_mode='plain'):
        super().__init__()
        if feature_mode not in ('plain', 'contrast'):
            raise ValueError('Unknown independent noise feature mode')
        self.feature_mode = feature_mode
        self.stem = nn.Sequential(nn.Conv2d(3 if feature_mode == 'contrast' else 1, 12, 5, padding=2), nn.SiLU(),
                                  nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.SiLU())
        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, y):
        features = y
        if self.feature_mode == 'contrast':
            # Fixed image-only contrast statistics expose noise amplitude without
            # introducing oracle sigma or any feature shared with the generator.
            hp = y-F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
            variance = F.avg_pool2d(F.pad(hp.square(), (2, 2, 2, 2), mode='reflect'), 5, stride=1)
            features = torch.cat((y, torch.tanh(hp.abs()/0.10), torch.tanh(variance/0.01)), 1)
        coarse = 0.15 * torch.sigmoid(self.head(F.adaptive_avg_pool2d(self.stem(features), (4, 4))))
        return F.interpolate(coarse, y.shape[-2:], mode='bilinear', align_corners=False)

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self
