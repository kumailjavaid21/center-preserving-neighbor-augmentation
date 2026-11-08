import torch
from torch import nn

class JiangMCDnCNN3(nn.Module):
    """Original implementation of the declared three-plane adaptation.

    Ten 3x3 convolutions: 3->64, eight 64->64 with BN, then 64->1.
    The network predicts central noise; restoration has the subtractive sign.
    """
    method = 'JIANG'

    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, padding=1, bias=True), nn.ReLU()]
        for _ in range(8):
            layers.extend((nn.Conv2d(64, 64, 3, padding=1, bias=False),
                           nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True),
                           nn.ReLU()))
        layers.append(nn.Conv2d(64, 1, 3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)
        self._G_microbatch = 4
        self._warmup_microbatch = 1
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, y_center, sigma_unused, y_minus, y_plus):
        assert y_center.shape == y_minus.shape == y_plus.shape
        return y_center - self.net(torch.cat((y_minus, y_center, y_plus), dim=1))
