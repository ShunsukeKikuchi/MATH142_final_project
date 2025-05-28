import torch
import torch.nn as nn
from typing import List

# ──────────────────────────────────────────────────
# 1-D Bottleneck block (expansion = 4)
# ──────────────────────────────────────────────────
class Bottleneck1d(nn.Module):
    expansion = 4

    def __init__(self,
                 in_planes,
                 planes,
                 stride = 1,
                 downsample=None):
        super().__init__()

        # 1 × 1
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)

        # 3 × 1
        self.conv2 = nn.Conv1d(planes, planes,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)

        # 1 × 1
        self.conv3 = nn.Conv1d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ──────────────────────────────────────────────────
# Generic ResNet-1D
# ──────────────────────────────────────────────────
class ResNet1d(nn.Module):
    def __init__(self,
                 block: type[Bottleneck1d],
                 layers: List[int],
                 in_channels: int = 1,
                 num_classes: int = 1000,
                 width_mult: float = 1.2):          # tuned for ~25.5 M params
        super().__init__()

        # widen channels
        c1 = int(64  * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)

        self.in_planes = c1

        # stem
        self.conv1  = nn.Conv1d(in_channels, c1, kernel_size=7,
                                stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm1d(c1)
        self.relu   = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # stages
        self.layer1 = self._make_layer(block, c1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, c2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, c3, layers[2], stride=2)
        self.layer4 = self._make_layer(block, c4, layers[3], stride=2)

        # head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(c4 * block.expansion, num_classes)

        # Kaiming normal init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    # ----------------------------------------------
    def _make_layer(self,
                    block: type[Bottleneck1d],
                    planes: int,
                    blocks: int,
                    stride: int):
        downsample = None
        out_planes = planes * block.expansion
        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_planes),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    # ----------------------------------------------
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).squeeze(-1)   # (B, C, 1) → (B, C)
        return self.fc(x)

def resnet50_1d(num_classes: int = 1000,
                in_channels: int = 1,
                width_mult: float = 1.2) -> ResNet1d:
    """ResNet-50 depth (3-4-6-3 bottleneck blocks) for 1-D inputs."""
    return ResNet1d(Bottleneck1d, [3, 4, 6, 3],
                    in_channels=in_channels,
                    num_classes=num_classes,
                    width_mult=width_mult)