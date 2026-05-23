import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class CSRNet(nn.Module):
    """CSRNet: VGG-16 front-end (10 conv layers, split into 4 named stages)
    + dilated conv back-end (no BN).
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg16(weights=weights)
        features = list(vgg.features.children())

        # Split VGG front-end into named stages for staged unfreezing
        # conv1: indices 0-4  (conv1_1→relu→conv1_2→relu→pool1)  stride=2,  64ch
        # conv2: indices 5-9  (conv2_1→relu→conv2_2→relu→pool2)  stride=4, 128ch
        # conv3: indices 10-16 (conv3_1→relu→conv3_2→relu→conv3_3→relu→pool3) stride=8, 256ch
        # conv4: indices 17-22 (conv4_1→relu→conv4_2→relu→conv4_3→relu) stride=8, 512ch
        self.vgg_conv1 = nn.Sequential(*features[:5])
        self.vgg_conv2 = nn.Sequential(*features[5:10])
        self.vgg_conv3 = nn.Sequential(*features[10:17])
        self.vgg_conv4 = nn.Sequential(*features[17:23])

        # Back-end: 6 dilated conv layers (no BN, light spatial dropout)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 128, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, 0, 0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.vgg_conv1(x)
        x = self.vgg_conv2(x)
        x = self.vgg_conv3(x)
        x = self.vgg_conv4(x)      # (B, 512, H/8, W/8)
        x = self.backend(x)        # (B,  64, H/8, W/8)
        x = self.output_layer(x)   # (B,   1, H/8, W/8)
        return x

    def vgg_stages(self):
        """Return VGG stages in order: conv1, conv2, conv3, conv4."""
        return [self.vgg_conv1, self.vgg_conv2, self.vgg_conv3, self.vgg_conv4]
