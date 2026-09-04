
"""
U-Net Architecture for Satellite Flood Detection
Supports both single-channel (SAR) and multi-channel (SAR + Optical) inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for Flood Segmentation

    Args:
        n_channels: Number of input channels (1 for SAR, 2 for dual-pol SAR, 13 for Sentinel-2)
        n_classes: Number of output classes (1 for binary flood, 3 for multi-class)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetPlusPlus(nn.Module):
    """
    U-Net++ with Nested Skip Connections
    Better for capturing fine-grained flood boundaries
    """
    def __init__(self, n_channels=1, n_classes=1):
        super(UNetPlusPlus, self).__init__()
        filters = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv0_0 = self._block(n_channels, filters[0])
        self.conv1_0 = self._block(filters[0], filters[1])
        self.conv2_0 = self._block(filters[1], filters[2])
        self.conv3_0 = self._block(filters[2], filters[3])
        self.conv4_0 = self._block(filters[3], filters[4])

        # Nested skip connections
        self.conv0_1 = self._block(filters[0] + filters[1], filters[0])
        self.conv1_1 = self._block(filters[1] + filters[2], filters[1])
        self.conv2_1 = self._block(filters[2] + filters[3], filters[2])
        self.conv3_1 = self._block(filters[3] + filters[4], filters[3])

        self.conv0_2 = self._block(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = self._block(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = self._block(filters[2]*2 + filters[3], filters[2])

        self.conv0_3 = self._block(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = self._block(filters[1]*3 + filters[2], filters[1])

        self.conv0_4 = self._block(filters[0]*4 + filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        return self.final(x0_4)


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test U-Net with single-channel SAR input
    model = UNet(n_channels=1, n_classes=1).to(device)
    x = torch.randn(2, 1, 256, 256).to(device)
    out = model(x)
    print(f"U-Net Input: {x.shape} -> Output: {out.shape}")

    # Test U-Net++ with dual-pol SAR (VV + VH)
    model_pp = UNetPlusPlus(n_channels=2, n_classes=1).to(device)
    x2 = torch.randn(2, 2, 256, 256).to(device)
    out2 = model_pp(x2)
    print(f"U-Net++ Input: {x2.shape} -> Output: {out2.shape}")
