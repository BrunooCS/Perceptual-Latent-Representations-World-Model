import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2D -> BatchNorm -> SiLU activation.
    """
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class OutConv(nn.Module):
    """
    Output convolutional layer.
    """
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DownStride(nn.Module):
    """
    Downscaling block with conv2d with stride.
    """
    def __init__(self, in_ch, out_ch):
        super(DownStride, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down(x)
    
class DownPool(nn.Module):
    """
    Downscaling block with average pooling.
    """
    def __init__(self, output_size):
        super(DownPool, self).__init__()
        self.down = nn.Sequential(
            nn.AdaptiveAvgPool2d((output_size,output_size)),
        )

    def forward(self, x):
        return self.down(x)


class UpStride(nn.Module):
    """
    Upscaling block with convtranspose2d with stride.
    """
    def __init__(self, in_ch, out_ch):
        super(UpStride, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)
    
class UpSample(nn.Module):
    """
    Upscaling block with bilinear upsampling.
    """
    def __init__(self, output_size):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=(output_size,output_size), mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.up(x)