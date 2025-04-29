import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Basic block with Conv2D -> ReLU.
    """
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
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