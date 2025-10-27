# Copyright (c) 2024-2025, Bruno Corcuera
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Find the full license text in the LICENSE file at the root of this repository.

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