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


class Down(nn.Module):
    """
    Downscaling block with MaxPool2D -> ConvBlock.
    """
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    Upscaling block with ConvTranspose2D and optional residual concatenation.
    """
    def __init__(self, in_ch, out_ch, residual=False):
        super(Up, self).__init__()
        self.residual = residual
        conv_in_ch = out_ch * 2 if residual else in_ch
        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch if residual else in_ch,
            kernel_size=2,
            stride=2
        )
        self.conv = ConvBlock(conv_in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            # Match dimensions of x1 to x2 via padding
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class OutConv(nn.Module):
    """
    Output convolutional layer.
    """
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)