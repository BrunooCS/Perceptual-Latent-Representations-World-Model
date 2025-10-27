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
from .modules import ConvBlock, OutConv, DownStride, DownPool



class DownscalingEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch):
        super(DownscalingEncoder, self).__init__()
        
        # 96 x 96 x 3
        self.conv1 = ConvBlock(in_ch, base_ch)              # 96 x 96 x base_ch        
        self.down1 = DownStride(base_ch, base_ch)           # 48 x 48 x base_ch
        self.conv2 = ConvBlock(base_ch, base_ch * 2)        # 48 x 48 x 2base_ch
        self.down2 = DownStride(base_ch*2, base_ch*2)       # 24 x 24 x 2base_ch
        self.conv3 = ConvBlock(base_ch * 2, base_ch * 4)    # 24 x 24 x 4base_ch
        self.down3 = DownPool(16)                           # 16 x 16 x 4base_ch
        self.conv4 = OutConv(base_ch * 4, out_ch)           # 16 x 16 x out_ch
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_sequence=False):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.conv2(x2)
        x4 = self.down2(x3)
        x5 = self.conv3(x4)
        x6 = self.down3(x5)
        x7 = self.sigmoid(self.conv4(x6))

        if return_sequence:
            return [x1, x2, x3, x4, x5, x6, x7]
        return x7