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

from torch import nn
from .modules import ConvBlock, Down, Up, OutConv

class UNet(nn.Module):
    """
    U-Net implementation with configurable base channels and residual connections.
    """
    def __init__(self, in_ch, out_channels, base_ch):
        super(UNet, self).__init__()

        self.residual = True
        self.inc = ConvBlock(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8, self.residual)
        self.up2 = Up(base_ch * 8, base_ch * 4, self.residual)
        self.up3 = Up(base_ch * 4, base_ch * 2, self.residual)
        self.up4 = Up(base_ch * 2, base_ch, self.residual)

        self.outc = OutConv(base_ch, out_channels)

    def forward(self, x, return_sequence=False):
        """
        Forward pass for U-Net. Optionally returns intermediate feature maps.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outc(x9)

        if return_sequence:
            return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
        return x10

    def perceptual_loss(self, x_dds, x_vae, max_index=9):
        """
        Compute perceptual loss between two inputs based on feature maps.
        """
        xs_vae = self.forward(x_vae, return_sequence=True)
        xs_dds = self.forward(x_dds, return_sequence=True)

        loss = 0
        for vae, dds in zip(xs_vae[:max_index], xs_dds[:max_index]):
            loss += (vae - dds).square().mean()
        return loss / max_index