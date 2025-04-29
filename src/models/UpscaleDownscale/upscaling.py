import torch
import torch.nn as nn
from .modules import ConvBlock, OutConv, UpStride, UpSample

class UpscalingDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch):
        super(UpscalingDecoder, self).__init__()
        
        # 16 x 16 x 4
        self.conv1 = ConvBlock(in_ch, base_ch * 2)              # 16 x 16 x 2base_ch
        self.up1 = UpSample(24)                                 # 24 x 24 x 2base_ch
        self.conv4 = ConvBlock(base_ch * 2, base_ch * 4)        # 24 x 24 x 4base_ch       
        self.up2 = UpStride(base_ch * 4, base_ch * 4)           # 48 x 48 x 4base_ch
        self.conv5 = ConvBlock(base_ch * 4, base_ch * 4)        # 48 x 48 x 4base_ch
        self.up3 = UpStride(base_ch * 4, base_ch * 4)           # 96 x 96 x 4base_ch
        self.outc = OutConv(base_ch * 4, out_ch)                # 96 x 96 x out_ch
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_sequence=False):
        x1 = self.conv1(x)
        x2 = self.up1(x1)
        x3 = self.conv4(x2)
        x4 = self.up2(x3)
        x5 = self.conv5(x4)
        x6 = self.up3(x5)
        x7 = self.sigmoid(self.outc(x6))

        if return_sequence:
            return [x1, x2, x3, x4, x5, x6, x7]
        return x7
    