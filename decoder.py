import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                int(in_channels), 
                int(out_channels), 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                padding_mode='reflect', 
                bias=True
            ),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)
     
def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")

class UpconvLayer(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpconvLayer, self).__init__()
        
        total_in_channels = in_channels + skip_channels
        
        self.conv_block1 = ConvBlock(total_in_channels, out_channels)
        self.conv_block2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x_up = upsample(x)
        x_concat = torch.cat([x_up, skip], dim=1)
        
        out = self.conv_block1(x_concat)
        out = self.conv_block2(out)
        
        return out
    

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

class DisparityHead(nn.Module):
    def __init__(self, in_channels, min_depth=0.01, max_depth=0.2):
        super(DisparityHead, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        self.conv = ConvBlock(in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        disp_raw = self.sigmoid(self.conv(x))
        scaled_disp, depth = disp_to_depth(disp_raw, self.min_depth, self.max_depth)
        
        # scaled_disp for Smoothness Loss and depth for Photometric Loss
        return scaled_disp, depth