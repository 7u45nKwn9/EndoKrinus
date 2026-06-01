import torch
import torch.nn as nn

from encoder import Encoder 
from decoder import UpconvLayer, DisparityHead, ConvBlock, upsample

class DepthNetwork(nn.Module):
    def __init__(self, min_depth=0.01, max_depth=0.2):
        super(DepthNetwork, self).__init__()
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Encoder
        # [f_fused1, f_fused2, f_fused3, f_fused4]
        self.encoder = Encoder() 
        
        # Decoder
        # Upconv Layer
        self.upconv3 = UpconvLayer(in_channels=512, skip_channels=256, out_channels=256)
        self.upconv2 = UpconvLayer(in_channels=256, skip_channels=128, out_channels=128)
        self.upconv1 = UpconvLayer(in_channels=128, skip_channels=64, out_channels=64)
        self.upconv0 = ConvBlock(in_channels=64, out_channels=16)
        
        # Disparity Heads đa độ phân giải (Multi-scale)
        self.disp_head3 = DisparityHead(in_channels=256, min_depth=self.min_depth, max_depth=self.max_depth)
        self.disp_head2 = DisparityHead(in_channels=128, min_depth=self.min_depth, max_depth=self.max_depth)
        self.disp_head1 = DisparityHead(in_channels=64,  min_depth=self.min_depth, max_depth=self.max_depth)
        self.disp_head0 = DisparityHead(in_channels=16,  min_depth=self.min_depth, max_depth=self.max_depth)

    def forward(self, x):
        # Encoder
        f1, f2, f3, f4 = self.encoder(x)
        outputs = {} # for Multi-scale Loss
        
        # Decoder
        d3 = self.upconv3(f4, f3) 
        scaled_disp3, depth3 = self.disp_head3(d3)
        outputs[("disp", 3)] = scaled_disp3
        outputs[("depth", 3)] = depth3
        
        d2 = self.upconv2(d3, f2)
        scaled_disp2, depth2 = self.disp_head2(d2)
        outputs[("disp", 2)] = scaled_disp2
        outputs[("depth", 2)] = depth2
        
        d1 = self.upconv1(d2, f1)
        scaled_disp1, depth1 = self.disp_head1(d1)
        outputs[("disp", 1)] = scaled_disp1
        outputs[("depth", 1)] = depth1
        
        d0_up = upsample(d1) 
        d0 = self.upconv0(d0_up)
        scaled_disp0, depth0 = self.disp_head0(d0)
        outputs[("disp", 0)] = scaled_disp0
        outputs[("depth", 0)] = depth0
        
        return outputs
