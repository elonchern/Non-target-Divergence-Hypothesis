#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
DDRNet
jieli_cn@163.com
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/home/elon/SSC-master')
from models.projection_layer import Project2Dto3D
from models.DDR import DDR_ASPP3d
from models.DDR import BottleneckDDR2d, BottleneckDDR3d, DownsampleBlock3d


# DDRNet
# ----------------------------------------------------------------------
class SSC_RGBD_DDRNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SSC_RGBD_DDRNet, self).__init__()
        print('SSC_RGBD_DDRNet: RGB and Depth streams with DDR blocks for Semantic Scene Completion')

        w, h, d = 240, 144, 240
        # --- depth
        c_in, c, c_out, dilation, residual = 1, 4, 8, 1, True
        self.dep_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_dep = Project2Dto3D(w, h, d)  # w=240, h=144, d=240
        self.dep_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=4, c_out=16, dilation=1, residual=True),
            DownsampleBlock3d(16, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )

        # --- RGB
        c_in, c, c_out, dilation, residual = 3, 4, 8, 1, True
        self.rgb_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_rgb = Project2Dto3D(w, h, d)  # w=240, h=144, d=240

        self.rgb_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=4, c_out=16, dilation=1, residual=True),
            DownsampleBlock3d(16, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )

        # -------------1/4

        # ck = 256
        # self.ds = DownsamplerBlock_3d(64, ck)
        ck = 64
        c = 16
        # c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True
        self.res3d_1d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.res3d_1r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=16, c_out=64)
        # self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=64, c_out=int(ck * 4))

        # 64 * 5 = 320
        self.conv_out = nn.Sequential(
            nn.Conv3d(320, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, 1, 1, 0)
        )

        # ----  weights init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)  # gain=1
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)

    def forward(self, x_depth=None, x_rgb=None, p=None):
        # input: x (BS, 3L, 240L, 144L, 240L)
        # print('SSC: x.shape', x.shape)

        x0_rgb = self.rgb_feature2d(x_rgb)
        x0_rgb = self.project_layer_rgb(x0_rgb, p)
        x0_rgb = self.rgb_feature3d(x0_rgb)

        x0_depth = self.dep_feature2d(x_depth)
        x0_depth = self.project_layer_dep(x0_depth, p)
        x0_depth = self.dep_feature3d(x0_depth)

        f0 = torch.add(x0_depth, x0_rgb)

        x_4_d = self.res3d_1d(x0_depth)
        x_4_r = self.res3d_1r(x0_rgb)

        f1 = torch.add(x_4_d, x_4_r)

        x_5_d = self.res3d_2d(x_4_d)
        x_5_r = self.res3d_2r(x_4_r)

        f2 = torch.add(x_5_d, x_5_r)

        x_6_d = self.res3d_3d(x_5_d)
        x_6_r = self.res3d_3r(x_5_r)

        f3 = torch.add(x_6_d, x_6_r)

        x = torch.cat((f0, f1, f2, f3), dim=1)  # channels concatenate
        # x = torch.cat((x0_rgb, x_4_r, x_5_r, x_6_r), dim=1)
        # x = torch.cat((x0_depth, x_4_d, x_5_d, x_6_d), dim=1)
        
        # print('SSC: channels concatenate x', x.size())  # (BS, 256L, 60L, 36L, 60L)

        x = self.aspp(x)

        y = self.conv_out(x)  # (BS, 12L, 60L, 36L, 60L)

        return y


class SSC_RGB_DDRNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SSC_RGB_DDRNet, self).__init__()
        print('SSC_RGB_DDRNet: RGB streams with DDR blocks for Semantic Scene Completion')

        w, h, d = 240, 144, 240

        # --- RGB
        c_in, c, c_out, dilation, residual = 3, 4, 8, 1, True
        self.rgb_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_rgb = Project2Dto3D(w, h, d)  # w=240, h=144, d=240

        self.rgb_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=4, c_out=16, dilation=1, residual=True),
            DownsampleBlock3d(16, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )

        # -------------1/4

        # ck = 256
        # self.ds = DownsamplerBlock_3d(64, ck)
        ck = 64
        c = 16
        # c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True
        self.res3d_1d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.res3d_1r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=16, c_out=64)
        # self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=64, c_out=int(ck * 4))

        # 64 * 5 = 320
        self.conv_out = nn.Sequential(
            nn.Conv3d(320, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, 1, 1, 0)
        )

        # ----  weights init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)  # gain=1
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)

    def forward(self, x_rgb=None, p=None):
        # input: x (BS, 3L, 240L, 144L, 240L)
        # print('SSC: x.shape', x.shape)

        x0_rgb = self.rgb_feature2d(x_rgb)
        x0_rgb = self.project_layer_rgb(x0_rgb, p)
        x0_rgb = self.rgb_feature3d(x0_rgb)

        f0 = x0_rgb

        x_4_r = self.res3d_1r(x0_rgb)

        f1 = x_4_r

        x_5_r = self.res3d_2r(x_4_r)

        f2 = x_5_r

        x_6_r = self.res3d_3r(x_5_r)

        f3 = x_6_r

        x = torch.cat((f0, f1, f2, f3), dim=1)  # channels concatenate
        # print('SSC: channels concatenate x', x.size())  # (BS, 256L, 60L, 36L, 60L)

        x = self.aspp(x)

        y = self.conv_out(x)  # (BS, 12L, 60L, 36L, 60L)

        return y
    
    
class SSC_Depth_DDRNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SSC_Depth_DDRNet, self).__init__()
        print('SSC_Depth_DDRNet: Depth streams with DDR blocks for Semantic Scene Completion')

        w, h, d = 240, 144, 240
        # --- depth
        c_in, c, c_out, dilation, residual = 1, 4, 8, 1, True
        self.dep_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_dep = Project2Dto3D(w, h, d)  # w=240, h=144, d=240
        self.dep_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=4, c_out=16, dilation=1, residual=True),
            DownsampleBlock3d(16, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )


        # -------------1/4

        # ck = 256
        # self.ds = DownsamplerBlock_3d(64, ck)
        ck = 64
        c = 16
        # c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True
        self.res3d_1d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.res3d_1r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=16, c_out=64)
        # self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=64, c_out=int(ck * 4))

        # 64 * 5 = 320
        self.conv_out = nn.Sequential(
            nn.Conv3d(320, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, 1, 1, 0)
        )

        # ----  weights init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)  # gain=1
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)

    def forward(self, x_depth=None, p=None):
        # input: x (BS, 3L, 240L, 144L, 240L)
        # print('SSC: x.shape', x.shape)


        x0_depth = self.dep_feature2d(x_depth)
        x0_depth = self.project_layer_dep(x0_depth, p)
        x0_depth = self.dep_feature3d(x0_depth)

        f0 = x0_depth

        x_4_d = self.res3d_1d(x0_depth)

        f1 = x_4_d

        x_5_d = self.res3d_2d(x_4_d)

        f2 = x_5_d

        x_6_d = self.res3d_3d(x_5_d)

        f3 = x_6_d
        
        x = torch.cat((f0, f1, f2, f3), dim=1)  # channels concatenate
        # print('SSC: channels concatenate x', x.size())  # (BS, 256L, 60L, 36L, 60L)

        x = self.aspp(x)

        y = self.conv_out(x)  # (BS, 12L, 60L, 36L, 60L)

        return y    

if __name__ == '__main__':
    model = SSC_RGBD_DDRNet()
    x = torch.rand(1, 1, 370,1220)
    y = torch.rand(1, 3, 480,640)
    out = model(x,y)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # 计算参数量
    imagenet_params = count_parameters(model)


    print(f"ImageNet model parameters: {imagenet_params}")