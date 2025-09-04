### Some functions
import torch  
import numpy as np
import torch.nn as nn

class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        
        s_N, s_C = s_shape
        t_N, t_C = t_shape
        assert s_C == t_C

        self.bn = nn.BatchNorm1d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)