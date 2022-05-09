import numpy as np
import torch
import torch.nn as nn

class GraspFormer(nn.Module):
    def __init__(self, verbose=False):
        super(GraspFormer, self).__init__()
        self.verbose = verbose
        pass

    def forward(self, x):
        pass