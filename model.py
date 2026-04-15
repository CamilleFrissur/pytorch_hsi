import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ResidualBlock(nn.Module):
    """
    A residual block for 1D CNN with proper padding and resampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2  # Calculate padding to maintain dimensions
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # If the input and output channels are different, use a 1x1 convolution to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Resample the shortcut to match the output dimensions
        shortcut = self.shortcut(x)
        if out.size(-1) != shortcut.size(-1):
            shortcut = F.interpolate(shortcut, size=out.size(-1), mode='linear', align_corners=False)

        out += shortcut
        out = F.relu(out)
        return out

class CNN_1D(nn.Module):
    """
    CNN model implementation with residual blocks.
    """
    def __init__(self, view_size, size_dim=[512, 1024]):
        super(CNN_1D, self).__init__()
        self.view_size = view_size
        self.size_dim = size_dim

        # Define initial convolutional layer
        self.conv0 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=32, stride=2)
        self.bn0 = nn.BatchNorm1d(8)

        # Define residual blocks
        self.res_block1 = ResidualBlock(8, 16, kernel_size=5)
        self.res_block2 = ResidualBlock(16, 32, kernel_size=10)
        self.res_block3 = ResidualBlock(32, 64, kernel_size=4,stride=3)

    def forward(self, x):
        """
        Get the encoding of the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.max_pool1d(self.res_block1(x), 2)
        x = F.max_pool1d(self.res_block2(x), 2)
        x = F.max_pool1d(self.res_block3(x), 2)
        x = x.view(x.size(0), -1)
        return x
    
    
if __name__ == "__main__":
    
    PATH = 'modelTrained_3_Nepoch_15.pth'
    model = CNN_1D(view_size=343)
    
    print(model)
    chkpt = torch.load(PATH, map_location=torch.device('cpu'))
    
    for k,v  in chkpt.items():
        if "running_mean" in k:
            print(k)
            print(v.shape)
    
    # sys.exit(1)
    
    print(chkpt.keys())
    layers_to_keep = []
    for k, v in chkpt.items():
        if k.startswith("encoder."):
            layers_to_keep.append((k[len("encoder."):], v))
    layers_to_keep = layers_to_keep[:-10]
    d = OrderedDict(layers_to_keep)
    print(d.keys())
            
    # model.load_state_dict(d, strict=False)
    model.load_state_dict(d)
    model.eval()
    print(model)
    
    
    x = torch.rand(1,1,343)
    
    print(model(x).shape)