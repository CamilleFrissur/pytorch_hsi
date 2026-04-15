import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import rasterio

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    
    
def get_tensor_from_pixel_coord(hsi_path, coords, bands):
    
    with rasterio.open(hsi_path) as src:
        iterator = src.sample(coords, bands)
        ds = []
        
        for pixel in iterator:
            ds.append(pixel)
    
    ds = torch.Tensor(ds).unsqueeze(1)
    ds = ds / 10_000
    
    return ds
    
if __name__ == "__main__":
    
    PATH = 'modelTrained_3_Nepoch_15.pth'
    # HSI_PATH = 'hsi_roi/NOU_FL09_20160921_184602_PS01_IMG011_atm_slice_004_x2.hdr'
    HSI_PATH = 'hsi_roi/NOU_FL09_20160921_184602_PS01_IMG011_atm_slice_004_x2.hyspex'
    
    with rasterio.open(HSI_PATH) as src:
        ds = src.read()
    
    ds = torch.Tensor(ds)
    print(ds.shape)
    
    
    rgb_bands = [5,79,139]
    x_start = int(0.3 * ds.shape[1])
    x_stop = int(0.8 * ds.shape[1])
    y_start = int(0.2 * ds.shape[2])
    y_stop = int(0.8 * ds.shape[2])
    # slice_x = [50:250]
    
    # slice_y = [30:100]
    rgb = ds[rgb_bands,x_start:x_stop,y_start:y_stop]
    rgb = rgb.swapaxes(0,2)
    rgb = rgb.swapaxes(0,1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb[:,:,0] = (rgb[:,:,0] - rgb[:,:,0].min()) / (rgb[:,:,0].max() - rgb[:,:,0].min())
    rgb[:,:,1] = (rgb[:,:,1] - rgb[:,:,1].min()) / (rgb[:,:,1].max() - rgb[:,:,1].min())
    rgb[:,:,2] = (rgb[:,:,2] - rgb[:,:,2].min()) / (rgb[:,:,2].max() - rgb[:,:,2].min())
    print(rgb.min(), rgb.max()) 
    ds = ds[:343,x_start:x_stop,y_start:y_stop]

    chans, x, y = ds.shape
    
    ds = ds.transpose(0,2)
    ds = ds.transpose(0,1)
    # print(ds.shape)
    ds = ds.flatten(0,1)
    ds = ds.unsqueeze(1)
    # print(ds.shape)

    
    model = CNN_1D(view_size=343)
    
    chkpt = torch.load(PATH, map_location=torch.device('cpu'))

    layers_to_keep = []
    for k, v in chkpt.items():
        if k.startswith("encoder."):
            layers_to_keep.append((k[len("encoder."):], v))
    layers_to_keep = layers_to_keep[:-10]
    d = OrderedDict(layers_to_keep)
            
    # model.load_state_dict(d, strict=False)
    model.load_state_dict(d)
    model.eval()
    
    # BoundingBox(left=313322.075, bottom=449405.77, right=313672.075, top=451679.77)
    emb = ds
    emb = emb.squeeze()
    # with torch.no_grad():
    #     emb = model(ds)
    # print(emb.shape)
    
    pca = PCA(n_components=3)
    pca_emb = pca.fit_transform(emb.numpy())
    
    # print(pca_emb.shape)
    # print(pca_emb.max())
    
    pca_emb = pca_emb.reshape(x,y,3)
    # pca_emb = (pca_emb - pca_emb.min()) / (pca_emb.max() - pca_emb.min())
    pca_emb[:,0] = (pca_emb[:,0] - pca_emb[:,0].min()) / (pca_emb[:,0].max() - pca_emb[:,0].min())
    pca_emb[:,1] = (pca_emb[:,1] - pca_emb[:,1].min()) / (pca_emb[:,1].max() - pca_emb[:,1].min())
    pca_emb[:,2] = (pca_emb[:,2] - pca_emb[:,2].min()) / (pca_emb[:,2].max() - pca_emb[:,2].min())
    
    fig, ax = plt.subplots(1,2, figsize=(5,10))
    ax[0].imshow(pca_emb)
    ax[1].imshow(rgb)

    plt.tight_layout()
    plt.show()


    
    # # x = torch.rand(100,1,343)
    # with torch.no_grad():
    #     print(model(torch.ones(1000,1,343)).mean())
    #     print(model(torch.zeros(1000,1,343)).mean())