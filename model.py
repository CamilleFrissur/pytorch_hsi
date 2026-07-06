import sys
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import rasterio
from rasterio.mask import mask

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import pandas as pd
import geopandas as gpd

from shapely.geometry import box

class ResidualBlock(nn.Module): # using nn.Module is Pytorch standard
    """
    A residual block for 1D CNN with proper padding and resampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1): #self calls the class itself
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2  # Calculate padding to maintain dimensions
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels) # normalization at the batch level
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding) #Two convolutions based on Colin's model
        self.bn2 = nn.BatchNorm1d(out_channels)

        # If the input and output channels are different, use a 1x1 convolution to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        """
        Args:
            self: the class itself
            x (torch.Tensor): 3 dimensions (batch, dimension, channels) input tensor.
        """
        out = F.relu(self.bn1(self.conv1(x))) # REctified Liear Unit Function gives more flexibility to non-linear models
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
        """
        Args:
            view_size: number of bands (343 for HSI data)
        """
        super(CNN_1D, self).__init__()
        self.view_size = view_size
        self.size_dim = size_dim

        # Define initial convolutional layer
        self.conv0 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=32, stride=2)
        self.bn0 = nn.BatchNorm1d(8)

        # Define residual blocks
        #Residual blocks help avoiding vanishing gradients using raw data
        self.res_block1 = ResidualBlock(8, 16, kernel_size=5) # Values based on Colin's model
        self.res_block2 = ResidualBlock(16, 32, kernel_size=10)
        self.res_block3 = ResidualBlock(32, 64, kernel_size=4,stride=3)

    def forward(self, x):
        """
        Get the encoding of the input.

        Args:
            self: the class itself
            x (torch.Tensor): 3 dimensions (batch, dimension, channels) input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.max_pool1d(self.res_block1(x), 2) #maxpooling reduces spatial dimension while keeping images info
        x = F.max_pool1d(self.res_block2(x), 2)
        x = F.max_pool1d(self.res_block3(x), 2)
        x = x.view(x.size(0), -1)
        return x
    
    
def get_tensor_from_geodataframe(hsi_path, gdf):
    
    # working with rasterio.open reduce the cost when working with voluminous data
   
    with rasterio.open(hsi_path) as src: #src for source
        # ds = src.read() #ds for dataset
        gdf = gdf.to_crs(src.crs)
        out, _ = mask(src, gdf.geometry, invert=False, crop=True)
        out = out[:343,:,:] # remove once Colin application of bands have been done
    
    ds = torch.Tensor(out).unsqueeze(1) # unsqueeze create an empty dimension to fit the needs
    ds = ds / 10_000 # needed to convert the HSI values to right format 
    print (f"ds shape is:{ds.shape}")
    ds = ds.transpose(0,2)
    print (f"ds shape is:{ds.shape}")
    ds = ds.transpose(0,1)
    print (f"ds shape is:{ds.shape}")
    ds = ds.flatten(0,1)
    print (f"ds shape is:{ds.shape}")
    ds = ds.unsqueeze (1) #add an empty dimension
    print (f"ds shape is:{ds.shape}")
    print(ds.dtype)
    
    return ds
    
def get_tensor (ds):
    """_summary_

    Args:
        ds (int16): hsi_path opened with rasterio

    Returns:
        torch.float32: Tensor
    """
    ds = torch.Tensor(ds)
    ds = ds/10_000
    return ds

def tensor_format (ds):
    """
    Modify ds arguments to fit the order batch, dimension, channels

    Args:
        ds (torch.float3): Tensor

    Returns:
        torch.float32: Tensor formatted to be used in the model
    """
    ds = ds.transpose(0,2)
    ds = ds.transpose(0,1)
    ds = ds.flatten(0,1)
    ds = ds.unsqueeze (1) #add an empty dimension
    print(ds.dtype)
    return ds

def encoder_only(weights):
    """
    Removing everything related to the projector (as the model is not self-supervised) and keeping the encoding

    Args:
        weights (_type_): Weights of the model

    Returns:
        dictionnary: _description_
    """
    layers_to_keep = []
    for k, v in weights.items():
        if k.startswith("encoder."):
            layers_to_keep.append((k[len("encoder."):], v))
    layers_to_keep = layers_to_keep[:-10]
    d = OrderedDict(layers_to_keep) #d for dictionnary
    return d
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use a gpu if available
    print(f"Using {device} device")
    
    PATH = 'modelTrained_3_Nepoch_15.pth' # Colin's model
    #HSI_PATH = 'hsi_roi/NOU_FL09_20160921_184602_PS01_IMG011_atm_slice_004_x2.hyspex' #
    HSI_PATH = "./hsi_roi/NOU_FL12_20160921_182748_PS01_IMG008_atm_slice_002_x2.hyspex"
    HSI_PATH_LIANA ="./hsi_roi/NOU_FL12_20160921_182748_PS01_IMG008_atm_slice_002_x2.hyspex" #zone where we know there are liana
    SHP_PATH = "liana_shapefile/LianaOrNotLiana.shp"
    
    ## OPTION 1: use path and dataframe to have a general (not working yet)
    # with rasterio.open(HSI_PATH) as src:
    #     xmin = src.bounds.left
    #     xmax = src.bounds.right
    #     ymin = src.bounds.bottom
    #     ymax = src.bounds.top
    #     hyspex_crs = src.crs

    # # Équivalent de vos x_start, x_stop, y_start, y_stop
    # x_start = xmin + 0.3 * (xmax - xmin)
    # x_stop  = xmin + 0.8 * (xmax - xmin)
    # y_start = ymin + 0.2 * (ymax - ymin)
    # y_stop  = ymin + 0.8 * (ymax - ymin)
    
    # slice_x = [50:250]
    # slice_y = [30:100]
    
    # #Creating a geodataframe based on this coordinates
    # gdf_map = gpd.GeoDataFrame(geometry=[box(x_start, y_start, x_stop, y_stop)], crs=hyspex_crs)
    
    # # Opening liana shapefile and creating the mask
    # gdf_liana  = gpd.read_file(SHP_PATH, where="Type = 1") # Type = 1 for polygons of liana, Type= 0 for polygons without liana
    
    # out = get_tensor_from_geodataframe(HSI_PATH_LIANA, gdf_liana)
    # print (f"out shape is:{out.shape}")
    # ds = get_tensor_from_geodataframe (HSI_PATH, gdf_map)
    # print (f"ds shape is:{ds.shape}")
    
    ## OPTION 2
    sim_i = []
    for i in range (2):
        gdf  = gpd.read_file(SHP_PATH, where=f"Type = {i}")
        with rasterio.open(HSI_PATH_LIANA) as src: #src for source
            # ds = src.read() #ds for dataset
            gdf = gdf.to_crs(src.crs)
            out, _ = mask(src, gdf.geometry, invert=False, crop=True)
            out = out[:343,:,:]
        
        with rasterio.open(HSI_PATH) as src: #src for source
            ds = src.read() #ds for dataset
            
        # # Creating the tensors
        out = get_tensor(out)
        out = tensor_format(out)
        print(out.shape)
        
        ds= get_tensor(ds)
        print(ds.shape)
        
        # Part of the map to plot
        x_start = int(0.3 * ds.shape[1])
        x_stop = int(0.8 * ds.shape[1])
        y_start = int(0.2 * ds.shape[2])
        y_stop = int(0.8 * ds.shape[2])
        
        rgb_bands = [5,79,139]
        
        rgb = ds[rgb_bands,:,:]
        rgb = rgb.swapaxes(0,2)
        rgb = rgb.swapaxes(0,1)
        
        # Normalization of RGB bands
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        
        # Normalization of RGB bands one by one
        rgb[:,:,0] = (rgb[:,:,0] - rgb[:,:,0].min()) / (rgb[:,:,0].max() - rgb[:,:,0].min())
        rgb[:,:,1] = (rgb[:,:,1] - rgb[:,:,1].min()) / (rgb[:,:,1].max() - rgb[:,:,1].min())
        rgb[:,:,2] = (rgb[:,:,2] - rgb[:,:,2].min()) / (rgb[:,:,2].max() - rgb[:,:,2].min())
        
        rgb = rgb[x_start:x_stop, y_start:y_stop, :] 
        
        # print(rgb.min(), rgb.max()) 
        ds = ds[:343,x_start:x_stop,y_start:y_stop]

        #y, _, chans, x = ds.shape
        chans, x, y = ds.shape
        
        
        ds = tensor_format(ds)
        print(ds.shape)
        
        model = CNN_1D(view_size=343) #view_size = number of HSI bands
        
        chkpt = torch.load(PATH, map_location=torch.device('cpu')) #weights are loaded

        d = encoder_only(chkpt) #keeping only weights related to encoding
                
        # model.load_state_dict(d, strict=False)
        model.load_state_dict(d) #model applied for encoding only
        model = model.to(device) # model on gpu if available
        model.eval() #deactivate normalization
        
        # BoundingBox(left=313322.075, bottom=449405.77, right=313672.075, top=451679.77)
        # emb = ds
        # emb = emb.squeeze()
        
        with torch.no_grad():
            #emb = model(ds)
            emb = model(ds.to(device)) #ds is the tensor
            template_emb = model(out.to(device))
        
        # emb = (emb - emb.mean(dim=0)) / emb.std(dim=0)
        emb = (emb - emb.mean()) / emb.std()
        print ("EMbedding's type:", emb.type)
        
        template = template_emb.mean(dim=0)
        
        
        # # Applying PCA
        # pca = PCA(n_components=3) # 3 for RGB
        # #pca_emb = pca.fit_transform(emb.numpy())
        # pca_emb = pca.fit_transform(emb.cpu().numpy()) #numpy runs on cou only
        
        # print(pca_emb.shape)
        # print(pca_emb.max())
        
        # pca_emb = pca_emb.reshape(x,y,3)
        # pca_emb = (pca_emb - pca_emb.min()) / (pca_emb.max() - pca_emb.min())

        
        # Applying cosine similarity
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # sim = cos(template, emb)
        # sim = sim.reshape(x,y)
        # # sim = torch.where(sim > 0.5, sim, sim.mean())
        # sim = torch.where(sim > 0, sim, sim.mean())
        
        
        # Cosine similarity without model (nm=no model)
        ds_nm = ds.squeeze(1)   # (nb_pixels, bands)
        out_sq = out.squeeze(1)                # (nb_pixels, 343)
        valid_mask = (out_sq != 0).any(dim=1)
        valid_pixels = out_sq[valid_mask]  #only the pixels from the shapefile polygons are kept
        template_nm = valid_pixels.mean(dim=0) # mean on pixels
        template_nm = template_nm.unsqueeze(0).expand(ds_nm.shape[0], -1)  # (59640, 343)
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cos(template_nm, ds_nm)
        sim = sim.reshape(x, y)
        
        # Normalization
        # pca_emb[:,0] = (pca_emb[:,0] - pca_emb[:,0].min()) / (pca_emb[:,0].max() - pca_emb[:,0].min())
        # pca_emb[:,1] = (pca_emb[:,1] - pca_emb[:,1].min()) / (pca_emb[:,1].max() - pca_emb[:,1].min())
        # pca_emb[:,2] = (pca_emb[:,2] - pca_emb[:,2].min()) / (pca_emb[:,2].max() - pca_emb[:,2].min())
        sim_i.append(sim)
    
    sim0, sim1= sim_i[0], sim_i[1]
    print("Shape de sim0 et et sim1:", sim0.shape, sim1.shape)
    sim_diff = sim1 - sim0
    print("Shape de sim_diff:", sim_diff.shape)
    print("Type de sim_diff:", sim_diff.type)
    
    print("sim0 mean:", sim0.mean().item())
    print("sim1 mean:", sim1.mean().item())
    print("sim_diff mean:", sim_diff.mean().item())
    print("sim_diff min:", sim_diff.min().item())
    print("sim_diff max:", sim_diff.max().item())
    
    fig, ax = plt.subplots(1,3, figsize=(5,15))
    
    # ax[0].imshow(pca_emb)
    im0 = ax[0].imshow(sim0) 
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(sim1)
    fig.colorbar(im1, ax=ax[1])
    #ax[0].imshow(sim_diff)
    # ax[0].imshow(sim_diff)
    im2 = ax[2].imshow(rgb)
    fig.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    plt.show()
    
    # # x = torch.rand(100,1,343)
    # with torch.no_grad():
    #     print(model(torch.ones(1000,1,343)).mean())
    #     print(model(torch.zeros(1000,1,343)).mean())