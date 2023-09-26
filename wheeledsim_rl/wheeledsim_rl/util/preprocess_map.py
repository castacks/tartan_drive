import numpy as np
import torch
import torchvision
import copy
import time
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.functional import interpolate
from offroad_env.AirsimMap import AirsimMap

def preprocess_map(envmap, preprocess_dict):
    """
    Preprocess the input map. Convert seg to one-hot, normalize heights and RGBs.
    Args:
        envmap: the map to preprocess
        preprocess_dict: A dict contaning the segmentation mappings, as well as the mean and std of the RGB channels and heightmap.
    """
    return [
            preprocess_rgbmap(torch.tensor(envmap.mapRGB).permute(2, 0, 1), preprocess_dict).float(),
            preprocess_heightmap(torch.tensor(envmap.mapHeight), preprocess_dict).float(),
            preprocess_segmap(torch.tensor(envmap.mapSeg), preprocess_dict).float(),
            torch.tensor(envmap.mapMask).unsqueeze(0).float()
            ]

def preprocess_segmap(segmap, preprocess_dict):
    """
    get a dict of segmentation classes and return the one-hotted segmap.
    """
    seg_dict = preprocess_dict['segmentation_classes']

    out = torch.zeros(len(seg_dict.keys()), *segmap.shape)
    for k,v in seg_dict.items():
        out[v] = (segmap == k).float()

    return out

def preprocess_heightmap(heightmap, preprocess_dict):
    if len(heightmap.shape) == 2:
        heightmap = heightmap.float().unsqueeze(0)
    mu = preprocess_dict['heightmap_mean']
    sig = preprocess_dict['heightmap_std']
    return (heightmap - mu)/sig

def preprocess_rgbmap(rgbmap, preprocess_dict):
    """
    Normalize per-channel
    """
    if rgbmap.dtype == torch.float:
        print("Expects map values as integers from 0-255 not 0-1 floats.")

    mu = preprocess_dict['rgb_mean'].unsqueeze(1).unsqueeze(1)
    sig = preprocess_dict['rgb_std'].unsqueeze(1).unsqueeze(1)

    return (rgbmap - mu)/sig

if __name__ == '__main__':
    import os

    config_fp = 'env_configs'
    config_dirs = os.listdir(config_fp)
    config_dirs = [x for x in config_dirs if not os.path.isfile(os.path.join(config_fp, x))] #Dirs, not files.

    pdict = torch.load(os.path.join(config_fp, 'preprocess_dict.pt'))

    print(pdict)

    for cdir in config_dirs:
        map_fp = os.path.join(config_fp, cdir, 'map.pt')
        envmap = torch.load(map_fp)

        rgb, height, seg, mask = preprocess_map(envmap, pdict)

        plt.imshow(rgb.permute(1, 2, 0));plt.show()
        plt.imshow(height[0]);plt.show()
        plt.imshow(seg.argmax(dim=0));plt.show()
        plt.imshow(mask[0]);plt.show()
