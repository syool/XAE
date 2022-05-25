import torch
from torch.autograd import Variable

import numpy as np
from glob import glob
import math
import random
import imageio
import matplotlib.cm

import os


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def seeds(val):
    ''' generate fixed random values '''
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(val)
        torch.cuda.manual_seed_all(val)


def label_encapsule(labels, path, clip_length):
    ''' encapsule labels w.r.t the window size '''
    videos = sorted(glob(os.path.join(path, '*')))
    
    capsules = []
    count = 0
    for vid in videos:
        frames = sorted(glob(os.path.join(vid, '*')))
            
        n_frames = len(frames)
        l = labels[count:count+n_frames]
        
        cap = l[clip_length:]
        capsules = np.append(capsules, cap)
        
        count += n_frames
    
    return capsules


def psnr(mse):
    return 10 * math.log10(1/mse)


def score_norm(arr):
    ''' input must be a numpy array '''
    return (arr - arr.min()) / (arr.max() - arr.min())


def enlarge_image(img, scaling = 3):
    if scaling < 1 or not isinstance(scaling,int):
        print ('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H,W = img.shape
        out = np.zeros((scaling*H, scaling*W))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling] = img[h,w]
    elif len(img.shape) == 3:
        H,W,D = img.shape
        out = np.zeros((scaling*H, scaling*W,D))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling,:] = img[h,w,:]
                
    return out


def hm_to_rgb(R: torch.tensor, scaling: int, cmap: str, normalize: bool = True):
    cmap = eval(f'matplotlib.cm.{cmap}')
    if normalize:
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        # R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping
    R = enlarge_image(R, scaling)
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    
    return rgb


def visualize(relevances, dir, vid_idx, frame_idx):
    # visualize the relevance
    heatmap = np.sum(relevances, axis=3)
    heatmaps = []
    for heat in heatmap:
        maps = (hm_to_rgb(heat, scaling=3, cmap='afmhot')*255).astype(np.uint8)
        heatmaps.append(maps)
        logpath = f'{dir}/results/{vid_idx}'
        os.makedirs(logpath, exist_ok=True)
        imageio.imsave(f'{logpath}/{frame_idx}.jpg', maps, vmax=1, vmin=-1)
