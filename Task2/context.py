import torch
import os
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from tqdm import tqdm
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


device = "cpu"


def calculate_metrics(model_outputs, gt_path):
    """Calculates average PSNR and SSIM on the passed data

    Parameters
    ----------
        model_outputs
            list of deinterlaced frames
        gt_path : string
            path to gt frames

    Returns
    -------
        (PSNR, SSIM)
    """
    gt_frames = [np.array(Image.open(os.path.join(gt_path, x)), dtype=np.float32) / 255. 
                 for x in sorted(os.listdir(gt_path)) if '.DS_Store' not in x]
    assert len(gt_frames) == len(model_outputs)
    ssim_list = []
    psnr_list = []
    for gt_frame, deinterlaced_frame in tqdm(zip(gt_frames, model_outputs)):
        ssim_list.append(ssim(gt_frame, deinterlaced_frame, multichannel=True))
        psnr_list.append(psnr(gt_frame, deinterlaced_frame))
    return np.mean(ssim_list), np.mean(psnr_list) 


def evaluate(model, interlaced_image):
    with torch.no_grad():
        # Add batch dimension and wrap it into a tensor on the GPU
        model_input = torch.tensor(interlaced_image[np.newaxis, ...], device=device)
        # Predict 
        current_even, next_odd = model(model_input)

        current_even = current_even.cpu().numpy()[0]
        next_odd = next_odd.cpu().numpy()[0]

        first = np.zeros_like(interlaced_image)
        first[::2] = interlaced_image[::2]
        first[1::2] = current_even

        second = np.zeros_like(interlaced_image)
        second[1::2] = interlaced_image[1::2]
        second[::2] = next_odd

        return first, second

