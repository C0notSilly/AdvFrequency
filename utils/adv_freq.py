import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import itertools
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.fft as fft


def FFT_Batch(image):
    img_fft = fft.fft2(image)
    img_fft = torch.fft.fftshift(img_fft)
    img_fft_amp = torch.abs(img_fft)  # 幅度谱
    img_fft_phase = torch.angle(img_fft)  # 相位谱
    return img_fft_amp, img_fft_phase


def IFFT_Batch(amp, phase):
    img_recon = amp * torch.exp(1j * phase)
    img_recon = torch.fft.ifftshift(img_recon)
    img_recon = torch.abs(torch.fft.ifft2(img_recon))
    img_recon = img_recon / torch.max(img_recon)
    return img_recon


def FIF(image):
    amp, phase = FFT_Batch(image)
    recon_image = IFFT_Batch(amp, phase)
    return amp, recon_image

def FreqMask(img, half_mask_w, half_mask_h):
    B, C, H, W = img.size()

    half_H, half_W = H // 2, W // 2

    amp, phase = FFT_Batch(img)

    img_recon = amp * torch.exp(1j * phase)

    mask_l = torch.zeros(amp.shape, device=amp.device)
    mask_l[:, :, half_H - half_mask_h:half_H + half_mask_h, half_W - half_mask_w:half_W + half_mask_w] = 1

    # reconstruct low frequency

    img_recon_l = img_recon * mask_l
    img_recon_l = torch.fft.ifftshift(img_recon_l)
    img_recon_l = torch.abs(torch.fft.ifft2(img_recon_l))
    img_recon_l = img_recon_l / torch.max(img_recon_l)

    # same for high frequency
    mask_h = torch.ones(amp.shape, device=amp.device)
    mask_h[:, :, half_H - half_mask_h:half_H + half_mask_h, half_W - half_mask_w:half_W + half_mask_w] = 0

    img_recon_h = img_recon * mask_h
    img_recon_h = torch.fft.ifftshift(img_recon_h)
    img_recon_h = torch.abs(torch.fft.ifft2(img_recon_h))
    img_recon_h = img_recon_h / torch.max(img_recon_h)

    img_recon_h = img_recon_h.detach()
    img_recon_l = img_recon_l.detach()

    return min_max_scaler(img_recon_l).detach(), min_max_scaler(img_recon_h).detach()



def min_max_scaler(ten):
    flatten_tensor = ten.view(ten.shape[0], ten.shape[1], -1)
    flatten_max = torch.max(flatten_tensor, dim=2, keepdim=True).values
    flatten_min = torch.min(flatten_tensor, dim=2, keepdim=True).values

    expand_max = flatten_max.unsqueeze(dim=3).expand(ten.shape)
    expand_min = flatten_min.unsqueeze(dim=3).expand(ten.shape)

    scaler_ten = (ten - expand_min) / (expand_max - expand_min)
    return scaler_ten


class UnNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
