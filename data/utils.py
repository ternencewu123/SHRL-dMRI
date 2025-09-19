import numpy as np
import logging
import pandas as pd
import torch.nn as nn
import torch.fft
import pytorch_wavelets as pw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F
import torch


def sample_gradient(d):
    res_index = []
    for i in range(90):
        if i not in d:
            res_index.append(i)

    return res_index


def fft2c3d(data):
    data = torch.fft.ifftshift(data, dim=(-3, -2, -1))
    data = torch.fft.fftn(data, dim=(-3, -2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-3, -2, -1))
    return data


def ifft2c3d(data):
    data = torch.fft.ifftshift(data, dim=(-3, -2, -1))
    data = torch.fft.ifftn(data, dim=(-3, -2, -1), norm='ortho')
    data = torch.fft.fftshift(data, dim=(-3, -2, -1))
    return data


def fft2c2d(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def ifft2c2d(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifftn(x, dim=(-2, -1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def make_coord(shape, ranges=None, flatten=True):
    """
    shape: [90, 10, 10, 10]
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])

    return ret


def batch_psnr_ssim(recon, ref):
    # recon: [b, k]
    recon, ref = recon.detach().cpu().numpy(), ref.detach().cpu().numpy()
    b_psnr, b_ssim = [],  []
    for i in range(recon.shape[0]):
        for j in range(recon.shape[1]):
            b_psnr.append(peak_signal_noise_ratio(recon[i, j, ...], ref[i, j, ...], data_range=ref[i, j, ...].max()))
            b_ssim.append(structural_similarity(recon[i, j, ...], ref[i, j, ...], data_range=ref[i, j, ...].max()))

    return np.mean(b_psnr), np.mean(b_ssim)


def get_logger(filepath):
    logger = logging.getLogger('SHRL')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def dict_to_csv(data, columns, url):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(url, index=False)

class WaveletFrequencyLoss(nn.Module):
    def __init__(self, device, wavelet='db4', levels=3):
        super(WaveletFrequencyLoss, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.dwt = pw.DWTForward(J=self.levels, wave=self.wavelet, mode='zero').to(device)

    def forward(self, input, target):
        # apply DWT
        _, input_details = self.dwt(input)  # Approximation Coefficients and Detail Coefficients[_, []]
        _, target_details = self.dwt(target)

        # calculate frequency loss
        wavelet_loss = 0.
        for level in range(self.levels):  # Detail Coefficients
            input_coeff = input_details[level]
            target_coeff = target_details[level]
            wavelet_loss += F.l1_loss(input_coeff, target_coeff)

        return wavelet_loss

class MyWFLoss(nn.Module):
    def __init__(self, gamma, device):
        super(MyWFLoss, self).__init__()
        self.gamma = gamma
        self.wavelet_loss = WaveletFrequencyLoss(device)

    def forward(self, pre, gt):
        # l1 loss
        l1_loss = F.l1_loss(pre, gt)

        # wavelet loss
        wavelet_loss = self.wavelet_loss(pre, gt)

        # total loss
        total_loss = l1_loss + self.gamma * wavelet_loss
        # print('l1_loss: {}, wavelet_loss: {}'.format(l1_loss, wavelet_loss))
        return l1_loss, wavelet_loss, total_loss