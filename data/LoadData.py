import os
import torch
import random
import numpy as np
from dipy.io.image import load_nifti
from torch.utils import data

from .utils import fft2c2d, ifft2c2d, make_coord


def sample_gradient(d):
    res_index = []
    for i in range(90):
        if i not in d:
            res_index.append(i)

    return res_index


def downsample(kspace, czero, mask, scale):
    h, w = kspace.shape[2:]  # [145, 174]
    h_size, w_size = int(h / scale), int(w / scale)
    h_gap, w_gap = (h - h_size) // 2, (w - w_size) // 2

    czero[:, :, h_gap:h_gap + h_size, w_gap:w_gap + w_size] = kspace[:, :, h_gap:h_gap + h_size, w_gap:w_gap + w_size]
    mask[:, :, h_gap:h_gap + h_size, w_gap:w_gap + w_size] = 1

    return czero, mask


class HCPDataFidelity(data.Dataset):
    def __init__(self, data_path, bvecs_path, g, out):
        super(HCPDataFidelity).__init__()
        self.g = g
        self.out = out
        self.file_path = []
        filenames = sorted(os.listdir(data_path))
        for filename in filenames:
            name = filename.split('_')[0]
            self.file_path.append((os.path.join(data_path, filename),
                                   os.path.join(bvecs_path, name + '_b1000_bvecs')))

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        d6 = [0, 14, 24, 45, 74, 79]  # in python  # in python
        d30 = [7, 9, 10, 15, 19, 26, 27, 28, 29, 32, 33, 34, 35, 43, 50, 54, 55,
               56, 58, 61, 64, 65, 69, 72, 78, 79, 81, 82, 87, 88]  # in python

        dwi_path, bvec_path = self.file_path[idx]

        dwi, _ = load_nifti(dwi_path)  # [145, 174, 3, 90]
        bvecs = np.genfromtxt(bvec_path)  # [90, 3]

        hr_dwi = torch.from_numpy(dwi).permute(3, 2, 0, 1).float()  # [90, 3, 145, 174]
        bvecs = torch.from_numpy(bvecs).float()  # [90, 3]

        if self.g == 6:
            lar_dwi = hr_dwi[d6, ...]  # [6, 3, 145, 174]

            ik = fft2c2d(lar_dwi)
            czero = torch.zeros_like(ik)
            mask = torch.zeros_like(lar_dwi)

            scale = np.round(random.uniform(2, 3 + 0.04), 1)
            ckspace, mask = downsample(ik, czero, mask, scale)

            # ifft
            clr_dwi = ifft2c2d(ckspace)

            # generate coordinate set
            xy_hr = make_coord(lar_dwi.shape[1:], flatten=True)
            xy_hr = xy_hr.unsqueeze(0).repeat(self.out, 1, 1)  # [6, 75690, 3]

            return clr_dwi, xy_hr, hr_dwi[:, 1, ...], bvecs, mask, np.array(d6)
        elif self.g == 30:
            lar_dwi = hr_dwi[d30, ...]  # [30, 3, 145, 174]

            ik = fft2c2d(lar_dwi)
            czero = torch.zeros_like(ik)
            mask = torch.zeros_like(lar_dwi)

            scale = np.round(random.uniform(2, 3 + 0.04), 1)
            ckspace, mask = downsample(ik, czero, mask, scale)

            # ifft
            clr_dwi = ifft2c2d(ckspace)

            # generate coordinate set
            xy_hr = make_coord(lar_dwi.shape[1:], flatten=True)
            xy_hr = xy_hr.unsqueeze(0).repeat(self.out, 1, 1)  # [6, 75690, 3]

            return clr_dwi, xy_hr, hr_dwi[:, 1, ...], bvecs, mask, np.array(d30)