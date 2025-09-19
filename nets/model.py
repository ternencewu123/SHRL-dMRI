import torch.nn as nn
import torch
import torch.nn.functional as F
from . import decoder
from . import encoder
from .spherical_harmonics_coefficients import sh_to_sf


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


def sh_to_sf_batch(sh, bvec, device, shs=6):
    N = bvec.shape[0]
    batch_sh = []
    for i in range(N):
        batch_sh.append(sh_to_sf(sh[i], bvec[i], sh_order=shs, device=device))  #
    return torch.stack(batch_sh, dim=0)


def data_fidelity(img_lr, pre, mask, index):
    # print(img_lr.shape, pre.shape, mask.shape, len(index))
    pre[:, index, ...] = torch.abs(ifft2c2d(fft2c2d(img_lr) * mask +
                                               fft2c2d(pre[:, index, ...]) * (1 - mask)))
    return pre

class DCSARL(nn.Module):
    def __init__(self, decoder_depth, decoder_width, in_dim, out_dim, layers, act, device):
        super(DCSARL, self).__init__()
        self.device = device
        self.layer = layers
        self.encoder = encoder.RDN(i_dim=in_dim, o_dim=out_dim)
        if act == 'aglu':
            self.decoder = decoder.AGLU_MLP(in_dim=(1 + 3)*out_dim, out_dim=28, width=decoder_width)
            self.decoder2 = decoder.AGLU_MLP(in_dim=(1 + 3)*out_dim*3, out_dim=28, width=decoder_width)
        else:
            self.decoder = decoder.MLP(in_dim=(1 + 3)*out_dim, out_dim=28, depth=decoder_depth, width=decoder_width)
            self.decoder2 = decoder.MLP(in_dim=(1 + 3)*out_dim*3, out_dim=28, depth=decoder_depth, width=decoder_width)

    def forward(self, img_lr, xy_hr, bvec, mask, index):
        """
        :param img_lr: Nx6x3x145x174
        :param xy_hr: Nx6x(3x145x174)x3
        :param sampling: [6]
        :param bvec: Nx90x3
        :param mask: Nx6x3x145x174
        :return: Nx90x145x174
        """
        temp = torch.abs(img_lr)
        for i in range(self.layer):
            feature_map = self.encoder(temp)
            N, C, S, H, W = feature_map.shape

            feature_vector = F.grid_sample(feature_map, xy_hr[:, 0, ...].flip(-1).unsqueeze(1).unsqueeze(1),
                                           mode='bilinear', align_corners=False)[:, :, 0, 0, :]
            # print(feature_vector.shape)  

            feature_vector_and_xy_hr = torch.cat([xy_hr, feature_vector.unsqueeze(-1)], dim=-1)  

            if i < self.layer - 1:
                feature_vector_and_xy_hr = feature_vector_and_xy_hr.permute(0, 2, 1, 3).contiguous()  
                _, K = feature_vector_and_xy_hr.shape[:2]
                intensity_sh = self.decoder(feature_vector_and_xy_hr.reshape(N * K, -1)).reshape(N, K, -1)  

                # sh_to_sf
                intensity_pre = sh_to_sf_batch(intensity_sh, bvec, self.device)  
                intensity_pre = intensity_pre.permute(0, 2, 1).reshape(N, bvec.shape[1], S, H, W)  

                # data fidelity
                pre = data_fidelity(img_lr, intensity_pre, mask, index)  

                temp = pre[:, index, ...]  
            else:
                feature_vector_and_xy_hr = feature_vector_and_xy_hr.reshape(N, C, S, H * W, 4)
                feature_vector_and_xy_hr = feature_vector_and_xy_hr.permute(0, 3, 1, 2, 4).contiguous()  
                _, K = feature_vector_and_xy_hr.shape[:2]
                intensity_sh = self.decoder2(feature_vector_and_xy_hr.reshape(N * K, -1)).reshape(N, K, -1)  
                x_final = intensity_sh

        return x_final
