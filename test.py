# -*- coding:utf-8 -*-

import numpy as np
import os
from nets import DCSARL, sh_to_sf_batch, data_fidelity
from data import utils
import torch
import argparse
from tqdm import tqdm
from time import time
from dipy.io.image import load_nifti, save_nifti


def downsample(kspace, czero, mask, scale):
    h, w = kspace.shape[2:]  # [145, 174]
    h_size, w_size = int(h / scale), int(w / scale)
    h_gap, w_gap = (h - h_size) // 2, (w - w_size) // 2

    czero[:, :, h_gap:h_gap + h_size, w_gap:w_gap + w_size] = kspace[:, :, h_gap:h_gap + h_size, w_gap:w_gap + w_size]
    mask[:, :, h_gap:h_gap + h_size, w_gap:w_gap + w_size] = 1

    return czero, mask


if __name__ == '__main__':

    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()

    # about ArSSR model
    parser.add_argument('-encoder', type=str, default='RDN', dest='encoder_name',
                        help='the type of encoder network, including RDN (default), ResCNN, and SRResnet.')
    parser.add_argument('-depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-width', type=int, default=256, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-in_dim', type=int, default=30, dest='in_dim',
                        help='the number of gradient.')
    parser.add_argument('-out_dim', type=int, default=30, dest='out_dim',
                        help='the width of the decoder network.')
    parser.add_argument('-pre_trained_model', type=str,
                        default='./checkpoint/20241011-164827/RDN_d30.pkl',
                        dest='pre_trained_model', help='the file path of LR input image for testing')

    # about GPU
    parser.add_argument('-gpu', type=int, default=2, dest='gpu',
                        help='the number of GPU')

    # about file
    parser.add_argument('-data_path', type=str,
                        default='/media/2033127b-b5be-41bb-a893-3022d6f8b72a/data/hcp/DTI/3T/145x174x145/50_30/single-shell/b1000/test/',
                        dest='data_path',
                        help='the file path of LR input image')
    parser.add_argument('-bvecs', type=str,
                        default='/media/2033127b-b5be-41bb-a893-3022d6f8b72a/data/hcp/DTI/3T/145x174x145/50_30/single-shell/b1000/bvecs',
                        dest='bvecs', help='the file path of bvecs for training')
    parser.add_argument('-output_path', type=str, default='./test/x2q30/', dest='output_path',
                        help='the file save path of reconstructed result')
    parser.add_argument('-scale', type=float, default='2', dest='scale',
                        help='the up-sampling scale k')

    args = parser.parse_args()
    encoder_name = args.encoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    in_dim = args.in_dim
    out_dim = args.out_dim
    pre_trained_model = args.pre_trained_model
    gpu = args.gpu
    data_path = args.data_path
    bvecs = args.bvecs
    output_path = args.output_path
    scale = args.scale
    activation = 'ReLU'
    layer = 3


    # -----------------------
    # model
    # -----------------------

    DEVICE = torch.device('cuda:{}'.format(str(gpu)))

    INRSSR = DCSARL(decoder_depth=int(decoder_depth / 2), decoder_width=decoder_width,
                        in_dim=in_dim, out_dim=out_dim, layers=layer, act=activation, device=DEVICE).to(DEVICE)

    # load pre-trained model weights
    checkpoint = torch.load(pre_trained_model, map_location=DEVICE)['model']

    # 修改key 去除 ‘‘module.’’前缀
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v

    # 将修改后的key加载到网络中
    INRSSR.load_state_dict(new_state_dict)

    d6 = [0, 14, 24, 45, 74, 79]  # in python
    d30 = [7, 9, 10, 15, 19, 26, 27, 28, 29, 32, 33, 34, 35, 43, 50, 54, 55,
           56, 58, 61, 64, 65, 69, 72, 78, 79, 81, 82, 87, 88]  # in python

    # -----------------------
    # SR
    # -----------------------
    INRSSR.eval()
    filenames = sorted(os.listdir(data_path))
    for f in tqdm(filenames):
        start_time = time()
        name = f.split('_')[0]
        hr_dwis, affine = load_nifti(r'{}/{}'.format(data_path, f))  # [145, 174, 145, 90]
        hr_vols = torch.from_numpy(hr_dwis).permute(3, 2, 0, 1).float().to(DEVICE)  # 90xz×h×w
        print(hr_vols.shape)

        bvec = np.genfromtxt(os.path.join(bvecs, name + '_b1000_bvecs'))  # [90, 3]
        bvec = torch.from_numpy(bvec).float().to(DEVICE)
        print(bvec.shape)

        z_slice = np.zeros_like(hr_dwis)  # 145x174x145x90
        for z in range(143):
            hr_slice = hr_vols[d30, z:z+3, :, :]  # 6×3x145×174 d6

            ik = utils.fft2c2d(hr_slice)
            czero = torch.zeros_like(ik)
            mask = torch.zeros_like(hr_slice)

            ckspace, mask = downsample(ik, czero, mask, scale)

            lr_slice = utils.ifft2c2d(ckspace)  # 6×3×145×174

            with torch.no_grad():

                _, _, H, W = lr_slice.shape
                # generate coordinate set
                xy_hr = utils.make_coord(lr_slice.shape[1:], flatten=True)  # [3x145x174, 3]  # 25230
                # print(xy_hr.shape)

                xy_hr = xy_hr.unsqueeze(0).float().to(DEVICE)
                # print(xy_hr.shape)
                xy_hr = xy_hr.repeat(out_dim, 1, 1)  # 6×3x25230×3

                pre = INRSSR(lr_slice.unsqueeze(0), xy_hr.unsqueeze(0), bvec.unsqueeze(0), mask.unsqueeze(0), d30)  # 1xkx28

                # sh_to_sf
                pre = sh_to_sf_batch(pre, bvec.unsqueeze(0), DEVICE)  # 1xkx90
                pre = pre.permute(0, 2, 1).reshape(1, 90, H, W)  # 1x90x145x174

                # dc
                pre = data_fidelity(lr_slice[:, 1, ...].unsqueeze(0), pre, mask[:, 1, ...].unsqueeze(0), d30)  # d30

                pre = pre.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 145x174x90
                z_slice[:, :, z+1, :] = pre

        print(z_slice.shape)

        # save file
        save_nifti(r'{}/SHRL_{}_recon_{}'.format(output_path, encoder_name, f), z_slice, affine)

        print('save to: {}/SHRL_{}_recon_{}'.format(output_path, encoder_name, f))

        print('time: {} m'.format((time() - start_time) / 60.))