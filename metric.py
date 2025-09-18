import os
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse
from data.utils import dict_to_csv
from dipy.io.image import load_nifti


# -----------------------  claculate PSNR and SSIM -----------------------
recon_url = './test/x2q30/'
refer_url = '/media/2033127b-b5be-41bb-a893-3022d6f8b72a/data/hcp/DTI/3T/145x174x145/50_30/single-shell/b1000/test/'
mask_url = '/media/2033127b-b5be-41bb-a893-3022d6f8b72a/data/hcp/DTI/3T/145x174x145/50_30/single-shell/mask/'

recon_files = sorted(os.listdir(recon_url))
refer_files = sorted(os.listdir(refer_url))

psnrs, ssims, nrmses = [], [], []
dicts = {'psnr': [], 'ssim': [], 'nrmse': []}
for i, (recon_file, refer_file) in enumerate(zip(recon_files, refer_files)):
    name = refer_file.split('_')[0]

    print(recon_file, refer_file)
    recon_img, _ = load_nifti(os.path.join(recon_url, recon_file))
    print(recon_img.shape)
    print(recon_img.max(), recon_img.min())

    refer_img, _ = load_nifti(os.path.join(refer_url, refer_file))
    print(refer_img.shape)
    print(refer_img.max(), refer_img.min())

    mask_img, _ = load_nifti(os.path.join(mask_url, name + '_mask.nii.gz'))
    print(mask_img.shape)
    print(mask_img.max(), mask_img.min())

    recon_img = recon_img * mask_img[..., np.newaxis]
    refer_img = refer_img * mask_img[..., np.newaxis]

    d_psnr, d_ssim, d_nrmse = [], [], []
    for d in range(recon_img.shape[-1]):
        psnr = peak_signal_noise_ratio(recon_img[..., d], refer_img[..., d],
                                       data_range=(refer_img[..., d]).max())
        ssim = structural_similarity(recon_img[..., d], refer_img[..., d],
                                     data_range=(refer_img[..., d]).max())
        nrmse = normalized_root_mse(recon_img[..., d], refer_img[..., d])
        d_psnr.append(psnr)
        d_ssim.append(ssim)
        d_nrmse.append(nrmse)

    mean_psnr = np.mean(d_psnr)
    mean_ssim = np.mean(d_ssim)
    mean_nrmse = np.mean(d_nrmse)

    psnrs.append(mean_psnr)
    ssims.append(mean_ssim)
    nrmses.append(mean_nrmse)

    print('psnr:', mean_psnr)
    print('ssim:', mean_ssim)
    print('nrmse:', mean_nrmse)

    dicts['psnr'].append(mean_psnr)
    dicts['ssim'].append(mean_ssim)
    dicts['nrmse'].append(mean_nrmse)

print('mean psnr:', np.mean(psnrs))
print('mean ssim:', np.mean(ssims))
print('mean nrmse:', np.mean(nrmses))

print('std psnr:', np.std(psnrs))
print('std ssim:', np.std(ssims))
print('std nrmse:', np.std(nrmses))

dicts['psnr'].append(np.mean(psnrs))
dicts['ssim'].append(np.mean(ssims))
dicts['nrmse'].append(np.mean(nrmses))

dicts['psnr'].append(np.std(psnrs))
dicts['ssim'].append(np.std(ssims))
dicts['nrmse'].append(np.std(nrmses))

# save
columns = ['psnr', 'ssim', 'nrmse']
dict_to_csv(dicts, columns, './txt/metrics_x2q30.txt')

