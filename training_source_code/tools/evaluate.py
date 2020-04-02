from skimage.measure import compare_psnr, compare_ssim, compare_nrmse
import numpy as np


def get_psnr(real, fake, data_range=None):
    psnr = 0
    length = len(real)
    for i in range(length):
        psnr += compare_psnr(real[i][0].cpu().detach().numpy(), fake[i][0].cpu().detach().numpy(), data_range)
    return np.array([psnr / length])


def get_ssim(real, fake, data_range=None):
    ssim = 0
    length = len(real)
    for i in range(length):
        ssim += compare_ssim(real[i][0].cpu().detach().numpy(), fake[i][0].cpu().detach().numpy(),data_range=data_range)
    return np.array([ssim / length])


def get_nrmse(real, fake):
    nrmse = 0
    length = len(real)
    for i in range(length):
        nrmse += compare_nrmse(real[i][0].cpu().detach().numpy(), fake[i][0].cpu().detach().numpy())
    return np.array([nrmse / length])


# def get_msssim(real, fake):
    # ms_ssim = 0
    # length = len(real)
    # for i in range(length):
    #     ms_ssim += msssim(real[i][0].cpu().detach().numpy(), fake[i][0].cpu().detach().numpy())
    # return np.array([ms_ssim / length])
