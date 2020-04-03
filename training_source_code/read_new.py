import os
import shutil
import sys
from functools import wraps

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
torch.multiprocessing.set_start_method('spawn', force=True)

from PIL import Image
from jdit import Model
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse
from skimage.restoration import denoise_tv_bregman
from skimage.transform import radon, iradon, iradon_sart
from tqdm import tqdm

from mypackage.model.unet_standard import NestedUNet


# import multiprocessing as mp


def clear_all_images(root_path, methods):
    classification = ["sinogram", "recon", "recon_fft"]

    paths = [os.path.join(root_path, p, c) for p in methods for c in classification]
    paths += [
        '%s/original_fft' % root_path,
        # 'img_show_TVM/filled_sinogram',
    ]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    print("clear all images!")


def save_np2img(path, img_np_256, norm=True, show=False):
    if norm:
        img_np_256 = 255 * normlize(img_np_256)
    image = Image.fromarray(img_np_256).convert("L")
    if show:
        image.show()
    image.save(path)


def normlize(img_np_256):
    return (img_np_256 - img_np_256.min()) / (img_np_256.max() - img_np_256.min() + 1e-8)


def fft(original_img):
    original_img = normlize(original_img)
    dfgram = np.fft.fftshift(np.fft.fft2(original_img))
    disdfgram = abs(dfgram)
    disdfgram_np = np.log(disdfgram / disdfgram.max() + 1e-5)
    # disdfgram_np = normlize(disdfgram_np)
    # fft_img = Image.fromarray(disdfgram_np * 255)
    # fft_img.show()
    return disdfgram_np


def prepare_model(weights_path, gpu_ids_abs, net, verbose):
    model = Model(net, gpu_ids_abs=gpu_ids_abs, verbose=verbose)
    model.load_weights(weights_path, strict=True)
    if verbose:
        print("load model successfully!")
    model.eval()
    return model.model


class Cache(object):
    def __init__(self, cache_times=1, cache_index_name="cache_index"):
        self.time = cache_times
        self.index = cache_index_name
        self.cache_dict = {}

    def __call__(self, func):
        @wraps(func)
        def wrap(*args, **kwargs):
            if kwargs[self.index] is None:
                return func(*args, **kwargs)
            if kwargs[self.index] not in self.cache_dict:
                result = func(*args, **kwargs)
                self.cache_dict[kwargs[self.index]] = [result, self.time]
            else:
                result = self.cache_dict[kwargs[self.index]][0]
                self.cache_dict[kwargs[self.index]][1] -= 1
                if self.cache_dict[kwargs[self.index]][1] == 0:
                    del self.cache_dict[kwargs[self.index]]
            return result

        return wrap


class Recon(object):
    @staticmethod
    @Cache(2)
    def wbp(sinogram_np, angles, cache_index=None):
        # assert sinogram_np.max() <= 1, str(img_np.max())
        recon_np = iradon(sinogram_np, theta=angles, circle=True)  # iradon 0. 255.
        return recon_np

    @staticmethod
    def deartifact(model, sinogram_np, angles, times, cache_index=None):
        # assert sinogram_np.max() <= 1, str(img_np.max())
        recon_np = Recon.wbp(sinogram_np, angles, cache_index=cache_index)
        # recon_np = Recon.sart(sinogram_np, angles)
        # recon_np = Recon.sirt(sinogram_np, angles,numIter=12)
        # TODO：change to WBP
        # recon_np = iradon(sinogram_np, theta=angles, circle=True) / 255  # iradon 0. 255.
        recon_np = recon_np / 255
        recon_np = recon_np[np.newaxis, np.newaxis, :]
        tensor = torch.Tensor(recon_np).float()
        if len(gpus) > 0:
            tensor = tensor.cuda(device)
        with torch.no_grad():
            out = tensor
            for i in range(times):
                out = model(out).detach()

        recon_np = out.cpu().numpy()[0][0] * 255
        return recon_np

    @staticmethod
    def sirt(sinogram_np, angles, numIter=12):
        # SIRT
        recon_SIRT = iradon(sinogram_np, theta=angles, filter=None, circle=True)
        for i in range(0, numIter):
            reprojs = radon(recon_SIRT, theta=angles, circle=True)
            ratio = sinogram_np / reprojs
            ratio[np.isinf(ratio)] = 1e8
            ratio[np.isnan(ratio)] = 1
            timet = iradon(ratio, theta=angles, filter=None, circle=True)
            timet[np.isinf(timet)] = 1
            timet[np.isnan(timet)] = 1
            recon_SIRT = recon_SIRT * timet
            # print('SIRT  %.3g' % i)
            # plt.imshow(recon_SIRT, cmap=plt.cm.Greys_r)
            # plt.show()
        return recon_SIRT

    @staticmethod
    def sart(sinogram_np, angles, relaxation=0.25, step=10):
        # assert sinogram_np.max() <= 1, str(img_np.max())
        recon_np = None
        for _ in range(step):
            recon_np = iradon_sart(sinogram_np.astype(np.float), angles, image=recon_np, relaxation=relaxation)
        return recon_np

    @staticmethod
    def sart_tvm(tils, angles, sart_iters=10, tv_w=0.1, tv_maxit=1000, tv_eps=1e-5, relaxation=0.3):
        image = None
        for _ in range(sart_iters):
            image = iradon_sart(tils, angles, image=image, relaxation=relaxation)
            image = denoise_tv_bregman(image, tv_w, eps=tv_eps, max_iter=tv_maxit, isotropic=False)
        image = iradon_sart(tils, angles, image=image, relaxation=relaxation)
        return image


def recon(root, index, sinogram, angles, iradon_method, show=False, norm=False):
    # input, sinogram
    if iradon_method == "WBP":
        recon_np = Recon.wbp(sinogram, angles, cache_index=index)
    elif iradon_method == "SART":
        recon_np = Recon.sart(sinogram, angles)
    elif iradon_method == "TVM":
        recon_np = Recon.sart_tvm(sinogram, angles)
    elif iradon_method == "SIRT":
        recon_np = Recon.sirt(sinogram, angles)
    elif iradon_method == "denoise1":
        recon_np = Recon.deartifact(denoise, sinogram, angles, 1, cache_index=index)
    elif iradon_method == "denoise2":
        recon_np = Recon.deartifact(denoise, sinogram, angles, 2, cache_index=index)
    else:
        raise ValueError(str(iradon_method))
    # recon is 0, 256

    save_np2img(os.path.join(root, "%s/recon" % iradon_method, index), recon_np, norm=norm, show=show)
    save_np2img(os.path.join(root, "%s/recon_fft" % iradon_method, index), fft(recon_np), norm=True, show=show)
    recon = Image.fromarray(recon_np).convert("L")
    return np.asarray(recon) / 255


def save_sinogram(sinogram, root, filename, show=False):
    if sinogram.max() <= 1:
        sinogram = sinogram * 255.0
    img = Image.fromarray(sinogram)
    if show:
        img.show()
    img.convert("L").save(os.path.join(root, filename))


class Eval():
    @staticmethod
    def psnr(real, fake, data_range=None):
        return compare_psnr(real, fake, data_range)

    @staticmethod
    def ssim(real, fake, data_range=None):
        return compare_ssim(real, fake, data_range=data_range)

    @staticmethod
    def nrmse(real, fake):
        return compare_nrmse(real, fake)

    @staticmethod
    def evaluate(df, index, target, reference, iradon_method, eval_funcs):
        for name, fuc in eval_funcs.items():
            result = fuc(reference, target)
            df.loc[index, iradon_method + "_" + name] = result

    @staticmethod
    def evaluate_mp(index, target, reference, iradon_method, eval_funcs):
        results = []
        for name, fuc in eval_funcs.items():
            result = fuc(reference, target)
            results.append((index, iradon_method + "_" + name, result))
            # df.loc[index, iradon_method + "_" + name] = result
        return results


def get_angles(wedges, deltas):
    ranges = []
    for g in wedges:
        for d in deltas:
            angles = np.concatenate(
                (np.arange(0, 90 - g / 2., d), np.arange(90 + g / 2., 180, d)))
            ranges.append((angles, g, d))
    return ranges


def compute_denoise(model, input, times, root, recon_type, filename, show=False):
    # (Image.fromarray(input.cpu().numpy()[0][0] * 255).convert("L")).show()
    out = input
    with torch.no_grad():
        for i in range(times):
            out = model(out)
            if len(out) == 4:
                out = model(out)[-1].detach()
            else:
                out = out.detach()
            if show:
                img = Image.fromarray(out.cpu().numpy()[0][0] * 255)
                img.show()
    # out = normlize(out.cpu().numpy()[0][0]) * 255
    out = out.cpu().numpy()[0][0] * 255
    # out = rescale_intensity(out, out_range=(0, 1))* 255
    # out = adjust_log(out)
    img = Image.fromarray(out).convert("L")
    # auto construct
    # img = ImageOps.autocontrast(img)

    img.save(os.path.join(root, "%s_denoise%d/recon" % (recon_type, times), filename))
    save_np2img(os.path.join(root, "%s_denoise%d/recon_fft" % (recon_type, times), filename), fft(out), norm=True)
    return np.asarray(img) / 255


def get_index(filename, wedge, gap):
    return "_".join([filename[:-4], str(wedge), str(gap), filename[-4:]])


def get_filenames(original_imgs, missing_wedge, delta):
    filenames = []
    wedges = []
    deltas = []
    index = []
    for img_name in original_imgs:
        for mw in missing_wedge:
            for d in delta:
                filenames.append(img_name[:-4])
                wedges.append(mw)
                deltas.append(d)
                index.append("_".join([img_name[:-4], str(mw), str(d), img_name[-4:]]))
    return index, filenames, wedges, deltas



def eval(root_path, img_name, iradon_method, ranges, eval_methods):
    path = os.path.join(os.path.join(root_path, "original"), img_name)
    with Image.open(path) as original_img:
        original_img = original_img.convert("L")
        original_img_np = np.asarray(original_img)
        original_fft_np = fft(original_img_np)
        save_np2img(os.path.join(root_path, "original_fft", img_name), original_fft_np, norm=True, show=False)
    results = []
    for range, wedge, delta in ranges:
        index = get_index(img_name, wedge, delta)
        sinogram_np = radon(original_img_np, theta=range, circle=True)  # radon 0.255.

        reference = original_img_np / 255.0  # (0 - 1)
        # recon(0 - 1)
        recon_np = recon(root_path, index, sinogram_np, range, iradon_method, norm=False)
        result = Eval.evaluate_mp(index, recon_np, reference, iradon_method, eval_methods)
        results.append(result)
    return results


def init_dataframe(original_imgs, methods, eval_methods, wedges, deltas):
    feature_names = [m + "_" + e for e in eval_methods for m in methods]
    index, filenames, wedges, deltas = get_filenames(original_imgs, wedges, deltas)
    df = pd.DataFrame(columns=feature_names, index=index)

    df.insert(0, "filename", filenames)
    df.insert(1, "wedge", wedges)
    df.insert(2, "delta", deltas)

    return df


def update(results):
    global pbar
    pbar.update(1)
    for result in results:
        for each_eval in result:
            df.loc[each_eval[0], each_eval[1]] = each_eval[2]


def show_err(e):
    print(e)
    exit(1)


gpus = [1]
device = torch.device(1)

weights_path = "log/%s/checkpoint/Weights_netG_60.pth" % sys.argv[2]
# weights_path = "log/train_1dark/checkpoint/Weights_netG_60.pth"

denoise = prepare_model(weights_path, gpus, NestedUNet(), False)
denoise.share_memory()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = "img_show_test"
    methods = ["WBP",
               "SART",
               "TVM",
               "denoise1",
               "denoise2"
               ]
    eval_methods = {"psnr": Eval.psnr, "nrmse": Eval.nrmse, "ssim": Eval.ssim}
    #wedges = [40, 45, 50, 55, 60, 65, 70, 80]
    wedges = [65]
    deltas = [2]
    print("model:%s" % sys.argv[2], "result:%s" % root_path)

    clear_all_images(root_path, methods)
    original_imgs = os.listdir(os.path.join(root_path, "original"))
    ranges = get_angles(wedges, deltas)
    df = init_dataframe(original_imgs, methods, eval_methods, wedges, deltas)
    print(mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    pbar = tqdm(total=len(methods) * len(original_imgs))
    for img in original_imgs:
        for method in methods:
            pool.apply_async(eval,
                             args=(root_path, img, method, ranges, eval_methods),
                             callback=update,
                             error_callback=show_err)
    pool.close()
    pool.join()
    print("\nfinish")
    df.to_excel(os.path.join(root_path, "result.xls"), index_label="index")
