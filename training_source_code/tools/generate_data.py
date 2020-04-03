from skimage.transform import radon, rescale
from skimage.transform import iradon, iradon_sart
from skimage import img_as_ubyte, img_as_float, img_as_float32
import numpy as np
import random
from tqdm import tqdm
import warnings
import os
from PIL import Image
import torch
from torchvision.transforms import *

r_affine = RandomAffine(180, (0.02, 0.02), (0.4, 1.2), 50)
r_resize = RandomResizedCrop(256, (0.8, 1.0), ratio=(0.5, 2.0))
r_vert = RandomVerticalFlip()
r_rotate = RandomRotation(180)
r_trans = Compose([r_affine, r_resize, r_vert, r_rotate])



def generate_data(paths, angles, save_root=r"..\data", nums=1000):
    warnings.filterwarnings('error')
    for i in tqdm(range(nums)):
        path = random.choice(paths)
        with Image.open(path) as img:
            original = img.convert("F")
        original_img = r_trans(original)
        (original_img.convert("L")).save(r"%s\original_img\%04d.png" % (save_root, i))
        repeat = True
        while repeat:
            try:
                original_np = np.asarray(original_img)  # 0. 255.
                angles = np.concatenate((np.arange(0, 67.5, 1.40625), np.arange(112.5, 180, 1.40625)))
                tiltser_np, recon_bp_np = radnon_and_iradnon(original_np, angles)

                # original_img = Image.fromarray(original_np)
                # tiltser_img = Image.fromarray(((tiltser_np-tiltser_np.min())/(tiltser_np.max() - tiltser_np.min()))*255)
                # tiltser_img = InnerPad(32)(tiltser_img)
                # bp_img = Image.fromarray(bp_np)
                # original_img.show()
                # tiltser_img.show()
                # exit(1)
                # bp_img.show()

                # (Image.fromarray(tiltser)).show()
                np.save(r"%s\tiltser\%04d" % (save_root, i), tiltser_np)
                np.save(r"%s\recon_bp\%04d" % (save_root, i), recon_bp_np)
                repeat = False
            except UserWarning as w:
                print("regenerate!!")
                original_img = r_trans(original_img)
                (original_img.convert("L")).save(r"%s\original_img\%04d.png" % (save_root, i))
                repeat = True


def generate_dense_data(dense_angles,
                        original_img_path=r"..\data\original_img",
                        save_root=r"..\data"):
    names = os.listdir(original_img_path)
    for name in tqdm(names):
        path = os.path.join(original_img_path, name)
        with Image.open(path) as img:
            original_img = img.convert("F")
        original_np = np.asarray(original_img)  # 0. 255.
        dense_tiltser_np, dense_recon_bp_np = radnon_and_iradnon(original_np, dense_angles)
        np.save(r"%s\dense_tiltser\%s" % (save_root, name[:-4]), dense_tiltser_np)
        np.save(r"%s\dense_recon_bp\%s" % (save_root, name[:-4]), dense_recon_bp_np)


def radnon_and_iradnon(original_np, angles):
    assert original_np.max() == 255
    tiltser_np = radon(original_np / original_np.max(), theta=angles, circle=True)  # radon 0.255.
    recon_bp_np = iradon(tiltser_np, theta=angles, filter=None, circle=True)  # iradon 0. 255.
    return tiltser_np, recon_bp_np


if __name__ == '__main__':
    # names = os.listdir(r"..\data\original")
    # paths = [os.path.join(r"..\data\original", name) for name in names]
    # angles = np.arange(-90., 90., 11.25)  # 16
    # angles = np.arange(-90., 90., 5.625)  # 32
    # dense_angles = np.arange(-90., 90., 1.40625)  # 128
    # generate_data(paths, angles, 1000)
    # generate_dense_data(dense_angles)
    # ================================================
    # Inpaint
    # names = os.listdir(r"..\data_inpaint\original")
    # paths = [os.path.join(r"..\data_inpaint\original", name) for name in names]
    # angles_96 = np.arange(-68., 67., 1.40625)  # 96 ， 96 +16 +16
    # angles_128 = np.arange(-90., 90., 1.40625)  # 128 ， 128 +16 +16
    # generate_data(paths, angles_96, r"..\data_inpaint", 1000)
    # generate_dense_data(angles_128,
    #                     original_img_path=r"..\data_inpaint\original_img",
    #                     save_root=r"..\data_inpaint")
    # ================================================
    # angles = np.arange(-70., 70., 4.357)  # 32
    # angles = np.arange(-70., 70., 2.1875)  # 64
    # dense_angles = np.arange(-70., 70., 1.09375)  # 128
    # dense_angles = np.arange(-87.5., 87.5., 1.09375)  # 160 ， 128 +16 +16
    # dense_angles = np.arange(-90., 90., 1.09375)  # 164.5714  =》165
    # ================================================
    # Inner Inpaint
    names = os.listdir(r"..\data_inner_inpaint\original")
    paths = [os.path.join(r"..\data_inner_inpaint\original", name) for name in names]

    def get_inner_degrees(full_angles, inner_num=32):
        assert isinstance(inner_num, int) and inner_num % 2 == 0
        center = len(full_angles) // 2
        angles = [full_angles[center]]
        for i in range(1, inner_num // 2 + 1):
            if len(angles) < inner_num:
                angles.insert(0, full_angles[center - i])
            if len(angles) < inner_num:
                angles.append(full_angles[center + i])
        return np.array(angles)

    # angles_128 = np.arange(0., 180., 1.40625)  # 128 ， 128 +16 +16
    # angles = np.concatenate((np.arange(0, 67.5, 1.40625), np.arange(112.5, 180, 1.40625))) # 96 ，48 +(32) + 48
    angles_128 = np.arange(0., 180., 1.40625)  # 128 ， 128 +16 +16
    angles = np.concatenate((np.arange(0, 67.5, 1.40625), np.arange(112.5, 180, 1.40625))) # 96 ，48 +(32) + 48

    generate_data(paths, angles, r"..\data_inner_inpaint", 1000)
    generate_dense_data(angles_128,
                        original_img_path=r"..\data_inner_inpaint\original_img",
                        save_root=r"..\data_inner_inpaint")
