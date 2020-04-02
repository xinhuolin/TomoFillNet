# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from PIL import Image, ImageChops
import cv2

fake_path = sys.argv[1]
real_path = sys.argv[2]
result_path = "%s_error.png" % fake_path if len(sys.argv) <= 3 else sys.argv[3]


def absdiff(fake_path, real_path, result_path, norm=False, show=True):
    with Image.open(fake_path) as img:
        fake_img = img.convert("L")
    with Image.open(real_path) as img:
        real_img = img.convert("L")

    fake_np = np.asarray(fake_img, dtype=np.float32)
    real_np = np.asarray(real_img, dtype=np.float32)
    result_np = abs(real_np - fake_np)
    if norm:
        result_np = (result_np - result_np.min()) / (result_np.max() - result_np.min()) * 255
        # print(result_np.min(),result_np.max())
    result_img = Image.fromarray(result_np).convert("L")
    result_img.save(result_path)
    if show:
        result_img.show()


absdiff(fake_path, real_path, result_path, False, True)

"""
def absdiff(fake_path, real_path, result_path)
    fake_img = cv2.imread(fake_path).astype(np.int16)
    real_img = cv2.imread(real_path).astype(np.int16)
    err = cv2.absdiff(real_img,fake_img)  
    cv2.imwrite(result_path,err)
"""
