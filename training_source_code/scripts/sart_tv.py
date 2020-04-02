import numpy as np
from skimage.transform import radon, iradon, iradon_sart
from skimage.restoration import denoise_tv_bregman
from matplotlib import pyplot as plt
from tqdm import tqdm


def sart_tv(tils, angles, sart_iters, tv_w, tv_maxit, tv_eps=1e-3, relaxation=0.3, show=False):
    image = None
    for i in tqdm(range(sart_iters)):
        image = iradon_sart(tils, angles, image=image, relaxation=relaxation)
        if show:
            plt.imshow(image, cmap="gray")
            plt.show()

        image = denoise_tv_bregman(image, tv_w, eps=tv_eps, max_iter=tv_maxit,isotropic=False)
        if show:
            plt.imshow(image, cmap="gray")
            plt.show()
    image = iradon_sart(tils, angles, image=image, relaxation=relaxation)
    return image


if __name__ == '__main__':
    b = np.zeros((256, 256))
    b[100:150, 100:150] = 1
    plt.imshow(b, cmap="gray")
    plt.show()
    angles = np.arange(-50, 50, 4)
    tils = radon(b, angles)
    sart_tv_img = sart_tv(tils, angles, 10, 0.1, 1000, tv_eps=1e-5, relaxation=0.3)
    plt.imshow(sart_tv_img, cmap="gray")
    plt.show()
    sart_tv_img = sart_tv(tils, angles, 10, 0.3, 1000, tv_eps=1e-5, relaxation=0.3)
    plt.imshow(sart_tv_img, cmap="gray")
    plt.show()
    sart_tv_img = sart_tv(tils, angles, 10, 0.5, 1000, tv_eps=1e-5, relaxation=0.3)
    plt.imshow(sart_tv_img, cmap="gray")
    plt.show()
