import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from jdit import Model
from skimage.transform import iradon, iradon_sart
from tqdm import tqdm

from mypackage.model.unet_standard import NestedUNet

torch.multiprocessing.set_start_method('spawn', force=True)


def update(r):
    pbar.update(1)


def error(e):
    print(e)
    # exit(1)


def init_test_img():
    anglesFile = scio.loadmat("real_imgs/angles.mat")
    tiltsFile = scio.loadmat("real_imgs/tiltsercorr_realign.mat")
    angles = anglesFile["angles"][0]
    titls = tiltsFile["tiltsercorr_realign"]
    # titl = titls[:, 50, :]
    # res = iradon(titl, angles)
    # plt.imshow(res, cmap="gray")
    # plt.show()
    return titls, angles

def init_test_img():
    anglesFile = scio.loadmat("real_imgs/angles.mat")
    tiltsFile = scio.loadmat("real_imgs/tiltsercorr_realign.mat")
    angles = anglesFile["angles"][0]
    titls = tiltsFile["tiltsercorr_realign"]
    # titl = titls[:, 50, :]
    # res = iradon(titl, angles)
    # plt.imshow(res, cmap="gray")
    # plt.show()
    return titls, angles


def iradon_and_save(sinogram_np, angles, wbp_dir, sart_dir, denoise_dir, sart_denoise_dir):
    sinogram_np = sinogram_np * 0.6
    recon_np = iradon(sinogram_np, angles)[20:276, 20:276]
    Image.fromarray(recon_np * 255).convert("L").save(wbp_dir)
    recon_np = recon_np[np.newaxis, np.newaxis, :]
    tensor = torch.Tensor(recon_np).float()
    tensor = tensor.cuda(device)
    with torch.no_grad():
        out = model(tensor).detach()

    recon_np = out.cpu().numpy()[0][0] * 255
    Image.fromarray(recon_np).convert("L").save(denoise_dir)
    # ---------------------------
    recon_np = None
    for _ in range(10):
        recon_np = iradon_sart(sinogram_np.astype(np.float), angles, image=recon_np, relaxation=0.25)
    recon_np = recon_np[20:276, 20:276]
    Image.fromarray(recon_np * 255).convert("L").save(sart_dir)
    recon_np = recon_np[np.newaxis, np.newaxis, :]
    tensor = torch.Tensor(recon_np).float()
    tensor = tensor.cuda(device)
    with torch.no_grad():
        out = model(tensor).detach()

    recon_np = out.cpu().numpy()[0][0] * 255
    Image.fromarray(recon_np).convert("L").save(sart_denoise_dir)


def deartifact(img, gpus, save_name):
    recon_np = np.asarray(img) / 255
    recon_np = recon_np[np.newaxis, np.newaxis, :]
    tensor = torch.Tensor(recon_np).float()
    if len(gpus) > 0:
        tensor = tensor.cuda(device)
    with torch.no_grad():
        out = model(tensor).detach()

    recon_np = out.cpu().numpy()[0][0] * 255
    Image.fromarray(recon_np * 255).convert("L").save(save_name)


device = torch.device(1)
weights_path = "log/spd40_80/checkpoint/Weights_netG_60.pth"
model = Model(NestedUNet(), gpu_ids_abs=[1])
model.load_weights(weights_path, strict=True)
model.eval()
model = model.model
model.share_memory()

if __name__ == '__main__':
    titls, angles = init_test_img()
    length = titls.shape[1]  # 321
    pbar = tqdm(total=length)
    pool = Pool(mp.cpu_count())
    for i in range(length):
        titl = titls[:, i, :]
        pool.apply_async(iradon_and_save,
                         (titl, angles,
                          "eval_img/wbp/%03d.png" % i,
                          "eval_img/sart/%03d.png" % i,
                          "eval_img/denoise/%03d.png" % i,
                          "eval_img/sart_denoise/%03d.png" % i,
                          ),
                         callback=update,
                         error_callback=error)
    pool.close()
    pool.join()

    # imgs_seq = None
    # for img in os.listdir("eval_img"):
    #     path = os.path.join("eval_img", img)
    #     with Image.open(path) as img:
    #         img = img.convert("L")
    #     img_np = np.asarray(img)[np.newaxis, :]
    #     if imgs_seq is None:
    #         imgs_seq = img_np
    #     else:
    #         imgs_seq = np.concatenate([imgs_seq, img_np], 0)
    # print(imgs_seq.shape)
    # seq = []
    # for i in range(imgs_seq.shape[2]):
    #     img = imgs_seq[:, :, i]
    #     seq.append(Image.fromarray(img))
    # mimsave('eval.gif', seq, duration=0.1)

    #
    # for img_name in tqdm(os.listdir("eval_img/original")):
    #     path = os.path.join("eval_img/original", img_name)
    #     img = np.load()
    #     with Image.open(path) as img:
    #         img = img.convert("L")
    #     deartifact(img, [1], "eval_img/denoise/%s" % img_name)
    #
    print("Finish!")
