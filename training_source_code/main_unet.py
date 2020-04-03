import torch
from jdit import Optimizer, Model
from jdit.trainer import Pix2pixGanTrainer

from dataset import RadonNoisedDatasets
from mypackage.model.unet_standard import NestedUNet, NLD
from tools.evaluate import get_nrmse, get_ssim, get_psnr


# pytorch.set_default_tensor_type('torch.DoubleTensor')
class RadonPix2pixGanTrainer(Pix2pixGanTrainer):

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets):
        super(RadonPix2pixGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets)
        # self.gan_rate = 0.04

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach(), self.input)
        d_real = self.netD(self.ground_truth, self.input)
        var_dic = {}
        var_dic["LOSS_D/RaLS_GAN"] = (torch.mean((d_real - d_fake.mean() - 1) ** 2) +
                                      torch.mean((d_fake - d_real.mean() + 1) ** 2)) / 2
        return var_dic["LOSS_D/RaLS_GAN"], var_dic

    def compute_g_loss(self):
        var_dic = {}
        # print(self.input.cpu().detach().numpy().min(), self.input.cpu().numpy().max())
        # print(self.ground_truth.cpu().detach().numpy().min(), self.ground_truth.cpu().numpy().max())
        d_fake = self.netD(self.fake, self.input)
        d_real = self.netD(self.ground_truth, self.input)
        var_dic["LOSS_G/GAN"] = (torch.mean((d_real - d_fake.mean() + 1) ** 2) +
                                 torch.mean((d_fake - d_real.mean() - 1) ** 2)) / 2
        # var_dic["LOSS_G/MSE"] = mse_loss(self.fake, self.ground_truth)
        # var_dic["LOSS_G/RaLS_GAN"] = self.gan_rate * var_dic["LOSS_G/GAN"] + var_dic["LOSS_G/MSE"]
        var_dic["LOSS_G/RaLS_GAN"] = var_dic["LOSS_G/GAN"]
        return var_dic["LOSS_G/RaLS_GAN"], var_dic

    def compute_valid(self):
        var_dic = {}
        d_fake = self.netD(self.fake.detach(), self.input)
        d_real = self.netD(self.ground_truth, self.input)
        var_dic["LOSS_D/RaLS_GAN"] = (torch.mean((d_real - d_fake.mean() - 1) ** 2) +
                                      torch.mean((d_fake - d_real.mean() + 1) ** 2)) / 2

        var_dic["LOSS_G/GAN"] = (torch.mean((d_real - d_fake.mean() + 1) ** 2) +
                                 torch.mean((d_fake - d_real.mean() - 1) ** 2)) / 2
        # var_dic["LOSS_G/MSE"] = mse_loss(self.fake, self.ground_truth)
        # var_dic["LOSS_G/RaLS_GAN"] = self.gan_rate * var_dic["LOSS_G/GAN"] + var_dic["LOSS_G/MSE"]
        var_dic["LOSS_G/RaLS_GAN"] = var_dic["LOSS_G/GAN"]
        var_dic["Eval/PSNR"] = torch.from_numpy(get_psnr(self.ground_truth.detach(), self.fake.detach()))
        var_dic["Eval/SSIM"] = torch.from_numpy(get_ssim(self.ground_truth.detach(), self.fake.detach()))
        var_dic["Eval/NRMSE"] = torch.from_numpy(get_nrmse(self.ground_truth.detach(), self.fake.detach()))
        return var_dic

    def test(self):
        pass

    def _watch_images(self, tag: str, grid_size: tuple = (3, 3), shuffle=False, save_file=True):
        self.watcher.image(self.fake,
                           self.current_epoch,
                           tag="%s/fake" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.ground_truth,
                           self.current_epoch,
                           tag="%s/real" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.input,
                           self.current_epoch,
                           tag="%s/input" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

    def valid_epoch(self):
        super(Pix2pixGanTrainer, self).valid_epoch()
        self.netG.eval()
        self.netD.eval()
        if self.fixed_input is None:
            for batch in self.datasets.loader_test:
                if isinstance(batch, (list, tuple)):
                    self.fixed_input, self.fixed_ground_truth = self.get_data_from_batch(batch, self.device)
                    self.watcher.image(self.fixed_ground_truth, self.current_epoch, tag="Fixed/groundtruth",
                                       grid_size=(6, 6),
                                       shuffle=False)
                    self.watcher.image(self.fixed_input, self.current_epoch, tag="Fixed/input",
                                       grid_size=(6, 6),
                                       shuffle=False)
                else:
                    self.fixed_input = batch.to(self.device)
                self.watcher.image(self.fixed_input, self.current_epoch, tag="Fixed/input",
                                   grid_size=(6, 6),
                                   shuffle=False)
                break

        # watching the variation during training by a fixed input
        with torch.no_grad():
            fake = self.netG(self.fixed_input).detach()
        self.watcher.image(fake, self.current_epoch, tag="Fixed/fake", grid_size=(6, 6), shuffle=False)
        # saving training processes to build a .gif.
        self.watcher.set_training_progress_images(fake, grid_size=(6, 6))

        self.netG.train()
        self.netD.train()


if __name__ == '__main__':
    gpus = [0, 1]
    logdir = "log/spd60_80"
    print(logdir)
    batch_size = 16  # 32
    valid_size = 5000
    nepochs = 60
    G_hprams = {"optimizer": "Adam",
                "lr_decay": 0.1,
                "decay_position": [40, 55],
                "position_type": "epoch",
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "amsgrad": False
                }
    D_hprams = {"optimizer": "RMSprop",
                "lr_decay": 0.1,
                "decay_position": [40, 55],
                "position_type": "epoch",
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "alpha": 0.99,
                "momentum": 0
                }

    torch.backends.cudnn.benchmark = True
    print('===> Build dataset')
    datasets = RadonNoisedDatasets("/home/dgl/dataset", "stpd40_80", batch_size=batch_size, valid_size=valid_size,
                                   num_workers=2)
    print('===> Building model')

    # G_net = UNet(1, (64, 128, 256, 512, 1024))  # (32, 64, 128, 256, 512),  (64, 128, 256, 512, 1024)
    # G_net = resnet10(no_cuda=False)
    G_net = NestedUNet()  # 7
    D_net = NLD(32)  # 64
    # D_net = NLD_LG_inpaint(64)
    net_G = Model(G_net, gpus, verbose=True, check_point_pos=10)
    net_D = Model(D_net, gpus, verbose=True, check_point_pos=10)
    print('===> Building optimizer')
    optG = Optimizer(net_G.parameters(), **G_hprams)
    optD = Optimizer(net_D.parameters(), **D_hprams)
    print('===> Training')
    Trainer = RadonPix2pixGanTrainer(logdir, nepochs, gpus, net_G, net_D, optG, optD, datasets)

    import sys

    _DEBUG_ = len(sys.argv) > 1 and sys.argv[1].strip().lower() == "-d"
    if _DEBUG_:
        Trainer.debug()
    else:
        Trainer.train(show_network=False, subbar_disable=False)
