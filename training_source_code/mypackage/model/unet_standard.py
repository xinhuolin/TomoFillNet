# -*- coding: utf-8 -*-
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


class VGGBlock(nn.Module):
    # nn.LeakyReLU(0.1),act_func=nn.SELU(),
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.SELU(), use_drop=.0):
        # nn.ReLU
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.use_drop = use_drop
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.AlphaDropout(use_drop)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_drop:
            out = self.drop(out)
        out = self.bn.1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        if self.use_drop:
            out = self.drop(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


# ReLu => Leaky Relu
# Dropout add to VGGBlock
class NestedUNet(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=False, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0], use_drop=0)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], use_drop=0.2)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], use_drop=0.2)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], use_drop=0.5)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], use_drop=0.5)

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], use_drop=0)
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], use_drop=0.2)
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2], use_drop=0.2)
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3], use_drop=0.5)

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0], use_drop=0)
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1], use_drop=0.2)
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2], use_drop=0.5)

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0], use_drop=0)
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1], use_drop=0.2)

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0], use_drop=0)

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = nn.Sigmoid()(self.final1(x0_1))
            output2 = nn.Sigmoid()(self.final2(x0_2))
            output3 = nn.Sigmoid()(self.final3(x0_3))
            output4 = nn.Sigmoid()(self.final4(x0_4))
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return nn.Sigmoid()(output)


class NestedUNet_UNION(nn.Module):
    def __init__(self, in_channels=1, deepsupervision=True, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()
        self.deepsupervision = deepsupervision
        # nb_filter = [32, 64, 128, 256, 512]
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0], use_drop=0)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = nn.Sigmoid()(self.final1(x0_1))
            output2 = nn.Sigmoid()(self.final2(x0_2))
            output3 = nn.Sigmoid()(self.final3(x0_3))
            output4 = nn.Sigmoid()(self.final4(x0_4))
            return [[output1, output2, output3, output4], [x1_0, x2_0, x3_0, x4_0]]

        else:
            output = self.final(x0_4)
            return nn.Sigmoid()(output)


class NLD_UNION(nn.Module):
    def __init__(self, output_nc=1, nb_filter=(32, 64, 128, 256, 512)):
        super(NLD_UNION, self).__init__()
        self.norm = nn.BatchNorm2d
        # d2(128) => d5(16)
        self.d2 = nn.Sequential(
            nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=1, stride=1, padding=0),
            self.norm(nb_filter[1]), nn.LeakyReLU(0.2),
            nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[2]), nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
            nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[3]), nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
            nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[4]), nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2)
        )
        # d3(64) => d5(16)
        self.d3 = nn.Sequential(
            nn.Conv2d(nb_filter[2], nb_filter[2], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[2]), nn.LeakyReLU(0.2),
            nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[3]), nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
            nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[4]), nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2)
        )
        # d4(32) => d5(16)
        self.d4 = nn.Sequential(
            nn.Conv2d(nb_filter[3], nb_filter[3], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[3]), nn.LeakyReLU(0.2),
            nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[4]), nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2)
        )
        # d5(16) => d5(16)
        self.d5 = nn.Sequential(
            nn.Conv2d(nb_filter[4], nb_filter[4], kernel_size=3, stride=1, padding=1),
            self.norm(nb_filter[4]), nn.LeakyReLU(0.2),
        )

        self.d2_out = nn.Sequential(nn.Conv2d(nb_filter[4], output_nc, kernel_size=3, stride=1, padding=1))
        self.d3_out = nn.Sequential(nn.Conv2d(nb_filter[4], output_nc, kernel_size=3, stride=1, padding=1))
        self.d4_out = nn.Sequential(nn.Conv2d(nb_filter[4], output_nc, kernel_size=3, stride=1, padding=1))
        self.d5_out = nn.Sequential(nn.Conv2d(nb_filter[4], output_nc, kernel_size=3, stride=1, padding=1))
        # 256 x 256
        self.layer = nn.Sequential(nn.Conv2d(2, nb_filter[0], kernel_size=3, stride=1, padding=1),
                                   self.norm(nb_filter[0]),
                                   nn.LeakyReLU(0.2),
                                   # 256
                                   nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=3, stride=1, padding=1),
                                   self.norm(nb_filter[1]),
                                   nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
                                   # 128
                                   nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=3, stride=1, padding=1),
                                   self.norm(nb_filter[2]),
                                   nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
                                   # 64
                                   nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=3, stride=1, padding=1),
                                   self.norm(nb_filter[3]),
                                   nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
                                   # 32
                                   nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=3, stride=1, padding=1),
                                   self.norm(nb_filter[4]),
                                   nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
                                   # 16
                                   )
        # # 256 x 256
        # self.layer1 = nn.Sequential(nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=3, stride=1, padding=1),
        #                             self.norm(nb_filter[1]),
        #                             nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 128 x 128
        # self.layer2 = nn.Sequential(nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=3, stride=1, padding=1),
        #                             self.norm(nb_filter[2]),
        #                             nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 64 x 64
        # self.layer3 = nn.Sequential(nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=3, stride=1, padding=1),
        #                             self.norm(nb_filter[3]),
        #                             nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # # 32 x 32
        # self.layer4 = nn.Sequential(nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=3, stride=1, padding=1),
        #                             self.norm(nb_filter[4]),
        #                             nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 16 x 16
        self.layer_out = nn.Sequential(nn.Conv2d(nb_filter[4], output_nc, kernel_size=3, stride=1, padding=1))
        # d5 * 5
        self.union = nn.Sequential(nn.Conv2d(nb_filter[4] * 5, nb_filter[4], kernel_size=1, stride=1, padding=0),
                                   self.norm(nb_filter[4]),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(nb_filter[4], nb_filter[4], kernel_size=3, stride=1, padding=1),
                                   self.norm(nb_filter[4]),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(nb_filter[4], 1, kernel_size=3, stride=1, padding=1),

                                   )

    def forward(self, x, g, latents):
        d2, d3, d4, d5 = latents
        d2 = self.d2(d2)
        d3 = self.d3(d3)
        d4 = self.d4(d4)
        d5 = self.d5(d5)
        d2_out = self.d2_out(d2)
        d3_out = self.d3_out(d3)
        d4_out = self.d4_out(d4)
        d5_out = self.d5_out(d5)

        f = self.layer(torch.cat((x, g), 1))
        f_out = self.layer_out(f)
        union_out = self.union(torch.cat((d2, d3, d4, d5, f), 1))
        return d2_out, d3_out, d4_out, d5_out, f_out, union_out


class UNet(nn.Module):
    def __init__(self, in_channels=1, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()

        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = nn.Sigmoid()(self.final(x0_4))
        return output


class NLD(nn.Module):
    def __init__(self, depth=64, input_nc=1, output_nc=1, norm_type="batch"):
        super(NLD, self).__init__()
        self.norm = nn.BatchNorm2d

        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc + output_nc, depth, kernel_size=7, stride=1, padding=3),
                                    self.norm(depth),
                                    nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(depth * 2),
                                    nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(depth * 4),
                                    nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(depth * 4, depth * 8, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(depth * 8),
                                    nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2))
        # 16 x 16
        self.layer5 = nn.Sequential(nn.Conv2d(depth * 8, depth * 8, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(depth * 8),
                                    nn.LeakyReLU(0.2), nn.AvgPool2d(2, 2))
        # 8 x 8
        self.layer6 = nn.Sequential(nn.Conv2d(depth * 8, output_nc, kernel_size=8, stride=1, padding=0))

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        return out


if __name__ == '__main__':
    from jdit import Model

    unet = Model(NLD_UNION(), [])
    print(unet.num_params)
    input = torch.randn((2, 1, 256, 256), requires_grad=True)
    d2 = torch.randn((2, 64, 128, 128), requires_grad=True)
    d3 = torch.randn((2, 128, 64, 64), requires_grad=True)
    d4 = torch.randn((2, 256, 32, 32), requires_grad=True)
    d5 = torch.randn((2, 512, 16, 16), requires_grad=True)
    target = torch.Tensor([1]).squeeze()
    output = unet(input, input, (d2, d3, d4, d5))
    print(sum(i.mean() for i in output), target)
    res = torch.autograd.gradcheck(torch.nn.MSELoss(), (sum(i.mean() for i in output), target), eps=1e-6,
                                   raise_exception=True)

    print(output.size())
