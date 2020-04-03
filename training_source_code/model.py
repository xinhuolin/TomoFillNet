import torch.nn as nn
import math
import block as B
import torch


class RRDB_Net(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=16, upscale=4, norm_type=None, act_type='leakyrelu', \
                 mode='CNA', res_scale=1, upsample_mode='upconv'):
        super(RRDB_Net, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero', \
                            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        # elif upsample_mode == 'pixelshuffle':
        #     upsample_block = B.pixelshuffle_block
        elif upsample_mode == 'horpixelshuffle':
            upsample_block = B.horpixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv))
                                  , *upsampler, HR_conv0, HR_conv1, nn.Tanh())

    def forward(self, x):
        x = self.model(x)
        return x


class Inpaint_Net(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=16, nb=16, gc=16, norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(Inpaint_Net, self).__init__()
        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', dilation=(1, 2, 4), groups=1) for _ in
                     range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv))
                                  , HR_conv0, HR_conv1, nn.Sigmoid())

    def forward(self, x):
        # x [0, 1]
        x = self.model(x)
        return x


# Discriminator
class NLD(nn.Module):
    def __init__(self, depth=64, input_nc=1, output_nc=1):
        super(NLD, self).__init__()
        self.norm = nn.GroupNorm
        group = 4
        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc + output_nc, depth, kernel_size=4, stride=2, padding=1),
                                    self.norm(group, depth),
                                    nn.LeakyReLU())
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=1, dilation=1),
                                    self.norm(group, depth * 2),
                                    nn.LeakyReLU())
        # 64
        self.layer3 = nn.Sequential(nn.Conv2d(depth * 2, depth * 2, kernel_size=4, stride=2, padding=1, dilation=1),
                                    self.norm(group, depth * 2),
                                    nn.LeakyReLU())
        # 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1, dilation=1),
            self.norm(group, depth * 4),
            nn.LeakyReLU())
        # 16
        self.layer5 = nn.Sequential(
            nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1, dilation=1),
            self.norm(group, depth * 8),
            nn.LeakyReLU())
        # 8
        self.layer6 = nn.Sequential(
            nn.Conv2d(depth * 8, output_nc, kernel_size=8, stride=1, padding=0, dilation=1))

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        # print(x.size()) # torch.Size([3, 1, 256, 180])
        out = self.layer1(out)
        # print(out.size()) # torch.Size([3, 32, 128, 180])
        out = self.layer2(out)
        # print(out.size()) # torch.Size([3, 64, 62, 88])
        out = self.layer3(out)
        # print(out.size()) # torch.Size([3, 64, 26, 39])
        out = self.layer4(out)
        # print(out.size()) # torch.Size([3, 128, 8, 11])
        out = self.layer5(out)
        # print(out.size())
        out = self.layer6(out)
        # print(out.size())
        return out


# Discriminator
class NLD_inpaint(nn.Module):
    def __init__(self, depth=64, input_nc=1, output_nc=1):
        super(NLD_inpaint, self).__init__()
        self.norm = nn.GroupNorm
        group = 4
        # 256 x 128
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, depth, kernel_size=(4, 3), stride=(2, 1), padding=1),
            self.norm(group, depth),
            nn.LeakyReLU())
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(group, depth * 2),
                                    nn.LeakyReLU())
        # 128 x 128
        self.layer3 = nn.Sequential(nn.Conv2d(depth * 2, depth * 2, kernel_size=4, stride=2, padding=1, dilation=1),
                                    self.norm(group, depth * 2),
                                    nn.LeakyReLU())
        # 64 x 64
        self.layer4 = nn.Sequential(
            nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=4, dilation=4),
            self.norm(group, depth * 4),
            nn.LeakyReLU())
        # 64 x 64
        self.layer5 = nn.Sequential(
            nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1, dilation=1),
            self.norm(group, depth * 8),
            nn.LeakyReLU())
        # 32 x 32
        self.layer6 = nn.Sequential(
            nn.Conv2d(depth * 8, output_nc, kernel_size=3, stride=1, padding=1, dilation=1))
        # 32 x 32

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        # print(x.size()) # torch.Size([3, 1, 256, 180])
        out = self.layer1(out)
        # print(out.size()) # torch.Size([3, 32, 128, 180])
        out = self.layer2(out)
        # print(out.size()) # torch.Size([3, 64, 62, 88])
        out = self.layer3(out)
        # print(out.size()) # torch.Size([3, 64, 26, 39])
        out = self.layer4(out)
        # print(out.size()) # torch.Size([3, 128, 8, 11])
        out = self.layer5(out)
        # print(out.size())
        out = self.layer6(out)
        # print(out.size())
        return out


class NLD_LG_inpaint(nn.Module):
    def __init__(self, depth=64, input_nc=1, output_nc=1):
        super(NLD_LG_inpaint, self).__init__()
        self.norm = nn.GroupNorm
        group = 4
        # 256 x 128 => 128 x 128
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, depth, kernel_size=4, stride=2, padding=1),
            self.norm(group, depth),
            nn.LeakyReLU())
        # 128 => 128
        self.layer1 = nn.Sequential(nn.Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=2, dilation=2),
                                    self.norm(group, depth * 2),
                                    nn.LeakyReLU())
        # 128 => 64
        self.layer2 = nn.Sequential(nn.Conv2d(depth * 2, depth * 2, kernel_size=4, stride=2, padding=1, dilation=1),
                                    self.norm(group, depth * 2),
                                    nn.LeakyReLU())
        # 64 => 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=4, dilation=4),
            self.norm(group, depth * 4),
            nn.LeakyReLU())
        # 64 => 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1, dilation=1),
            self.norm(group, depth * 8),
            nn.LeakyReLU())
        # ===============================================
        # 128 => 64
        self.fast_layer1 = nn.Sequential(
            nn.Conv2d(depth * 1, depth * 2, kernel_size=4, stride=2, padding=1, dilation=1),
            self.norm(group, depth * 2),
            nn.LeakyReLU())
        # 64 = > 32
        self.fast_layer2 = nn.Sequential(
            nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1, dilation=1),
            self.norm(group, depth * 4),
            nn.LeakyReLU())
        # ===============================================
        # 32 => 32
        self.final_layer = nn.Sequential(
            nn.Conv2d(depth * 12, output_nc, kernel_size=3, stride=1, padding=1, dilation=1))

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        # print(x.size())  # torch.Size([3, 1, 256, 180])
        out = self.layer0(out)  # 256 x 128 => 128 x 128
        # print(out.size())  # torch.Size([3, 32, 128, 180])
        fast_out = self.fast_layer1(out)  # 128 => 64
        # print("fast_out_layer1", fast_out.size())
        fast_out = self.fast_layer2(fast_out)  # 64 => 32
        # print("fast_out_layer2", fast_out.size())
        out = self.layer1(out)  # 128 => 128
        # print(out.size())  # torch.Size([3, 64, 62, 88])
        out = self.layer2(out)  # 128 => 64
        # print(out.size())  # torch.Size([3, 64, 26, 39])
        out = self.layer3(out)  # 64 => 64
        # print(out.size())  # torch.Size([3, 128, 8, 11])
        out = self.layer4(out)  # 64 => 32
        # print(out.size())
        out = self.final_layer(torch.cat((out, fast_out), 1))  # 32 => 32
        # print(out.size())
        return out


if __name__ == '__main__':
    import torch

    # m = RRDB_Net(1, 1, 8, 8, 16, 2, upsample_mode="horpixelshuffle")
    # x = torch.ones(1, 1, 4, 4)
    # y = m(x)
    m = NLD_LG_inpaint(4)
    x = torch.ones(1, 1, 32, 16)
    y = m(x, x)
    print(x.size())
    print(y.size())
