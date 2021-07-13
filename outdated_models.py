""" Contains outdated and experimental ML Models"""
# expanded by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reimplemented code of Sheila Zingg in 2019 by David Sommer (david.sommer at inf.ethz.ch) in 2020

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import register_gradient_hook


class SigmoidModelCumSumLocation(torch.nn.Module):
    """Sigmoid Model"""

    def __init__(self, element_size, range_begin, args):
        super().__init__()
        self.element_size = element_size

        if args.random_init:
            self.scales = nn.Parameter(torch.rand(args.sig_num).double())
        else:
            self.scales = nn.Parameter(torch.tensor(args.sig_num * [args.scale_start]).double())
        self.scales.requires_grad = True

        self.slopes = torch.tensor(args.sig_num * [10]).double()

        # On Opposite flip
        # self.c = nn.Parameter(torch.tensor(np.linspace(0, range_begin, args.sig_num, endpoint=False)).double())
        # Random setting
        # self.c = nn.Parameter(torch.rand(args.sig_num).double() * range_begin)
        # self.c = nn.Parameter(torch.tensor(np.linspace(-math.sqrt(range_begin), 0, args.sig_num, endpoint=False)).double())
        # Uniform
        self.c = nn.Parameter(torch.tensor(args.sig_num * [range_begin / args.sig_num]).double())
        # self.c = nn.Parameter(torch.tensor(np.linspace(-range_begin, 0, args.sig_num, endpoint=False)).double())
        self.c.requires_grad = True

        self.d = nn.Parameter(torch.tensor(10).double())
        self.d.requires_grad = True

    def forward(self, x):
        # Make everything positive
        scales = self.scales * self.scales
        # try out c with relu

        c = torch.cumsum(torch.abs(self.c), dim=0)
        print("C", c, "self.c", self.c)
        d = self.d * self.d

        # Only use half
        x = x[: self.element_size // 2]
        # print(x, torch.tensordot(x, self.slopes, 0))
        print(torch.tensordot(x, self.slopes, 0))
        x = torch.tensordot(x, self.slopes, 0) - torch.mul(c, self.slopes)
        print(c)
        x = torch.sigmoid(x)
        x = torch.mul(scales, x)
        x = torch.sum(x, 1)
        x = x + d

        x = torch.cat((x, torch.flip(x, dims=(0,))))

        x = torch.log(x)
        x = F.softmax(x, dim=0)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, element_size, range_begin, args):
        super().__init__()

        self.activation = F.relu
        # 1x1x6000
        self.c1 = nn.Conv1d(1, 32, kernel_size=31, stride=1)
        # 1x32×5970
        self.c2 = nn.Conv1d(32, 32, kernel_size=31, stride=5)
        # 1x32x1188
        self.c3 = nn.Conv1d(32, 32, kernel_size=31, stride=5)
        # 1x64x232
        self.c4 = nn.Conv1d(32, 32, kernel_size=31, stride=1)
        # 1x64x202
        self.c5 = nn.Conv1d(32, 32, kernel_size=31, stride=1)
        # 1x128x172
        self.d5 = nn.ConvTranspose1d(32, 32, kernel_size=31, stride=1)
        # 1x64x202
        self.d4 = nn.ConvTranspose1d(32, 32, kernel_size=31, stride=1)
        # 1x64x232
        self.d3 = nn.ConvTranspose1d(32, 32, kernel_size=31, stride=5, output_padding=2)
        # 1x32x1188
        self.d2 = nn.ConvTranspose1d(32, 32, kernel_size=31, stride=5, output_padding=4)
        # 1x32x5970
        self.d1 = nn.ConvTranspose1d(32, 1, kernel_size=31, stride=1)
        # 1x32x6000

    def forward(self, x):
        x = x.view(1, 1, -1)

        x = self.activation(self.c1(x))
        x = self.activation(self.c2(x))
        x = self.activation(self.c3(x))
        x = self.activation(self.c4(x))
        x = self.activation(self.c5(x))

        x = self.activation(self.d5(x))
        x = self.activation(self.d4(x))
        x = self.activation(self.d3(x))
        x = self.activation(self.d2(x))
        x = self.activation(self.d1(x))

        return x.view(-1)


class ReluModel(torch.nn.Module):
    """Relu Model"""

    def __init__(self, element_size, range_begin, args):
        super().__init__()

        self.monotone = args.monotone
        self.element_size = element_size

        self.scales = nn.Parameter(torch.tensor(self.element_size // 2 * [1]).double())
        self.scales.requires_grad = True

        self.element_bias = nn.Parameter(torch.tensor(self.element_size // 2 * [10 ** -2]).double())
        self.element_bias.requires_grad = True

        self.bias = nn.Parameter(torch.tensor(10 ** -5).double())
        self.bias.requires_grad = True

    def forward(self, x):
        # Only use half
        x = x[: self.element_size // 2]
        x = x + self.element_bias
        register_gradient_hook(x, "After elementwise bias")

        x = torch.relu(x)
        register_gradient_hook(x, "After relu")
        x = torch.mul(self.scales, x)
        register_gradient_hook(x, "After multiplication with scale")

        if self.monotone:
            x = F.relu(x)
            register_gradient_hook(x, "After monotone activation")
            x = torch.cumsum(x, dim=0)
            register_gradient_hook(x, "After cumsum")
            x = torch.cat((x, torch.flip(x, dims=(0,))))
            register_gradient_hook(x, "After cat")

        x = x + self.bias
        print("Result", x)
        register_gradient_hook(x, "After softmax")
        return x


class ReluFlipModel(torch.nn.Module):
    """ReluFlip Model"""

    def __init__(self, element_size, range_begin, args):
        super().__init__()

        self.element_size = element_size

        if args.random_init:
            self.scales = nn.Parameter(torch.rand(self.element_size // 2, args.sig_num).double())
        else:
            self.scales = nn.Parameter(torch.tensor(args.sig_num * [args.scale_start]).double())
        self.scales.requires_grad = True

        self.slopes = torch.tensor(args.sig_num * [500]).double()

        # On Opposite flip
        # self.c = nn.Parameter(torch.tensor(np.linspace(0, range_begin, args.sig_num, endpoint=False)).double())
        # Random setting
        # self.c = nn.Parameter(torch.rand(args.sig_num).double() * range_begin)
        # self.c = nn.Parameter(torch.tensor(np.linspace(-math.sqrt(range_begin), 0, args.sig_num, endpoint=False)).double())
        # Uniform
        self.c = nn.Parameter(torch.tensor(args.sig_num * [range_begin / args.sig_num]).double())
        # self.c = nn.Parameter(torch.tensor(np.linspace(-range_begin, 0, args.sig_num, endpoint=False)).double())
        self.c.requires_grad = True

        self.d = nn.Parameter(torch.tensor(10).double())
        self.d.requires_grad = True

    def forward(self, x):
        # Make everything positive
        d = self.d * self.d

        # Only use half
        x = x[: self.element_size // 2]
        x = torch.tensordot(x, self.slopes, 0)  # - torch.mul(self.c, self.slopes)
        x = torch.mul(self.scales, x)
        x = torch.sigmoid(x)
        x = torch.sum(x, 1)
        x = x + d

        x = torch.cat((x, torch.flip(x, dims=(0,))))

        x = torch.log(x)
        x = F.softmax(x, dim=0)
        return x
