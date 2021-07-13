""" Contains the ML models"""
# expanded by Lukas Abfalterer in 2021 (labfalterer a.t. student.ethz.ch)
# reimplemented code of Sheila Zingg in 2019 by David Sommer (david.sommer at inf.ethz.ch) in 2020

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm


def param_variance_scaled_norm_trunc(shape, threshold=2, scale=0.1, loc=0):
    # scale down such that the expectation value of the final sum over all
    # sampled variances is scale**2 if threshold -> infty
    scale = scale / np.sqrt(np.prod(shape))

    samples = truncnorm.rvs(a=-threshold, b=threshold, loc=loc, scale=scale, size=shape)

    return torch.nn.Parameter(data=torch.from_numpy(samples))


class CNNModel(torch.nn.Module):
    def _make_dense(self, in_features, out_features):
        dense = torch.nn.Linear(in_features, out_features)
        dense.weight = param_variance_scaled_norm_trunc(dense.weight.shape, threshold=2)
        dense.bias = param_variance_scaled_norm_trunc(dense.bias.shape, threshold=2)
        return dense

    def _make_conv1d_padding_same(self, L_in, in_channels, out_channels, kernel_size, activation_func=None):
        """ returns a keras inspired conv1d layer with activation function. stride and dilation ist set to 1. """
        # using numpy 'same' padding definition.
        # using the formula given from torch doc:
        #   L_out​ = [(L_in​ + 2 * padding - dilation * (kernel_size - 1) - 1​ ) / stride ] + 1
        # with stride = delation = 1 leads to
        #   padding = (L_out - L_in + kernel_size - 1) / 2
        L_out = max(L_in, kernel_size)
        padding = (L_out - L_in + kernel_size) / 2
        # assert padding == int(padding), "kernel_size % 2 == 1 is not true"
        padding = int(padding)

        print(f"L_in {L_in} | L_out {L_out} | kernel_size {kernel_size} | padding {padding} | {2*padding + L_in - kernel_size}  {2*padding - L_in + kernel_size}")

        layer1 = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="zeros",
        )

        with torch.no_grad():
            layer1.weight.data = param_variance_scaled_norm_trunc(layer1.weight.shape, threshold=2)
        if activation_func is None:
            return [layer1]
        a = activation_func()
        return [layer1, a]

    # for dtype: https://discuss.pytorch.org/t/how-to-set-dtype-for-nn-layers/31274

    def __init__(self, element_size, range_begin, args, device):
        super().__init__()

        self.monotone = args.monotone
        self.network_width = 100
        self.element_size = element_size
        self.half_elements = int(element_size / 2)
        self.channels = 1

        # conv1d configuration
        conv_activation = nn.CELU  # None #tf.nn.sigmoid
        kernel_size = 15
        assert kernel_size % 2 == 1
        L_in = self.network_width  # max(self.network_width, kernel_size)
        print(f"L_in {L_in}")
        self.intern_size = self.half_elements
        self.dense1 = self._make_dense(in_features=self.half_elements, out_features=self.network_width)

        conv_layers_list = []

        first_conv_layer = self._make_conv1d_padding_same(
            L_in=self.network_width,
            in_channels=1,
            out_channels=self.channels,
            kernel_size=kernel_size,
            activation_func=conv_activation,
        )
        conv_layers_list += first_conv_layer

        for _ in range(args.cnns - 1):
            layer = self._make_conv1d_padding_same(
                L_in=L_in,
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=kernel_size,
                activation_func=conv_activation,
            )
            conv_layers_list += layer

        layer = self._make_conv1d_padding_same(
            L_in=L_in,
            in_channels=self.channels,
            out_channels=1,
            kernel_size=kernel_size,
            activation_func=conv_activation,
        )
        conv_layers_list += layer

        self.conv_layers = nn.Sequential(*conv_layers_list)

        self.dense3 = self._make_dense(in_features=L_in, out_features=self.intern_size)

        self.bias = nn.Parameter(torch.tensor(10 ** -30).double())
        self.bias.requires_grad = True

    def forward2(self, x):
        # only use one half
        x = x[: self.element_size // 2]

        x = x.view(1, 1, -1)

        x = self.dense1(x)
        x = x.view(1, 1, -1)
        x = self.conv_layers(x)
        x = self.dense3(x)
        x = x.view(-1)

        if self.monotone:
            x = F.relu(x)
            x = torch.cumsum(x, dim=0)

        x = torch.cat((x, torch.flip(x, dims=(0,))))

        x = torch.add(x, self.bias)
        x = torch.log(x)
        x = F.softmax(x, dim=0)
        return x

    def forward(self, x):
        # only use one half
        x = x[: self.element_size // 2]

        x = self.dense1(x)
        x = x.view(1, 1, -1)
        x = self.conv_layers(x)
        x = self.dense3(x)
        x = x.view(-1)

        if self.monotone:
            x = F.relu(x)
            x = torch.cumsum(x, dim=0)

        x = torch.cat((x, torch.flip(x, dims=(0,))))

        x = torch.add(x, self.bias)
        x = F.softmax(x, dim=0)
        return x


class SigmoidModel(torch.nn.Module):
    """Sigmoid Model"""

    def __init__(self, element_size, range_begin, args, device):
        super().__init__()

        self.monotone = args.monotone
        self.element_size = element_size

        if args.random_init:
            self.scales = nn.Parameter(torch.rand(args.sig_num).double())
        else:
            self.scales = nn.Parameter(torch.tensor(args.sig_num * [args.scale_start]).double())
        self.scales.requires_grad = True

        self.slopes = torch.tensor(args.sig_num * [args.slope]).double()
        self.slopes = self.slopes.to(device)

        self.c = nn.Parameter(torch.tensor(np.linspace(-range_begin, 0, args.sig_num, endpoint=False)).double())
        self.c.requires_grad = True

        self.d = nn.Parameter(torch.tensor(args.bias).double())
        self.d.requires_grad = True

    def forward(self, x):
        # Make everything positive
        slopes = self.slopes
        scales = self.scales * self.scales
        c = self.c
        d = self.d * self.d

        # Only use half
        x = x[: self.element_size // 2]
        x = torch.tensordot(x, slopes, 0) - torch.mul(c, slopes)
        x = torch.sigmoid(x)
        x = torch.mul(scales, x)
        x = torch.sum(x, 1)
        x = x + d

        # Flip over
        x = torch.cat((x, torch.flip(x, dims=(0,))))

        x = torch.log(x)
        x = F.softmax(x, dim=0)
        return x


class MixtureModel(torch.nn.Module):
    """Mixture Model"""

    def __init__(self, element_size, range_begin, args, device):
        super().__init__()

        self.monotone = args.monotone
        self.element_size = element_size

        if args.random_init:
            self.scalesX = nn.Parameter(torch.rand(args.sig_num).double())
        else:
            self.scalesX = nn.Parameter(torch.tensor(args.sig_num * [args.scale_start]).double())
        self.scalesX.requires_grad = True

        self.slopesX = torch.tensor(args.sig_num * [args.slope]).double()
        self.slopesX = self.slopesX.to(device)

        self.cX = nn.Parameter(torch.tensor(np.linspace(-range_begin, range_begin, args.sig_num, endpoint=True)).double())
        self.cX.requires_grad = True

        if args.random_init:
            self.scalesY = nn.Parameter(torch.rand(args.sig_num).double())
        else:
            self.scalesY = nn.Parameter(torch.tensor(args.sig_num * [args.scale_start]).double())
        self.scalesX.requires_grad = True

        self.slopesY = torch.tensor(args.sig_num * [args.slope]).double()
        self.slopesY = self.slopesY.to(device)

        self.cY = nn.Parameter(torch.tensor(np.linspace(-range_begin, range_begin, args.sig_num, endpoint=True)).double())
        self.cY.requires_grad = True

        self.d = nn.Parameter(torch.tensor(args.bias).double())
        self.d.requires_grad = True

    def forward(self, input):
        # Make everything positive
        slopes = self.slopesX
        scales = self.scalesX * self.scalesX
        c = self.cX
        d = self.d * self.d

        x = torch.tensordot(input, slopes, 0) - torch.mul(c, slopes)
        x = torch.sigmoid(x)
        x = torch.mul(scales, x)
        x = torch.sum(x, 1)

        slopes = self.slopesY
        scales = self.scalesY * self.scalesY
        c = self.cY

        y = torch.tensordot(input, slopes, 0) - torch.mul(c, slopes)
        y = torch.sigmoid(y)
        y = torch.mul(scales, y)
        y = torch.sum(y, 1)

        x = x - y
        x = x + d
        x = torch.abs(x)
        # x = F.softmax(torch.log(x), dim=0)
        x = x / x.sum()
        return x


class MidpointModel(torch.nn.Module):
    """Mixture Model"""

    def __init__(self, element_size, range_begin, args, device):
        super().__init__()

        self.monotone = args.monotone
        self.element_size = element_size

        if args.random_init:
            self.scales = nn.Parameter(torch.rand(args.sig_num).double())
        else:
            self.scales = nn.Parameter(torch.tensor(args.sig_num * [args.scale_start]).double())
        self.scales.requires_grad = True

        self.slopes = torch.tensor(args.sig_num * [args.slope]).double()
        self.slopes = self.slopes.to(device)

        self.c = nn.Parameter(torch.tensor(np.linspace(-range_begin, range_begin, args.sig_num, endpoint=True)).double())
        self.c.requires_grad = True

        self.d = nn.Parameter(torch.tensor(args.bias).double())
        self.d.requires_grad = True

        self.midpoint = nn.Parameter(torch.tensor(element_size // 2).double())
        self.midpoint.requires_grad = True
        self.mask = torch.zeros(element_size, device=device)

    def forward(self, input):
        slopes = self.slopes
        scales = self.scales * self.scales
        # c = self.c * self.c
        # c, _ = torch.sort(c)
        c = self.c
        d = self.d * self.d

        x = torch.tensordot(input, slopes, 0) - torch.mul(c, slopes)
        x = torch.sigmoid(x)
        x = torch.mul(scales, x)
        x = torch.sum(x, 1)

        subtract = x[-1] - x[torch.ceil(self.midpoint).int()]
        print(subtract, "last", x[-1], "mid", x[torch.ceil(self.midpoint).int()])
        print(torch.isnan(x).any(), subtract, torch.min(x), torch.max(x))
        x[torch.ceil(self.midpoint).int() + 1 :] = torch.flip(x[torch.ceil(self.midpoint).int() + 1 :], dims=(0,)) - subtract
        print(torch.isnan(x).any(), subtract, torch.min(x), torch.max(x))
        x = x + d
        print(torch.isnan(x).any(), subtract, torch.min(x), torch.max(x))
        x = torch.log(F.relu(x) + d)
        x = torch.log(torch.abs(x) + d)
        x = torch.log(x)
        print("AFTER LOG", torch.isnan(x).any(), subtract, torch.min(x), torch.max(x))
        x = F.softmax(x, dim=0)
        print(torch.isnan(x).any(), torch.min(x), torch.max(x))
        if torch.isnan(x).any():
            raise Exception
        return x


class CumSumCModel(torch.nn.Module):
    """CumSumC Model"""

    def __init__(self, element_size, range_begin, args, device):
        super().__init__()

        self.monotone = args.monotone
        self.element_size = element_size

        if args.random_init:
            self.scales = nn.Parameter(torch.rand(args.sig_num).double())
        else:
            self.scales = nn.Parameter(torch.tensor(args.sig_num * [args.scale_start]).double())
        self.scales.requires_grad = True

        self.slopes = torch.tensor(args.sig_num * [args.slope]).double()
        self.slopes = self.slopes.to(device)

        self.c = nn.Parameter(torch.tensor([2 * range_begin / args.sig_num] * args.sig_num).double())
        self.c.requires_grad = True

        self.d = nn.Parameter(torch.tensor(args.bias).double())
        self.d.requires_grad = True

        self.midpoint = nn.Parameter(torch.tensor(element_size // 2).double())
        self.midpoint.requires_grad = True
        self.mask = torch.zeros(element_size, device=device)

        self.range_begin = torch.tensor(range_begin, device=device)
        self.mask = torch.tensor([1] * range_begin + [-1] * range_begin, device=device)

    def normalize(self, x):
        min = torch.min(x)
        max = torch.max(x)
        return ((2 * self.range_begin) * (x - min)) / (max - min) - self.range_begin

    def forward(self, input):
        slopes = self.slopes
        scales = self.scales * self.scales
        scales = scales * self.mask
        # c = self.c * self.c
        # c, _ = torch.sort(c)
        c = torch.cumsum(self.c, 0)
        # print(c[0], c[999])
        c = self.normalize(c)
        d = self.d * self.d

        x = torch.tensordot(input, slopes, 0) - torch.mul(c, slopes)
        x = torch.sigmoid(x)
        x = torch.mul(scales, x)
        x = torch.sum(x, 1)

        # subtract = x[-1] - x[torch.ceil(self.midpoint).int()]
        # print(subtract, "last", x[-1], "mid", x[torch.ceil(self.midpoint).int()])
        # print(torch.isnan(x).any(), subtract, torch.min(x), torch.max(x))
        # x[torch.ceil(self.midpoint).int() + 1 :] = torch.flip(x[torch.ceil(self.midpoint).int() + 1 :], dims=(0,)) - subtract
        # print(torch.isnan(x).any(), subtract, torch.min(x), torch.max(x))
        # x = x + d  # des eventuell usse toa...
        # print(torch.isnan(x).any(), subtract, torch.min(x), torch.max(x))
        # x = torch.log(F.relu(x) + d)
        x = torch.log(x + torch.abs(torch.min(x)) + d)
        # x = torch.log(x)
        # print("AFTER LOG", torch.isnan(x).any(), subtract, torch.min(x), torch.max(x))
        x = F.softmax(x, dim=0)
        # print(torch.isnan(x).any(), torch.min(x), torch.max(x))
        if torch.isnan(x).any():
            raise Exception
        return x
