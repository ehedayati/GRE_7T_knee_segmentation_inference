# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch-only implementation of the archived MONAI UNet configurations.

Adapted from MONAI 1.6.0 UNet, Convolution, ResidualUnit, ADN and SkipConnection.
See THIRD_PARTY_NOTICES.md and licenses/MONAI-LICENSE.txt for attribution.
Supports only 3-D, kernel-3 convolutions, instance normalization, scalar PReLU,
zero dropout and concatenation skips used by this release. Module names match
MONAI so existing state dictionaries load strictly, unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class _Convolution(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, *, transposed=False,
                 conv_only=False):
        super().__init__()
        if transposed:
            conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                                      stride=stride, padding=1,
                                      output_padding=stride - 1, bias=True)
        else:
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                             stride=stride, padding=1, bias=True)
        self.add_module("conv", conv)
        if not conv_only:
            adn = nn.Sequential()
            adn.add_module("N", nn.InstanceNorm3d(
                out_channels, eps=1e-5, momentum=0.1, affine=False,
                track_running_stats=False))
            adn.add_module("D", nn.Dropout(p=0.0, inplace=False))
            adn.add_module("A", nn.PReLU(num_parameters=1, init=0.25))
            self.add_module("adn", adn)


class _ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, subunits=2, *,
                 last_conv_only=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        for index in range(subunits):
            self.conv.add_module(f"unit{index}", _Convolution(
                in_channels if index == 0 else out_channels, out_channels,
                stride if index == 0 else 1,
                conv_only=last_conv_only and index == subunits - 1))
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Conv3d(
                in_channels, out_channels, kernel_size=1 if stride == 1 else 3,
                stride=stride, padding=0 if stride == 1 else 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        return self.conv(x) + residual


class _SkipConnection(nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.submodule(x)], dim=1)


class UNet3D(nn.Module):
    """Checkpoint-compatible UNet/UNet2 using only standard torch.nn layers.

    Inputs: [batch, echoes, depth, height, width]. Spatial sizes must be divisible
    by the product of strides; inference pads/crops as before. Returns class
    logits with the same semantics as the original MONAI model.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 channels: Sequence[int], strides: Sequence[int],
                 num_res_units: int = 0):
        super().__init__()
        channels, strides = tuple(channels), tuple(strides)
        if len(channels) < 2 or len(strides) != len(channels) - 1:
            raise ValueError("Need two or more channel levels and one stride per transition")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (in_channels, out_channels, *channels, *strides)):
            raise ValueError("Channels and scalar strides must be positive integers")
        if not isinstance(num_res_units, int) or isinstance(num_res_units, bool) or num_res_units < 0:
            raise ValueError("num_res_units must be a nonnegative integer")

        def down(inc, outc, stride):
            if num_res_units:
                return _ResidualUnit(inc, outc, stride, num_res_units)
            return _Convolution(inc, outc, stride)

        def block(inc, outc, level_channels, level_strides, is_top):
            c, stride = level_channels[0], level_strides[0]
            if len(level_channels) > 2:
                subblock = block(c, c, level_channels[1:], level_strides[1:], False)
                up_channels = 2 * c
            else:
                subblock = down(c, level_channels[1], 1)
                up_channels = c + level_channels[1]
            encoder = down(inc, c, stride)
            decoder = _Convolution(up_channels, outc, stride, transposed=True,
                                   conv_only=is_top and num_res_units == 0)
            if num_res_units:
                decoder = nn.Sequential(decoder, _ResidualUnit(
                    outc, outc, subunits=1, last_conv_only=is_top))
            return nn.Sequential(encoder, _SkipConnection(subblock), decoder)

        self.model = block(in_channels, out_channels, channels, strides, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
