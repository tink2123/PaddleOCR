# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import KaimingNormal
import math
import paddle.nn.functional as F
from paddle.nn.functional import hardswish, hardsigmoid
from paddle.regularizer import L2Decay


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 act='hard_swish',
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(
                name + "_bn_scale", regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(
                name + "_bn_offset", regularizer=L2Decay(0.0)),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class DepthwiseSeparable(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 dw_size=3,
                 padding=1,
                 use_se=False,
                 name=None):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self._depthwise_conv = ConvBNLayer(
            num_channels=make_divisible(num_channels),
            num_filters=make_divisible(int(num_filters1 * scale)),
            filter_size=dw_size,
            stride=stride,
            padding=padding,
            num_groups=make_divisible(int(num_groups * scale)),
            name=name + "_dw")
        if use_se:
            self._se = SEModule(
                make_divisible(int(num_filters1 * scale)), name=name + "_se")
        self._pointwise_conv = ConvBNLayer(
            num_channels=make_divisible(int(num_filters1 * scale)),
            filter_size=1,
            num_filters=make_divisible(int(num_filters2 * scale)),
            stride=1,
            padding=0,
            name=name + "_sep")

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        if self.use_se:
            y = self._se(y)
        y = self._pointwise_conv(y)
        return y


class CPULiteNet(nn.Layer):
    def __init__(self, scale=0.5, class_dim=1000):
        super(CPULiteNet, self).__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            channels=3,
            num_filters=make_divisible(int(32 * scale)),
            stride=2,
            padding=1,
            name="conv1")

        conv2_1 = self.add_sublayer(
            "conv2_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(32 * scale),
                num_filters1=32,
                num_filters2=64,
                num_groups=32,
                stride=1,
                scale=scale,
                name="conv2_1"))
        self.block_list.append(conv2_1)

        conv2_2 = self.add_sublayer(
            "conv2_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(64 * scale),
                num_filters1=64,
                num_filters2=128,
                num_groups=64,
                stride=1,
                scale=scale,
                name="conv2_2"))
        self.block_list.append(conv2_2)

        conv3_1 = self.add_sublayer(
            "conv3_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=128,
                num_groups=128,
                stride=1,
                scale=scale,
                name="conv3_1"))
        self.block_list.append(conv3_1)

        conv3_2 = self.add_sublayer(
            "conv3_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(128 * scale),
                num_filters1=128,
                num_filters2=256,
                num_groups=128,
                stride=(2, 1),
                scale=scale,
                name="conv3_2"))
        self.block_list.append(conv3_2)

        conv4_1 = self.add_sublayer(
            "conv4_1",
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=256,
                num_groups=256,
                stride=1,
                scale=scale,
                name="conv4_1"))
        self.block_list.append(conv4_1)

        conv4_2 = self.add_sublayer(
            "conv4_2",
            sublayer=DepthwiseSeparable(
                num_channels=int(256 * scale),
                num_filters1=256,
                num_filters2=512,
                num_groups=256,
                stride=(2, 1),
                scale=scale,
                name="conv4_2"))
        self.block_list.append(conv4_2)

        for i in range(5):
            conv5 = self.add_sublayer(
                "conv5_" + str(i + 1),
                sublayer=DepthwiseSeparable(
                    num_channels=int(512 * scale),
                    num_filters1=512,
                    num_filters2=512,
                    num_groups=512,
                    stride=1,
                    dw_size=5,
                    padding=2,
                    scale=scale,
                    use_se=False,
                    name="conv5_" + str(i + 1)))
            self.block_list.append(conv5)

        conv5_6 = self.add_sublayer(
            "conv5_6",
            sublayer=DepthwiseSeparable(
                num_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=1024,
                num_groups=512,
                stride=(2, 1),
                dw_size=5,
                padding=2,
                scale=scale,
                use_se=True,
                name="conv5_6"))
        self.block_list.append(conv5_6)

        conv6 = self.add_sublayer(
            "conv6",
            sublayer=DepthwiseSeparable(
                num_channels=int(1024 * scale),
                num_filters1=1024,
                num_filters2=1024,
                num_groups=1024,
                stride=1,
                dw_size=5,
                padding=2,
                use_se=True,
                scale=scale,
                name="conv6"))
        self.block_list.append(conv6)

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        for block in self.block_list:
            y = block(y)
        y = self.pool(y)
        return y


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(name=name + "_1_offset"))
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name + "_2_weights"),
            bias_attr=ParamAttr(name=name + "_2_offset"))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hardsigmoid(outputs)
        return paddle.multiply(x=inputs, y=outputs)
