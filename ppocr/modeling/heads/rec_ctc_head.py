# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import math

import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F


def get_para_bias_attr(l2_decay, k):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]


class Embedding(nn.Layer):
    def __init__(self, in_timestep, in_planes, mid_dim=4096, embed_dim=300):
        super(Embedding, self).__init__()
        self.in_timestep = in_timestep
        self.in_planes = in_planes
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.eEmbed = nn.Linear(
            in_planes*in_timestep,
            self.embed_dim)  # Embed encoder output to a word-embedding like

    def forward(self, x):
        x = paddle.reshape(x, [paddle.shape(x)[0], -1])
        x = self.eEmbed(x)
        return x

class CTCHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            weight_attr, bias_attr = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels)
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        else:
            weight_attr1, bias_attr1 = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels)
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                weight_attr=weight_attr1,
                bias_attr=bias_attr1)

            weight_attr2, bias_attr2 = get_para_bias_attr(
                l2_decay=fc_decay, k=mid_channels)
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                weight_attr=weight_attr2,
                bias_attr=bias_attr2)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.embeder = Embedding(80, in_channels)

    def forward(self, x, targets=None):
        return_dict = {}
        embedding_vectors = self.embeder(x)
        if self.mid_channels is None:
            predicts = self.fc(x)
            return_dict['rec_pred'] = predicts
            return_dict['embedding_vectors'] = embedding_vectors
        else:
            predicts = self.fc1(x)
            predicts = self.fc2(predicts)
            
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
            return_dict['rec_pred'] = predicts
        return return_dict
