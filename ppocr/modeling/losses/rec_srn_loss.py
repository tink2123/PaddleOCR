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

import paddle
from paddle import nn


class SRNLoss(nn.Layer):
    def __init__(self, params):
        super(SRNLoss, self).__init__()
        self.char_num = params['char_num']
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, predicts, others):
        predict = predicts['predict']
        word_predict = predicts['word_out']
        gsrm_predict = predict['gsrm_out']
        label = others['label']
        lbl_weight = others['lbl_weight']

        casted_label = paddle.cast(x=label, dtype='int64')
        cost_word = self.loss_func(word_predict, label=casted_label)
        cost_gsrm = self.loss_func(gsrm_predict, label=casted_label)
        cost_vsfd = self.loss_func(predict, label=casted_label)

        sum_cost = paddle.sum([cost_word, cost_vsfd * 2.0, cost_gsrm * 0.15])

        return [sum_cost, cost_vsfd, cost_word]
